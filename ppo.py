"""PPO rollout storage, updates, checkpointing, and resumable training."""
from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from models import stack_observations
from qkd_attention import make_batched_distribution


LEARNING_RATE = 3e-5
CLIP_EPSILON = 0.1
INITIAL_ENTROPY_COEF = 0.003
BC_ENTROPY_COEF = 0.02
CRITIC_WARMUP_EPOCHS = 5


@dataclass
class Transition:
    observation: dict[str, np.ndarray]
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    action_valid: bool
    destination_node: int
    state_scalars: np.ndarray


class RolloutBuffer:
    """Per-transition immutable state snapshots, including destination/scalars."""
    def __init__(self):
        self.transitions: list[Transition] = []

    def add(self, observation: dict[str, Any], action: int, old_log_prob: float,
            value: float, reward: float, done: bool, action_valid: bool) -> None:
        snap = {k: np.array(v, copy=True) if isinstance(v, (np.ndarray, list)) else v
                for k, v in observation.items()}
        self.transitions.append(Transition(
            observation=snap, action=int(max(0, action)), old_log_prob=float(old_log_prob),
            value=float(value), reward=float(reward), done=bool(done),
            action_valid=bool(action_valid),
            destination_node=int(observation["destination_node"]),
            state_scalars=np.array(observation["state_scalars"], dtype=np.float32, copy=True),
        ))

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()


class RunningRewardNormalizer:
    """Running discounted-return scale; training updates use raw rewards in logs."""
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0):
        self.gamma, self.epsilon, self.clip = float(gamma), float(epsilon), float(clip)
        self.count, self.mean, self.m2 = 1e-4, 0.0, 1e-4
        self.discounted_return = 0.0

    def normalize(self, reward: float, done: bool = False) -> float:
        self.discounted_return = self.gamma * self.discounted_return + float(reward)
        self.count += 1.0
        delta = self.discounted_return - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (self.discounted_return - self.mean)
        scale = max(self.m2 / self.count, self.epsilon) ** 0.5
        normalized = float(np.clip(float(reward) / scale, -self.clip, self.clip))
        if done:
            self.discounted_return = 0.0
        return normalized

    def state_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in
                ("gamma", "epsilon", "clip", "count", "mean", "m2", "discounted_return")}

    def load_state_dict(self, state: dict[str, float]) -> None:
        for key in ("gamma", "epsilon", "clip", "count", "mean", "m2", "discounted_return"):
            if key in state:
                setattr(self, key, float(state[key]))


def _batch_from_transitions(transitions: list[Transition], device: torch.device):
    observations = [copy.deepcopy(t.observation) for t in transitions]
    # Reassert collection-time destination and scalar snapshots explicitly.
    for obs, transition in zip(observations, transitions):
        obs["destination_node"] = transition.destination_node
        obs["state_scalars"] = transition.state_scalars
    batch = stack_observations(observations)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                last_value: float, gamma: float = 0.99, lam: float = 0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = float(last_value)
    for i in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[i] = gae
        next_value = values[i]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def ppo_update(model: nn.Module, optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray,
               *, epochs: int = 4, batch_size: int = 64, clip_epsilon: float = CLIP_EPSILON,
               entropy_coef: float = INITIAL_ENTROPY_COEF, value_coef: float = 0.5,
               critic_only: bool = False, device: torch.device | None = None,
               value_clip_epsilon: float | None = 0.2,
               max_grad_norm: float = 0.5) -> dict[str, float]:
    device = device or next(model.parameters()).device
    transitions = buffer.transitions
    if not transitions:
        raise ValueError("cannot update from an empty rollout buffer")
    adv = torch.as_tensor(advantages, device=device, dtype=torch.float32)
    ret = torch.as_tensor(returns, device=device, dtype=torch.float32)
    old_log = torch.as_tensor([t.old_log_prob for t in transitions], device=device)
    actions = torch.as_tensor([t.action for t in transitions], device=device)
    trainable = torch.as_tensor([t.action_valid for t in transitions], device=device, dtype=torch.bool)
    if trainable.any():
        adv[trainable] = (adv[trainable] - adv[trainable].mean()) / (adv[trainable].std(unbiased=False) + 1e-8)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
    batches = 0
    original_requires_grad = {name: parameter.requires_grad
                              for name, parameter in model.named_parameters()}
    if critic_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("critic."))
    for _ in range(epochs):
        permutation = torch.randperm(len(transitions), device=device)
        for start in range(0, len(transitions), batch_size):
            indices = permutation[start:start + batch_size].tolist()
            selected = [transitions[i] for i in indices]
            obs_batch = _batch_from_transitions(selected, device)
            batch_actions = actions[indices]
            new_log, entropy, values = model.evaluate_actions(obs_batch, batch_actions)
            target_returns = ret[indices]
            if value_clip_epsilon is None:
                value_loss = torch.nn.functional.mse_loss(values, target_returns)
            else:
                old_values = torch.as_tensor([transitions[i].value for i in indices],
                                             device=device, dtype=torch.float32)
                clipped_values = old_values + torch.clamp(values - old_values,
                    -value_clip_epsilon, value_clip_epsilon)
                value_loss = 0.5 * torch.maximum(
                    (values - target_returns).square(),
                    (clipped_values - target_returns).square()).mean()
            policy_loss = torch.zeros((), device=device)
            mean_entropy = entropy.mean()
            if not critic_only:
                valid = trainable[indices]
                if valid.any():
                    ratio = torch.exp(new_log[valid] - old_log[indices][valid])
                    unclipped = ratio * adv[indices][valid]
                    clipped = torch.clamp(ratio, 1.0 - clip_epsilon,
                                          1.0 + clip_epsilon) * adv[indices][valid]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
            loss = value_coef * value_loss + policy_loss - (0.0 if critic_only else entropy_coef * mean_entropy)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(mean_entropy.detach())
            metrics["total_loss"] += float(loss.detach())
            batches += 1
    if critic_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
    return {k: v / max(batches, 1) for k, v in metrics.items()}


def save_checkpoint(path: str | Path, *, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    training_state: dict[str, Any] | None = None,
                    metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "training_state": training_state or {},
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    # On Windows a concurrent checkpoint reader may briefly hold a sharing
    # lock that prevents replacing the destination. Keep the write atomic,
    # but retry that transient lock rather than losing the training epoch.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("numpy_rng_state") is not None:
        np.random.set_state(payload["numpy_rng_state"])
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    return payload
