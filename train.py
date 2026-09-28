"""Behavior-cloning warm start and conservative PPO training entry point."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from models import GNNActorCritic, LSTMActorCritic, stack_observations
from ppo import (BC_ENTROPY_COEF, CLIP_EPSILON, CRITIC_WARMUP_EPOCHS,
                 INITIAL_ENTROPY_COEF, LEARNING_RATE, RolloutBuffer, RunningRewardNormalizer,
                 compute_gae, load_checkpoint, ppo_update, save_checkpoint)
from topology import CITIES


MODEL_TYPES = {"GNN": GNNActorCritic, "LSTM": LSTMActorCritic}


def _teacher_slot(env, observation) -> int:
    current = env.node_names[int(observation["current_node"])]
    destination = env.node_names[int(observation["destination_node"])]
    valid_slots = np.flatnonzero(observation["edge_valid_mask"])
    if not len(valid_slots):
        return -1
    nx = __import__("networkx")
    route_costs = {}
    hop_counts = {}
    for i in valid_slots:
        neighbor = env.node_names[int(observation["neighbor_indices"][i])]
        route_costs[int(i)] = (env.graph[current][neighbor]["distance_km"] +
                               env._km_dist_to_dest(neighbor))
        hop_counts[int(i)] = 1 + nx.shortest_path_length(
            env.graph, neighbor, destination)
    best_route_cost = min(route_costs.values())
    shortest_ties = [i for i, cost in route_costs.items()
                     if math.isclose(cost, best_route_cost, rel_tol=0.0, abs_tol=1e-6)]
    fewest_hops = min(hop_counts[i] for i in shortest_ties)
    hop_ties = [i for i in shortest_ties if hop_counts[i] == fewest_hops]
    # Prefer quality only when both total distance and route hop count tie.
    # This avoids teaching the agent to expand every fiber hop into an eight
    # segment FSO chain just because its per-link rate is larger.
    return int(max(hop_ties, key=lambda i: float(observation["edge_features"][i, 1])))


def _is_better_success_checkpoint(success: float, reward: float,
                                 avg_hops: float | None, *,
                                 best_success: float, best_reward: float,
                                 best_avg_hops: float) -> bool:
    """Rank route checkpoints by success first, reward then hop count."""
    if success != best_success:
        return success > best_success
    if reward != best_reward:
        return reward > best_reward
    hops = float(avg_hops) if avg_hops is not None else math.inf
    return hops < best_avg_hops


def behavior_clone(model, env, *, epochs: int = 15, episodes_per_epoch: int = 8,
                   entropy_coef: float = BC_ENTROPY_COEF, learning_rate: float = 1e-3,
                   checkpoint_dir: str | Path | None = None,
                   start_epoch: int = 0, optimizer=None,
                   randomize_endpoints: bool = False) -> tuple[Any, list[dict[str, float]]]:
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
    # BC is a supervised warm start and needs a useful step size. The main
    # training optimizer is deliberately initialized with PPO's much smaller
    # learning rates, so override its groups for BC (train_model restores PPO
    # rates immediately after this phase).
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    history = []
    original_pair = (env.source, env.destination)
    original_randomize = env.randomize_endpoints
    env.randomize_endpoints = False
    for epoch in range(start_epoch, epochs):
        states, labels = [], []
        for episode in range(episodes_per_epoch):
            seed = 10_000 + epoch * episodes_per_epoch + episode
            options = None
            if randomize_endpoints:
                pair_rng = random.Random(seed)
                source = pair_rng.choice(CITIES)
                destination = pair_rng.choice(tuple(
                    city for city in CITIES if city != source))
                options = {"source": source, "destination": destination}
            obs, _ = env.reset(seed=seed, options=options)
            for _ in range(env.max_steps):
                action = _teacher_slot(env, obs)
                if action < 0:
                    break
                states.append(obs)
                labels.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        if not states:
            metrics = {"bc_loss": 0.0, "bc_entropy": 0.0, "bc_match_rate": 0.0}
        else:
            batch = stack_observations(states)
            device = next(model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            target = torch.as_tensor(labels, device=device, dtype=torch.long)
            logits, _ = model(batch)
            dist = Categorical(logits=logits)
            loss = torch.nn.functional.nll_loss(dist.logits, target) - entropy_coef * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite behavior-cloning loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()
            metrics = {"bc_loss": float(loss.detach()), "bc_entropy": float(dist.entropy().mean().detach()),
                       "bc_match_rate": float((logits.argmax(-1) == target).float().mean().detach())}
        metrics.update(phase="bc", epoch=epoch + 1)
        history.append(metrics)
        if checkpoint_dir:
            save_checkpoint(Path(checkpoint_dir) / "latest.pt", model=model, optimizer=optimizer,
                            training_state={"phase": "bc", "bc_epoch": epoch + 1, "ppo_epoch": 0},
                            metadata={"model_name": type(model).__name__,
                                      "hidden_dim": model.hidden_dim})
    env.randomize_endpoints = False
    env.reset(seed=10_000 + epochs * episodes_per_epoch,
              options={"source": original_pair[0], "destination": original_pair[1]})
    env.randomize_endpoints = original_randomize
    return optimizer, history


def _collect_rollout(model, env, steps: int, obs=None, gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     reward_normalizer: RunningRewardNormalizer | None = None):
    buffer = RolloutBuffer()
    rewards = []
    zero_success = True
    successes = 0
    route = [env._current_node] if obs is not None else []
    if obs is None:
        obs, _ = env.reset()
        route = [env._current_node]
    for _ in range(steps):
        with torch.no_grad():
            action, log_prob, value, no_valid = model.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        learning_reward = (reward_normalizer.normalize(reward, done)
                           if reward_normalizer is not None else reward)
        buffer.add(obs, action, float(log_prob), float(value), learning_reward, done, not no_valid)
        rewards.append(float(reward))
        if info.get("success"):
            successes += 1; zero_success = False
        if info.get("node"):
            route.append(info["node"])
        obs = next_obs
        if done:
            obs, _ = env.reset()
            route = [env._current_node]
    with torch.no_grad():
        _, last_value = model(obs)
    adv, returns = compute_gae([t.reward for t in buffer.transitions],
                               [t.value for t in buffer.transitions],
                               [t.done for t in buffer.transitions], float(last_value.reshape(-1)[0]),
                               gamma=gamma, lam=gae_lambda)
    return buffer, adv, returns, obs, {
        "rollout_reward": float(np.sum(rewards)), "zero_success_epoch": float(zero_success),
        "rollout_successes": float(successes), "sample_path": route,
    }


def train_model(model_name: str, env, config: dict[str, Any], output_dir: str | Path,
                *, resume_from: str | Path | None = None,
                epoch_callback=None, stop_after_ppo_epochs: int | None = None,
                dropedge_probability: float = 0.0) -> dict[str, Any]:
    if model_name not in MODEL_TYPES:
        raise ValueError(f"model must be one of {tuple(MODEL_TYPES)}")
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cpu"))
    model_kwargs = {"hidden_dim": config.get("hidden_dim", 64)}
    if model_name == "GNN":
        model_kwargs["dropedge_probability"] = dropedge_probability
    model = MODEL_TYPES[model_name](**model_kwargs).to(device)
    actor_lr = float(config.get("actor_learning_rate", config.get("learning_rate", LEARNING_RATE)))
    critic_lr = float(config.get("critic_learning_rate", config.get("learning_rate", LEARNING_RATE)))
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("critic.")],
         "lr": actor_lr},
        {"params": list(model.critic.parameters()), "lr": critic_lr},
    ])
    reward_normalizer = (RunningRewardNormalizer(gamma=float(config.get("gamma", 0.99)))
                         if config.get("reward_normalization", False) else None)
    bc_epochs = int(config.get("bc_epochs", 15))
    ppo_epochs = int(config.get("ppo_epochs", 100))
    start_ppo_epoch = 0
    start_bc_epoch = 0
    history = []
    best_reward = -math.inf
    best_success = -math.inf
    best_success_reward = -math.inf
    best_success_hops = math.inf
    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, map_location=device)
        state = payload.get("training_state", {})
        history = list(state.get("history", []))
        if reward_normalizer is not None and state.get("reward_normalizer"):
            reward_normalizer.load_state_dict(state["reward_normalizer"])
        best_reward = float(state.get("best_reward", best_reward))
        best_success = float(state.get("best_success_rate", best_success))
        best_success_reward = float(state.get("best_success_reward", best_success_reward))
        best_success_hops = float(state.get("best_success_avg_hops", best_success_hops))
        if state.get("phase") == "bc":
            start_bc_epoch = int(state.get("bc_epoch", 0))
        else:
            start_bc_epoch = bc_epochs
            start_ppo_epoch = int(state.get("ppo_epoch", -1)) + 1
    if start_bc_epoch < bc_epochs:
        optimizer, bc_history = behavior_clone(
            model, env, epochs=bc_epochs, episodes_per_epoch=int(config.get("bc_episodes_per_epoch", 8)),
            entropy_coef=config.get("bc_entropy_coef", BC_ENTROPY_COEF),
            learning_rate=config.get("bc_learning_rate", 1e-3),
            checkpoint_dir=checkpoint_dir, start_epoch=start_bc_epoch, optimizer=optimizer,
            randomize_endpoints=bool(config.get("bc_randomize_endpoints", False)))
        history.extend(bc_history)
        critic_parameters = {id(p) for p in model.critic.parameters()}
        for group in optimizer.param_groups:
            is_critic = bool(group["params"]) and id(group["params"][0]) in critic_parameters
            group["lr"] = critic_lr if is_critic else actor_lr
    obs, _ = env.reset(seed=int(config.get("seed", 1234)) + start_ppo_epoch)
    best_reward_path = checkpoint_dir / "best_by_reward.pt"
    best_success_path = checkpoint_dir / "best_by_success_rate.pt"
    latest_path = checkpoint_dir / "latest.pt"
    for epoch in range(start_ppo_epoch, ppo_epochs):
        if hasattr(model, "enable_encoder_cache"):
            model.enable_encoder_cache(True)
        buffer, adv, returns, obs, rollout = _collect_rollout(
            model, env, int(config.get("steps_per_epoch", 144)), obs,
            gamma=float(config.get("gamma", 0.99)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            reward_normalizer=reward_normalizer)
        if hasattr(model, "enable_encoder_cache"):
            model.enable_encoder_cache(False)
        progress = epoch / max(ppo_epochs - 1, 1)
        entropy_coef = float(config.get("entropy_start", INITIAL_ENTROPY_COEF)) * (1.0 - progress)
        update_metrics = ppo_update(
            model, optimizer, buffer, adv, returns,
            epochs=int(config.get("update_epochs", 4)),
            batch_size=int(config.get("batch_size", 64)),
            clip_epsilon=float(config.get("clip_epsilon", CLIP_EPSILON)),
            entropy_coef=entropy_coef,
            critic_only=epoch < int(config.get("critic_warmup_epochs", CRITIC_WARMUP_EPOCHS)),
            value_clip_epsilon=config.get("value_clip_epsilon", 0.2),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
            device=device,
        )
        if hasattr(model, "clear_encoder_cache"):
            model.clear_encoder_cache()
        metrics = {"phase": "ppo", "epoch": epoch + 1, **rollout, **update_metrics,
                   "entropy_coef": entropy_coef}
        history.append(metrics)
        evaluation = epoch_callback(model, epoch + 1) if epoch_callback else {}
        metrics.update(evaluation or {})
        reward_score = float((evaluation or {}).get("overall_reward", rollout["rollout_reward"]))
        success_score = float((evaluation or {}).get("success_rate", rollout["rollout_successes"]))
        avg_hops_score = (evaluation or {}).get("avg_hops_success")
        new_best_reward = reward_score > best_reward
        new_best_success = _is_better_success_checkpoint(
            success_score, reward_score, avg_hops_score,
            best_success=best_success, best_reward=best_success_reward,
            best_avg_hops=best_success_hops)
        if new_best_reward:
            best_reward = reward_score
        if new_best_success:
            best_success = success_score
            best_success_reward = reward_score
            best_success_hops = (float(avg_hops_score) if avg_hops_score is not None
                                 else math.inf)
        latest_state = {"phase": "ppo", "bc_epoch": bc_epochs, "ppo_epoch": epoch,
                        "history": history, "best_reward": best_reward,
                        "best_success_rate": best_success,
                        "best_success_reward": best_success_reward,
                        "best_success_avg_hops": best_success_hops,
                        "reward_normalizer": (reward_normalizer.state_dict()
                            if reward_normalizer is not None else None)}
        save_checkpoint(latest_path, model=model, optimizer=optimizer,
                        training_state=latest_state,
                        metadata={"model_name": model_name, "hidden_dim": model.hidden_dim})
        if new_best_reward:
            save_checkpoint(best_reward_path, model=model, optimizer=optimizer,
                            training_state=latest_state,
                            metadata={"model_name": model_name, "hidden_dim": model.hidden_dim})
        if new_best_success:
            save_checkpoint(best_success_path, model=model, optimizer=optimizer,
                            training_state=latest_state,
                            metadata={"model_name": model_name, "hidden_dim": model.hidden_dim})
        if stop_after_ppo_epochs is not None and epoch - start_ppo_epoch + 1 >= stop_after_ppo_epochs:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{model_name.lower()}_training_metrics.json").open("w", encoding="utf8") as f:
        json.dump(history, f, indent=2, default=str)
    return {"model": model, "optimizer": optimizer, "history": history,
            "checkpoints": {"latest": latest_path, "best_by_reward": best_reward_path,
                            "best_by_success_rate": best_success_path}}


def main():
    import argparse
    import yaml
    from qkd_env import QKDRoutingEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_TYPES, default="LSTM")
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--output", default="experiments/manual")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
    environment_cfg = cfg.get("environment", cfg)
    env = QKDRoutingEnv(season=environment_cfg.get("season", "normal"),
                        use_case=environment_cfg.get("use_case", "standard"),
                        node_feature_mode=environment_cfg.get("node_feature_mode", "geographic"),
                        time_of_day_hours=environment_cfg.get("time_of_day_hours", 12.0),
                        dt_seconds=environment_cfg.get("dt_seconds", 300.0),
                        time_jitter_hours=environment_cfg.get("time_jitter_hours", 0.0),
                        max_steps=environment_cfg.get("max_steps", 100),
                        disabled_reward_terms=set(environment_cfg.get("disabled_reward_terms", [])),
                        randomize_endpoints=environment_cfg.get("randomize_endpoints", False))
    train_model(args.model, env, cfg, args.output, resume_from=args.resume)


if __name__ == "__main__":
    main()
