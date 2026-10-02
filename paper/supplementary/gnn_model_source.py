"""Graph-aware GNN and deliberately graph-blind LSTM actor-critic models."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge
from physics import QBER_HARD

from qkd_attention import (NoValidActionError, QKDAttentionHead,
                           make_batched_distribution, make_single_distribution)


def _tensor(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype or value.dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def stack_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack states from the same topology while keeping graph indices shared."""
    if not observations:
        raise ValueError("at least one observation is required")
    keys = observations[0].keys()
    result = {}
    for key in keys:
        if key == "edge_index":
            continue
        result[key] = torch.as_tensor(np.stack([np.asarray(o[key]) for o in observations]))
    result["edge_index"] = _tensor(observations[0]["edge_index"], torch.device("cpu"), torch.long)
    return result


class LSTMNodeProcessor(nn.Module):
    """Single-layer sequence LSTM with no access to graph edges."""
    def __init__(self, node_feature_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(node_feature_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        outputs, _ = self.lstm(node_features)
        return outputs


class _LearnedMessageLayer(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.message_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim))
        self.gru_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_valid: torch.Tensor) -> torch.Tensor:
        aggregate = self.propagate(edge_index, x=x, edge_valid=edge_valid,
                                   size=(x.size(0), x.size(0)))
        return self.gru_update(aggregate, x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_valid: torch.Tensor) -> torch.Tensor:
        message = self.message_mlp(torch.cat((x_i, x_j), dim=-1))
        return message * edge_valid.to(message.dtype).reshape(-1, 1)


class GNNEncoder(nn.Module):
    """Three learned message-passing layers with GRU state updates."""
    def __init__(self, node_feature_dim: int, hidden_dim: int,
                 dropedge_probability: float = 0.0):
        super().__init__()
        if not 0.0 <= dropedge_probability < 1.0:
            raise ValueError("dropedge_probability must be in [0, 1)")
        self.dropedge_probability = float(dropedge_probability)
        self.input = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([_LearnedMessageLayer(hidden_dim) for _ in range(3)])

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input(node_features))
        edge_valid = (edge_attr[:, 0] < QBER_HARD) & (edge_attr[:, 1] > 0.0)
        edge_index = edge_index[:, edge_valid]
        edge_weights = edge_valid[edge_valid]
        if self.training and self.dropedge_probability:
            # edge_index contains both directions for each physical link;
            # force_undirected removes both at once and preserves symmetry.
            edge_index, edge_mask = dropout_edge(edge_index, p=self.dropedge_probability,
                                                  force_undirected=True, training=True)
            edge_weights = edge_weights[edge_mask]
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_weights))
        return x


class ActorCriticBase(nn.Module):
    def __init__(self, node_feature_dim: int = 5, edge_feature_dim: int = 9,
                 hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.decision_head = QKDAttentionHead(hidden_dim, edge_feature_dim)
        self.critic = nn.Sequential(nn.Linear(3 * hidden_dim + 2, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))

    def _forward_encoded(self, node_embeddings: torch.Tensor,
                         obs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device = node_embeddings.device
        current = _tensor(obs["current_node"], device, torch.long).reshape(-1)
        destination = _tensor(obs["destination_node"], device, torch.long).reshape(-1)
        if node_embeddings.ndim == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
        batch_ids = torch.arange(node_embeddings.size(0), device=device)
        current_embedding = node_embeddings[batch_ids, current]
        destination_embedding = node_embeddings[batch_ids, destination]
        neighbors = _tensor(obs["neighbor_indices"], device, torch.long)
        if neighbors.ndim == 1:
            neighbors = neighbors.unsqueeze(0)
        neighbor_embeddings = node_embeddings[batch_ids[:, None], neighbors.clamp_min(0)]
        edge_features = _tensor(obs["edge_features"], device, torch.float32)
        if edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0)
        mask = _tensor(obs["edge_valid_mask"], device, torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        logits = self.decision_head(current_embedding, destination_embedding,
                                    neighbor_embeddings, edge_features, mask)
        pooled = node_embeddings.mean(dim=1)
        scalars = _tensor(obs["state_scalars"], device, torch.float32)
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(0)
        value = self.critic(torch.cat((pooled, current_embedding, destination_embedding,
                                       scalars), dim=-1)).squeeze(-1)
        return logits, value

    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False):
        logits, value = self(observation)
        row = logits[0] if logits.ndim == 2 and logits.size(0) == 1 else logits
        if row.ndim != 1:
            raise ValueError("act() expects one observation")
        try:
            distribution = make_single_distribution(row)
        except NoValidActionError:
            return -1, torch.zeros((), device=row.device), value.reshape(-1)[0], True
        action = row.argmax() if deterministic else distribution.sample()
        return int(action.item()), distribution.log_prob(action), value.reshape(-1)[0], False

    def evaluate_actions(self, observations: Mapping[str, Any], actions: torch.Tensor):
        logits, values = self(observations)
        distribution = make_batched_distribution(logits)
        actions = actions.to(device=logits.device, dtype=torch.long).reshape(-1)
        return distribution.log_prob(actions), distribution.entropy(), values


class LSTMActorCritic(ActorCriticBase):
    def __init__(self, node_feature_dim: int = 5, edge_feature_dim: int = 9,
                 hidden_dim: int = 64):
        super().__init__(node_feature_dim, edge_feature_dim, hidden_dim)
        self.encoder = LSTMNodeProcessor(node_feature_dim, hidden_dim)

    def forward(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        node_features = _tensor(observation["node_features"], device, torch.float32)
        embeddings = self.encoder(node_features)
        return self._forward_encoded(embeddings, observation)


class GNNActorCritic(ActorCriticBase):
    def __init__(self, node_feature_dim: int = 5, edge_feature_dim: int = 9,
                 hidden_dim: int = 64, dropedge_probability: float = 0.0):
        super().__init__(node_feature_dim, edge_feature_dim, hidden_dim)
        self.encoder = GNNEncoder(node_feature_dim, hidden_dim, dropedge_probability)
        self._encoder_cache_enabled = False
        self._encoder_cache: dict[tuple, torch.Tensor] = {}

    def enable_encoder_cache(self, enabled: bool = True) -> None:
        self._encoder_cache_enabled = bool(enabled)
        if not enabled:
            self._encoder_cache.clear()

    def clear_encoder_cache(self) -> None:
        self._encoder_cache.clear()

    def _encode_cached(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: torch.Tensor) -> torch.Tensor:
        if (not self._encoder_cache_enabled or torch.is_grad_enabled() or
                node_features.ndim != 2):
            return self.encoder(node_features, edge_index, edge_attr)
        valid = (edge_attr[:, 0] < QBER_HARD) & (edge_attr[:, 1] > 0.0)
        key = (tuple(p._version for p in self.encoder.parameters()),
               node_features.detach().cpu().numpy().tobytes(),
               edge_index.detach().cpu().numpy().tobytes(),
               valid.detach().cpu().numpy().tobytes())
        cached = self._encoder_cache.get(key)
        if cached is None:
            cached = self.encoder(node_features, edge_index, edge_attr).detach()
            if len(self._encoder_cache) >= 32:
                self._encoder_cache.pop(next(iter(self._encoder_cache)))
            self._encoder_cache[key] = cached
        return cached

    def forward(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        node_features = _tensor(observation["node_features"], device, torch.float32)
        edge_index = _tensor(observation["edge_index"], device, torch.long)
        edge_attr = _tensor(observation["edge_attr"], device, torch.float32)
        if node_features.ndim == 2:
            return self._forward_encoded(
                self._encode_cached(node_features, edge_index, edge_attr), observation)
        # Disjoint graph batch: concatenate node matrices and offset one copy
        # of the shared topology per observation, then run message passing once.
        batch_size, node_count, feature_count = node_features.shape
        offsets = torch.arange(batch_size, device=device, dtype=edge_index.dtype) * node_count
        batched_edge_index = (edge_index.unsqueeze(0) + offsets[:, None, None])
        batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, -1)
        batched_edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])
        embeddings = self.encoder(node_features.reshape(batch_size * node_count, feature_count),
                                  batched_edge_index, batched_edge_attr).reshape(
                                      batch_size, node_count, -1)
        return self._forward_encoded(embeddings, observation)
