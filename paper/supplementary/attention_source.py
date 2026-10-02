"""Shared attention head and numerically correct action masking."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class NoValidActionError(RuntimeError):
    """Raised when a single state has no physically valid neighbor."""


def apply_hard_mask(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mask invalid actions with true negative infinity, preserving all-masked rows."""
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~valid_mask, -torch.inf)


def all_masked_rows(logits: torch.Tensor) -> torch.Tensor:
    """Return a boolean per-row indicator for rows containing only negative infinity."""
    if logits.ndim == 1:
        return torch.isneginf(logits).all().reshape(1)
    return torch.isneginf(logits).all(dim=-1)


def safe_batched_logits(masked_logits: torch.Tensor) -> torch.Tensor:
    """Replace fully masked batch rows by uniform zero logits before Categorical."""
    if masked_logits.ndim == 1:
        masked_logits = masked_logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    dead = all_masked_rows(masked_logits)
    safe = masked_logits.clone()
    if dead.any():
        safe[dead] = 0.0
    return safe.squeeze(0) if squeeze else safe


def make_single_distribution(masked_logits: torch.Tensor) -> Categorical:
    """Single-state path explicitly rejects all-masked logits for caller fallback."""
    if masked_logits.ndim != 1:
        raise ValueError("single distribution expects a one-dimensional logit row")
    if bool(all_masked_rows(masked_logits)[0]):
        raise NoValidActionError("all neighbors are masked; route to no-valid-options fallback")
    return Categorical(logits=masked_logits)


def make_batched_distribution(masked_logits: torch.Tensor) -> Categorical:
    """Batch path repairs all-masked rows to a uniform distribution explicitly."""
    if masked_logits.ndim == 1:
        masked_logits = masked_logits.unsqueeze(0)
    return Categorical(logits=safe_batched_logits(masked_logits))


class QKDAttentionHead(nn.Module):
    """Shared policy decision head that attends to raw edge and node features."""
    def __init__(self, hidden_dim: int, edge_feature_dim: int = 9,
                 progress_prior_scale: float = 2.0,
                 skr_prior_scale: float = 0.5):
        super().__init__()
        self.progress_prior_scale = float(progress_prior_scale)
        self.skr_prior_scale = float(skr_prior_scale)
        self.query = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.key = nn.Sequential(nn.Linear(hidden_dim + edge_feature_dim, hidden_dim), nn.Tanh())
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, current_embedding: torch.Tensor, destination_embedding: torch.Tensor,
                neighbor_embeddings: torch.Tensor, edge_features: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        query = self.query(torch.cat((current_embedding, destination_embedding), dim=-1))
        keys = self.key(torch.cat((neighbor_embeddings, edge_features), dim=-1))
        logits = self.score(torch.tanh(keys + query.unsqueeze(-2))).squeeze(-1)
        # Preserve the distance-to-destination potential: among otherwise
        # comparable actions, a long edge that makes more immediate progress
        # can be preferred over entering a multi-hop detour.
        logits = logits + self.progress_prior_scale * edge_features[..., 8]
        # SKR values span orders of magnitude and the learned attention alone
        # learned an inverse preference in the controlled link-choice test.
        # Keep a monotone log-rate utility skip connection so higher positive
        # key rates receive a direct, scale-stable advantage. Route progress,
        # feasibility masking, and learned context can still affect the choice.
        skr = edge_features[..., 1].clamp_min(1e-6)
        logits = logits + self.skr_prior_scale * torch.log(skr)
        return apply_hard_mask(logits, valid_mask)
