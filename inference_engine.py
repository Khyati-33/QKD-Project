"""Reusable CPU inference utilities for deployment benchmarks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from models import stack_observations


_LOCAL_TOPOLOGY_CACHE: dict[bytes, dict[int, set[int]]] = {}


def configure_cpu(*, threads: int = 8, interop_threads: int = 2) -> None:
    """Configure bounded CPU parallelism before model execution starts."""
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except RuntimeError:
        # PyTorch permits changing inter-op threads only before parallel work.
        pass


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize supported linear layers while retaining graph operations in FP32."""
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8).eval()


@torch.inference_mode()
def act(model: torch.nn.Module, observation: Mapping[str, Any]) -> int:
    """Run one deterministic action with autograd fully disabled."""
    action, _, _, _ = model.act(observation, deterministic=True)
    return int(action)


@torch.inference_mode()
def batched_actions(model: torch.nn.Module,
                    observations: Sequence[Mapping[str, Any]]) -> list[int]:
    """Run independent routing decisions together for higher throughput."""
    batch = stack_observations(observations)
    logits, _ = model(batch)
    return logits.argmax(dim=-1).to(device="cpu").tolist()


def localize_observation(observation: Mapping[str, Any], *, hops: int = 2) -> dict[str, Any]:
    """Keep a small current-node neighborhood for inference-only GNN execution.

    The current node, destination, all legal next-hop candidates, and their
    ``hops``-hop support neighborhood are retained. Node indices are remapped
    locally while candidate slots stay in their original order.
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
    node_features = np.asarray(observation["node_features"])
    edge_index = np.asarray(observation["edge_index"], dtype=np.int64)
    edge_attr = np.asarray(observation["edge_attr"])
    current = int(np.asarray(observation["current_node"]).reshape(-1)[0])
    destination = int(np.asarray(observation["destination_node"]).reshape(-1)[0])
    neighbor_indices = np.asarray(observation["neighbor_indices"], dtype=np.int64).copy()
    neighbor_count = int(np.asarray(observation["neighbor_count"]).reshape(-1)[0])
    selected = {current, destination}
    frontier = {current}
    topology_key = edge_index.tobytes()
    adjacency = _LOCAL_TOPOLOGY_CACHE.get(topology_key)
    if adjacency is None:
        adjacency = {}
        for u, v in edge_index.T:
            adjacency.setdefault(int(u), set()).add(int(v))
        if len(_LOCAL_TOPOLOGY_CACHE) >= 4:
            _LOCAL_TOPOLOGY_CACHE.pop(next(iter(_LOCAL_TOPOLOGY_CACHE)))
        _LOCAL_TOPOLOGY_CACHE[topology_key] = adjacency
    for _ in range(hops):
        frontier = {v for u in frontier for v in adjacency.get(u, ())}
        selected.update(frontier)
    selected.update(int(v) for v in neighbor_indices[:neighbor_count] if v >= 0)
    ordered = sorted(selected)
    remap = {old: new for new, old in enumerate(ordered)}
    edge_keep = np.asarray([(int(u) in remap and int(v) in remap)
                            for u, v in edge_index.T], dtype=bool)
    localized = dict(observation)
    localized["node_features"] = node_features[ordered]
    localized["edge_index"] = np.asarray(
        [[remap[int(u)], remap[int(v)]] for u, v in edge_index[:, edge_keep].T],
        dtype=np.int64).T
    localized["edge_attr"] = edge_attr[edge_keep]
    localized["current_node"] = np.asarray(remap[current], dtype=np.int64)
    localized["destination_node"] = np.asarray(remap[destination], dtype=np.int64)
    localized["neighbor_indices"] = np.asarray(
        [remap.get(int(v), -1) if int(v) >= 0 else -1 for v in neighbor_indices],
        dtype=np.int64)
    return localized
