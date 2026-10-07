"""Routing baselines used for cloning and evaluation."""
from __future__ import annotations

import random

import networkx as nx
import numpy as np


BASELINE_NAMES = ("Random", "Dijkstra-km", "BFS-hop", "Max-SKR")


def choose_baseline_action(name: str, env, observation, rng: random.Random | None = None) -> int:
    """Return a feasible neighbor slot using the observation's action ordering."""
    rng = rng or random.Random()
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; expected {BASELINE_NAMES}")
    current = env.node_names[int(observation["current_node"])]
    destination = env.node_names[int(observation["destination_node"])]
    neighbors = list(env.graph.neighbors(current))
    # Action slots are defined by _candidate_mask(), including its ordering and
    # physical feasibility mask. Never let a baseline rely on env fallback.
    slots = [i for i, valid in enumerate(observation["edge_valid_mask"][:len(neighbors)])
             if valid]
    if not slots:
        return -1
    if name == "Random":
        return rng.choice(slots)
    if name in {"Dijkstra-km", "BFS-hop"}:
        weight = "distance_km" if name == "Dijkstra-km" else None
        # The topology and destination do not change during an episode, so
        # these distances are static. Cache them on the episode environment
        # instead of rerunning Dijkstra at every routing step.
        cache = getattr(env, "_baseline_distance_cache", None)
        if cache is None:
            cache = {}
            env._baseline_distance_cache = cache
        cache_key = (name, destination)
        if cache_key not in cache:
            cache[cache_key] = nx.single_source_dijkstra_path_length(
                env.graph, destination, weight=weight)
        remaining = cache[cache_key]
        def total_cost(i):
            edge_cost = (env.graph[current][neighbors[i]]["distance_km"]
                         if weight else 1.0)
            return (edge_cost + remaining.get(neighbors[i], float("inf")),
                    edge_cost, i)
        return min(slots, key=total_cost)
    if name == "Max-SKR":
        scores = observation["edge_features"][:len(neighbors), 1]
        return max(slots, key=lambda i: (float(scores[i]), -i))
    return min(slots, key=lambda i: env.graph[current][neighbors[i]]["distance_km"])


class BaselinePolicy:
    def __init__(self, name: str, seed: int = 0,
                 weighted_sp_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.name = name
        self.rng = random.Random(seed)
        self.weighted_sp_weights = tuple(float(w) for w in weighted_sp_weights)

    def act(self, observation, env) -> int:
        if self.name == "Weighted-SP":
            return choose_weighted_sp_action(env, observation, self.weighted_sp_weights)
        return choose_baseline_action(self.name, env, observation, self.rng)


def choose_weighted_sp_action(env, observation,
                              weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> int:
    """Receding-horizon shortest path on current normalized physical/resource costs.

    The three tunable weights multiply QBER, negative log normalized rate,
    and key-pool emptiness. Distance has fixed coefficient one.
    """
    w_q, w_r, w_k = (float(value) for value in weights)
    if min(w_q, w_r, w_k) < 0:
        raise ValueError("weighted-SP coefficients must be nonnegative")
    current = env.node_names[int(observation["current_node"])]
    destination = env.node_names[int(observation["destination_node"])]
    neighbors = list(env.graph.neighbors(current))
    slots = [i for i, valid in enumerate(observation["edge_valid_mask"][:len(neighbors)])
             if valid]
    if not slots:
        return -1

    weighted = nx.Graph()
    weighted.add_nodes_from(env.graph.nodes(data=True))
    for u, v, attrs in env.graph.edges(data=True):
        key = env._edge_key(u, v)
        state = env._edge_states[key]
        # Future-hop lookahead omits links already known to be unusable. The
        # current first hop is still filtered by the full path-aware mask.
        if (float(state["qber"]) >= float(env.qber_hard) or
                float(state["skr"]) <= 0.0 or float(env._key_pool[key]) <= 0.0):
            continue
        distance = float(attrs["distance_km"]) / max(float(env.max_edge_km), 1.0)
        q_fraction = float(state["qber"]) / max(float(env.qber_hard), 1e-12)
        rate_cost = -np.log(max(float(state["skr"]), 1e-6))
        pool_cost = 1.0 - float(env._key_pool[key])
        cost = distance + w_q * q_fraction + w_r * rate_cost + w_k * pool_cost
        weighted.add_edge(u, v, cost=float(cost))
    remaining = nx.single_source_dijkstra_path_length(weighted, destination, weight="cost")
    selected = min(slots, key=lambda i: (
        weighted[current][neighbors[i]]["cost"] + remaining.get(neighbors[i], float("inf")),
        i))
    env._weighted_sp_realized_cost = getattr(env, "_weighted_sp_realized_cost", 0.0) + \
        weighted[current][neighbors[selected]]["cost"]
    return selected
