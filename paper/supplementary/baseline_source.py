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
        remaining = nx.single_source_dijkstra_path_length(
            env.graph, destination, weight=weight)
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
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.rng = random.Random(seed)

    def act(self, observation, env) -> int:
        return choose_baseline_action(self.name, env, observation, self.rng)
