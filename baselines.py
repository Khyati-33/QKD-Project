"""Routing baselines used for cloning and evaluation."""
from __future__ import annotations

import random

import networkx as nx
import numpy as np


BASELINE_NAMES = ("Random", "Dijkstra-km", "BFS-hop", "Max-SKR")


def choose_baseline_action(name: str, env, observation, rng: random.Random | None = None) -> int:
    """Return a neighbor slot. Physics validity is intentionally left to the env."""
    rng = rng or random.Random()
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; expected {BASELINE_NAMES}")
    current = env.node_names[int(observation["current_node"])]
    destination = env.node_names[int(observation["destination_node"])]
    neighbors = list(env.graph.neighbors(current))
    if name == "Random":
        return rng.randrange(len(neighbors))
    if name in {"Dijkstra-km", "BFS-hop"}:
        weight = "distance_km" if name == "Dijkstra-km" else None
        try:
            path = nx.shortest_path(env.graph, current, destination, weight=weight)
            return neighbors.index(path[1]) if len(path) > 1 else 0
        except (nx.NetworkXNoPath, ValueError):
            pass
    if name == "Max-SKR":
        scores = observation["edge_features"][:len(neighbors), 1]
        return int(np.argmax(scores))
    # If a dynamic graph is externally disconnected, take the km-nearest node.
    return min(range(len(neighbors)), key=lambda i: env.graph[current][neighbors[i]]["distance_km"])


class BaselinePolicy:
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.rng = random.Random(seed)

    def act(self, observation, env) -> int:
        return choose_baseline_action(self.name, env, observation, self.rng)
