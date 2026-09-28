"""Gymnasium routing environment with link physics and trust-chain state."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from physics import (FIBER_OUTAGE_PROB, FSO_TURBULENCE_CORRELATION, QBER_HARD,
                     SEASONS, chain_parity_error, edge_is_valid,
                     sample_correlated_fso_cn2, sample_link_state)
from topology import (CITIES, DEFAULT_DESTINATION, DEFAULT_SOURCE, build_topology)


# Reward scales are centralized for transparency and ablation.
REWARD_WEIGHTS = {
    "progress": 0.05, "margin": 1.0, "skr": 0.2, "pool": 0.5,
    "latency": 0.2, "energy": 0.1, "congestion": 0.1,
}
# The defence profile follows the latest notebook's relative priorities while
# retaining this repo's per-link, explicitly calibrated reward implementation.
REWARD_PROFILES = {
    "standard": {"progress": 0.05, "progress_unit": "km", "margin": 1.0,
                 "skr": 0.2, "pool": 0.5, "latency": 0.2,
                 "energy": 0.1, "congestion": 0.1, "arrival": 20.0},
    "defence": {"progress": 0.2, "progress_unit": "normalized_km", "margin": 0.8,
                "skr": 1.2, "pool": 0.4, "latency": 0.4,
                "energy": 0.4, "congestion": 0.5, "arrival": 4.0},
}
STEP_COST = -0.10
REVISIT_PENALTY = -1.0
SWITCH_PENALTY = -0.25
C_SEC = -10.0
C_POOL_DEPLETED = -5.0
ARRIVAL_BONUS = 20.0
FAILURE_PENALTY = -5.0
BACKTRACK_PENALTY = -0.5
ALLOWED_ABLATIONS = {"margin", "skr", "pool", "latency", "energy", "congestion"}


def distance_normalized_skr_reward(skr: float, distance_km: float,
                                   max_edge_km: float, scale: float) -> float:
    """Return SKR utility proportional to distance, avoiding segment-count gain."""
    if max_edge_km <= 0 or distance_km < 0:
        raise ValueError("link distances must be nonnegative and max_edge_km positive")
    return (float(scale) * min(1.0, max(0.0, float(skr))) *
            float(distance_km) / float(max_edge_km))


def calibrate_norm_constants_over_pairs(graph: nx.Graph | None = None) -> dict[str, float]:
    """Calibrate latency/energy scales over every ordered city pair."""
    graph = graph or build_topology()
    distances = []
    for source in CITIES:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="distance_km")
        distances.extend(lengths[d] for d in CITIES if d != source)
    max_distance = max(distances)
    # 200,000 km/s propagation; normalizing by worst city pair avoids endpoint bias.
    return {"latency_s": max_distance / 200_000.0,
            "energy_units": max_distance * 1.0,
            "max_pair_distance_km": max_distance}


class QKDRoutingEnv(gym.Env):
    """One action selects a neighbor slot at the current node.

    Observation arrays are padded to the largest node degree. `edge_valid_mask`
    is the shared actor/environment candidate mask; edge feature columns are
    Per-candidate edge features are [QBER, SKR, FSO flag, normalized
    distance, projected chain error, key-pool level, link-switch flag,
    already-visited flag, normalized distance progress to destination]. Geographic or destination-relative node features
    are selected by configuration.
    """
    metadata = {"render_modes": []}

    def __init__(self, graph: nx.Graph | None = None, source: str = DEFAULT_SOURCE,
                 destination: str = DEFAULT_DESTINATION, season: str = "normal",
                 use_case: str = "standard", node_feature_mode: str = "geographic",
                 disabled_reward_terms: set[str] | None = None,
                 randomize_endpoints: bool = False,
                 fiber_outage_prob: float = FIBER_OUTAGE_PROB,
                 qber_hard: float = QBER_HARD,
                 reward_protection_overrides: dict[str, float] | None = None,
                 max_steps: int = 400, time_of_day_hours: float = 12.0,
                 dt_seconds: float = 300.0, time_jitter_hours: float = 0.0,
                 fso_temporal_correlation: float = FSO_TURBULENCE_CORRELATION):
        super().__init__()
        if season not in SEASONS:
            raise ValueError(f"season must be one of {SEASONS}")
        if use_case not in REWARD_PROFILES:
            raise ValueError(f"use_case must be one of {tuple(REWARD_PROFILES)}")
        if node_feature_mode not in {"geographic", "relative"}:
            raise ValueError("node_feature_mode must be 'geographic' or 'relative'")
        if source not in CITIES or destination not in CITIES or source == destination:
            raise ValueError("source and destination must be distinct configured cities")
        if not 0 <= fiber_outage_prob <= 1:
            raise ValueError("fiber_outage_prob must be in [0, 1]")
        self.graph = graph or build_topology()
        self.source, self.destination = source, destination
        self.season = season
        self.use_case = use_case
        self.reward_profile = REWARD_PROFILES[use_case]
        self.node_feature_mode = node_feature_mode
        self.randomize_endpoints = bool(randomize_endpoints)
        self.fiber_outage_prob = fiber_outage_prob
        self.qber_hard = float(qber_hard)
        self.reward_protections = {
            "step_cost": STEP_COST, "revisit": REVISIT_PENALTY,
            "switch": SWITCH_PENALTY, "security_penalty": C_SEC,
            "depletion_penalty": C_POOL_DEPLETED, "backtrack": BACKTRACK_PENALTY,
        }
        self.reward_protections.update(reward_protection_overrides or {})
        self.max_steps = max_steps
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        self.start_time_of_day_hours = float(time_of_day_hours) % 24.0
        self.time_of_day_hours = self.start_time_of_day_hours
        self.dt_seconds = float(dt_seconds)
        if time_jitter_hours < 0 or time_jitter_hours > 12:
            raise ValueError("time_jitter_hours must be between 0 and 12")
        self.time_jitter_hours = float(time_jitter_hours)
        if not 0.0 <= fso_temporal_correlation < 1.0:
            raise ValueError("fso_temporal_correlation must be in [0, 1)")
        self.fso_temporal_correlation = float(fso_temporal_correlation)
        self.disabled_reward_terms = set(disabled_reward_terms or ()) & ALLOWED_ABLATIONS
        self.node_names = list(self.graph.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        self.max_neighbors = max(dict(self.graph.degree()).values())
        self.max_edge_km = max(d["distance_km"] for _, _, d in self.graph.edges(data=True))
        # Bound progress normalization using configured city-pair routes. An
        # all-node-pairs search is needlessly expensive for the road-aligned
        # graph, which has several thousand candidate relays.
        self.norm_constants = calibrate_norm_constants_over_pairs(self.graph)
        self.longest_route_km = self.norm_constants["max_pair_distance_km"]
        self.action_space = spaces.Discrete(self.max_neighbors)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(-np.inf, np.inf, (len(self.node_names), 5), dtype=np.float32),
            "edge_index": spaces.Box(0, len(self.node_names) - 1, (2, self.graph.number_of_edges() * 2), dtype=np.int64),
            "edge_attr": spaces.Box(-np.inf, np.inf, (self.graph.number_of_edges() * 2, 4), dtype=np.float32),
            "neighbor_indices": spaces.Box(-1, len(self.node_names) - 1, (self.max_neighbors,), dtype=np.int64),
            # [QBER, SKR, FSO, distance, projected chain error, pool level,
            #  link-type switch indicator, already-visited indicator]
            "edge_features": spaces.Box(-np.inf, np.inf, (self.max_neighbors, 9), dtype=np.float32),
            "edge_valid_mask": spaces.MultiBinary(self.max_neighbors),
            "neighbor_count": spaces.Discrete(self.max_neighbors + 1),
            "current_node": spaces.Discrete(len(self.node_names)),
            "destination_node": spaces.Discrete(len(self.node_names)),
            "state_scalars": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        })
        self._rng = random.Random()
        self._edge_states: dict[frozenset, dict[str, Any]] = {}
        # Corridor outage is a persistent episode event. Redrawing it every
        # five-minute step can make an agent enter a detour and then have its
        # forward edges vanish midway through the relay chain.
        self._detour_outage_draws: dict[tuple[str, str], float] = {}
        self._fso_cn2_state: dict[tuple[str, str], float] = {}
        self._hop_qbers: list[float] = []
        self._visited: set[str] = set()
        self._key_pool: defaultdict[frozenset, float] = defaultdict(lambda: 1.0)
        self._previous_link_type: str | None = None
        self._current_node = source
        self._step_count = 0
        self._done = False
        self._destination_distances: dict[str, float] = {}
        self._refresh_link_states()

    def _edge_key(self, u: str, v: str) -> frozenset:
        return frozenset((u, v))

    def _refresh_link_states(self) -> None:
        self._edge_states = {}
        fso_cn2_draws = {}
        for _, _, attrs in self.graph.edges(data=True):
            if attrs["link_type"] != "fso":
                continue
            detour = attrs.get("detour_for")
            if detour in fso_cn2_draws:
                continue
            cn2, log_cn2 = sample_correlated_fso_cn2(
                self.season, self.time_of_day_hours, self._rng,
                self._fso_cn2_state.get(detour), self.fso_temporal_correlation)
            fso_cn2_draws[detour] = cn2
            self._fso_cn2_state[detour] = log_cn2
        # All links in a detour share an episode-scale corridor outage draw.
        for _, _, attrs in self.graph.edges(data=True):
            if attrs["link_type"] == "fso":
                detour = attrs.get("detour_for")
                if detour not in self._detour_outage_draws:
                    self._detour_outage_draws[detour] = self._rng.random()
        for u, v, attrs in self.graph.edges(data=True):
            state = sample_link_state(
                attrs["link_type"], attrs["distance_km"], self.season,
                self.time_of_day_hours, self._rng,
                fiber_outage_prob=self.fiber_outage_prob,
                qber_hard=self.qber_hard,
                outage_uniform=(self._detour_outage_draws.get(attrs.get("detour_for"))
                                if attrs["link_type"] == "fso" else None),
                      outage_distance_km=(attrs.get("outage_reference_distance_km")
                                          if attrs["link_type"] == "fso" else None),
                cn2_override=(fso_cn2_draws.get(attrs.get("detour_for"))
                              if attrs["link_type"] == "fso" else None),
            )
            self._edge_states[self._edge_key(u, v)] = dict(state)

    def _edge_valid(self, edge: tuple[str, str] | frozenset | dict[str, Any]) -> bool:
        """The one physical edge validity check shared by masking and fallback."""
        if isinstance(edge, dict):
            state = edge
        else:
            u, v = tuple(edge)
            state = self._edge_states[self._edge_key(u, v)]
        return edge_is_valid(float(state["qber"]), float(state["skr"]), self.qber_hard)

    def _projected_chain_error(self, u: str, v: str, qber: float | None = None) -> float:
        attrs = self.graph[u][v]
        q = float(self._edge_states[self._edge_key(u, v)]["qber"] if qber is None else qber)
        if self.graph.nodes[v].get("node_type") == "full_tn":
            return 0.0
        return chain_parity_error([*self._hop_qbers, q])

    def _candidate_mask(self, u: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        neighbors = list(self.graph.neighbors(u))
        features = np.zeros((self.max_neighbors, 9), dtype=np.float32)
        mask = np.zeros(self.max_neighbors, dtype=np.int8)
        indices = np.full(self.max_neighbors, -1, dtype=np.int64)
        for slot, v in enumerate(neighbors):
            attrs = self.graph[u][v]
            state = self._edge_states[self._edge_key(u, v)]
            projected = self._projected_chain_error(u, v)
            key = self._edge_key(u, v)
            link_type = attrs["link_type"]
            features[slot] = [state["qber"], state["skr"],
                              float(link_type == "fso"),
                              attrs["distance_km"] / self.max_edge_km, projected,
                              self._key_pool[key],
                              float(self._previous_link_type is not None and
                                    link_type != self._previous_link_type),
                              float(v in self._visited),
                              (self._km_dist_to_dest(u) - self._km_dist_to_dest(v)) /
                                  max(self.max_edge_km, 1.0)]
            indices[slot] = self.node_to_idx[v]
            mask[slot] = int(self._edge_valid((u, v)) and projected < self.qber_hard)
        return neighbors, features, mask, indices

    def _km_dist_to_dest(self, node: str) -> float:
        """Real Dijkstra-km distance to the current destination."""
        return float(self._destination_distances[node])

    def state_scalars(self) -> tuple[float, float]:
        remaining = self._km_dist_to_dest(self._current_node) / self.max_edge_km
        margin = (self.qber_hard - chain_parity_error(self._hop_qbers)) / self.qber_hard
        return float(remaining), float(margin)

    def _node_feature_matrix(self) -> np.ndarray:
        coords = [self.graph.nodes[n].get("pos", (0, 0)) for n in self.node_names]
        lat_min, lat_max = min(p[0] for p in coords), max(p[0] for p in coords)
        lon_min, lon_max = min(p[1] for p in coords), max(p[1] for p in coords)
        result = np.zeros((len(self.node_names), 5), dtype=np.float32)
        for i, n in enumerate(self.node_names):
            d = self.graph.nodes[n]
            if self.node_feature_mode == "relative":
                relative_km = self._destination_distances.get(n, self.longest_route_km)
                result[i] = [float(d.get("city", False)),
                             float(d.get("node_type") == "full_tn"),
                             float(d.get("node_type") == "stn"),
                             relative_km / max(self.longest_route_km, 1.0),
                             self.graph.degree[n] / max(self.max_neighbors, 1)]
            else:
                lat, lon = d.get("pos", (0, 0))
                result[i] = [float(d.get("city", False)),
                             float(d.get("node_type") == "full_tn"),
                             float(d.get("node_type") == "stn"),
                             (lat - lat_min) / max(lat_max - lat_min, 1e-8),
                             (lon - lon_min) / max(lon_max - lon_min, 1e-8)]
        return result

    def _get_obs(self) -> dict[str, Any]:
        neighbors, feats, mask, indices = self._candidate_mask(self._current_node)
        directed_edges, directed_attrs = [], []
        for u, v, attrs in self.graph.edges(data=True):
            state = self._edge_states[self._edge_key(u, v)]
            attr = [float(state["qber"]), float(state["skr"]),
                    float(attrs["link_type"] == "fso"), attrs["distance_km"] / self.max_edge_km]
            directed_edges.extend([[self.node_to_idx[u], self.node_to_idx[v]],
                                    [self.node_to_idx[v], self.node_to_idx[u]]])
            directed_attrs.extend([attr, attr])
        return {
            "node_features": self._node_feature_matrix(),
            "edge_index": np.asarray(directed_edges, dtype=np.int64).T,
            "edge_attr": np.asarray(directed_attrs, dtype=np.float32),
            "neighbor_indices": indices,
            "edge_features": feats,
            "edge_valid_mask": mask,
            "neighbor_count": len(neighbors),
            "current_node": self.node_to_idx[self._current_node],
            "destination_node": self.node_to_idx[self.destination],
            "state_scalars": np.asarray(self.state_scalars(), dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        if self.randomize_endpoints:
            choices = [c for c in CITIES]
            self.source = self._rng.choice(choices)
            self.destination = self._rng.choice([c for c in choices if c != self.source])
        elif options:
            self.source = options.get("source", self.source)
            self.destination = options.get("destination", self.destination)
        self._current_node = self.source
        jitter = self.np_random.uniform(-self.time_jitter_hours, self.time_jitter_hours) \
            if self.time_jitter_hours else 0.0
        self.time_of_day_hours = (self.start_time_of_day_hours + jitter) % 24.0
        self._step_count = 0
        self._done = False
        self._hop_qbers = []
        self._visited = {self.source}
        self._key_pool = defaultdict(lambda: 1.0)
        self._previous_link_type = None
        self._detour_outage_draws = {}
        self._fso_cn2_state = {}
        self._destination_distances = nx.single_source_dijkstra_path_length(
            self.graph, self.destination, weight="distance_km")
        self._refresh_link_states()
        return self._get_obs(), {"source": self.source, "destination": self.destination,
                                 "season": self.season,
                                 "time_of_day_hours": self.time_of_day_hours}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called after episode ended; call reset()")
        old_node = self._current_node
        neighbors, features, mask, _ = self._candidate_mask(old_node)
        valid_slots = [i for i, value in enumerate(mask[:len(neighbors)]) if value]
        fallback = False
        if not valid_slots:
            self._done = True
            return self._get_obs(), FAILURE_PENALTY, False, True, {
                "success": False, "failure": "no_valid_neighbor", "reward_terms": {}}
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= len(neighbors):
            action = valid_slots[0]
            fallback = True
        elif not mask[int(action)]:
            # Shared _edge_valid() powers _candidate_mask(), also used above.
            # On mixed 80 km fiber / 10 km FSO graphs, selecting the shortest
            # adjacent edge can send an agent back into the FSO chain it just
            # left. Fall forward by the remaining destination distance.
            action = min(valid_slots, key=lambda i: self._km_dist_to_dest(neighbors[i]))
            fallback = True
        next_node = neighbors[int(action)]
        edge_attrs = self.graph[old_node][next_node]
        state = self._edge_states[self._edge_key(old_node, next_node)]
        projected = self._projected_chain_error(old_node, next_node)
        old_distance = self._km_dist_to_dest(old_node)
        edge_key = self._edge_key(old_node, next_node)
        decision_time = self.time_of_day_hours
        old_pool = self._key_pool[edge_key]
        # A simple per-link buffer: QKD replenishes up to capacity; forwarding
        # consumes a fixed packet-equivalent amount each time the link is used.
        generated = min(0.05, max(0.0, float(state["skr"]) * 0.1))
        new_pool = min(1.0, old_pool + generated) - 0.1
        depleted = new_pool <= 0.0
        self._key_pool[edge_key] = max(0.0, new_pool)
        self._current_node = next_node
        self._step_count += 1
        revisit = next_node in self._visited
        self._visited.add(next_node)
        link_type = edge_attrs["link_type"]
        switched = self._previous_link_type is not None and link_type != self._previous_link_type
        self._previous_link_type = link_type

        if self.graph.nodes[next_node].get("node_type") == "full_tn":
            self._hop_qbers = []
            current_chain = 0.0
        else:
            self._hop_qbers.append(float(state["qber"]))
            current_chain = chain_parity_error(self._hop_qbers)
        security_breach = current_chain >= self.qber_hard
        new_distance = self._km_dist_to_dest(next_node)
        profile = self.reward_profile
        progress_distance = old_distance - new_distance
        if profile["progress_unit"] == "normalized_km":
            progress_distance /= self.max_edge_km
        progress_sign = (1.0 if progress_distance > 1e-9 else
                         -1.0 if progress_distance < -1e-9 else 0.0)

        raw = {
            "progress": profile["progress"] * progress_distance,
            "margin": progress_sign * profile["margin"] * max(0.0, (self.qber_hard - projected) / self.qber_hard),
            "skr": progress_sign * distance_normalized_skr_reward(
                float(state["skr"]), edge_attrs["distance_km"],
                self.max_edge_km, profile["skr"]),
            "pool_level": progress_sign * profile["pool"] * self._key_pool[edge_key],
            "pool_delta": progress_sign * profile["pool"] * (self._key_pool[edge_key] - old_pool),
            "latency": -profile["latency"] * (edge_attrs["distance_km"] / 200000.0)
                       / self.norm_constants["latency_s"],
            "energy": -profile["energy"] * edge_attrs["distance_km"]
                      / self.norm_constants["energy_units"],
            "congestion": -profile["congestion"] * max(0, self.graph.degree(old_node) - 2),
            "step_cost": self.reward_protections["step_cost"],
            "revisit": self.reward_protections["revisit"] if revisit else 0.0,
            "switch": self.reward_protections["switch"] if switched else 0.0,
            "backtrack": self.reward_protections["backtrack"] if progress_sign < 0.0 else 0.0,
            "arrival": profile["arrival"] if next_node == self.destination else 0.0,
        }
        for term in self.disabled_reward_terms:
            if term == "pool":
                raw["pool_level"] = 0.0
                raw["pool_delta"] = 0.0
            else:
                raw[term] = 0.0
        if security_breach:
            raw["security_penalty"] = self.reward_protections["security_penalty"]
        if depleted:
            raw["depletion_penalty"] = self.reward_protections["depletion_penalty"]
        total = float(sum(raw.values()))
        terminated = next_node == self.destination or security_breach or depleted
        truncated = self._step_count >= self.max_steps and not terminated
        self._done = terminated or truncated
        if not self._done:
            self.time_of_day_hours = (self.time_of_day_hours + self.dt_seconds / 3600.0) % 24.0
            self._refresh_link_states()
        info = {
            "success": next_node == self.destination,
            "source": self.source, "destination": self.destination,
            "season": self.season,
            "time_of_day_hours": decision_time,
            "next_time_of_day_hours": self.time_of_day_hours,
            "conditions": state.get("conditions", {}),
            "node": next_node, "qber": float(state["qber"]),
            "skr": float(state["skr"]), "projected_chain_error": projected,
            "chain_error": current_chain, "security_breach": security_breach,
            "pool": self._key_pool[edge_key], "fallback": fallback,
            "visited": list(self._visited), "link_type": link_type,
            "reward_terms": raw,
        }
        return self._get_obs(), total, terminated, truncated, info


if __name__ == "__main__":
    env = QKDRoutingEnv()
    obs, info = env.reset(seed=7)
    print(info, {k: (v.shape if hasattr(v, "shape") else v) for k, v in obs.items()})
