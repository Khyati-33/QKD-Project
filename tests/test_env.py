import random

import numpy as np
import pytest

from physics import QBER_HARD, chain_parity_error
from qkd_env import (C_SEC, FAILURE_PENALTY, REVISIT_PENALTY, STEP_COST,
                     SWITCH_PENALTY, QKDRoutingEnv, distance_normalized_skr_reward)
from train import _teacher_slot


def test_reset_random_steps_and_scalars():
    env = QKDRoutingEnv()
    obs, _ = env.reset(seed=13)
    assert len(env.state_scalars()) == 2
    assert all(np.isfinite(obs["state_scalars"]))
    assert 0 <= obs["state_scalars"][0] <= env.longest_route_km / env.max_edge_km
    assert 0 <= obs["state_scalars"][1] <= 1
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break


def test_diurnal_clock_advances_and_reset_restarts_at_configured_hour():
    env = QKDRoutingEnv(season="monsoon", time_of_day_hours=22.0,
                        dt_seconds=1800.0, max_steps=10)
    obs, info = env.reset(seed=22)
    assert info["time_of_day_hours"] == pytest.approx(22.0)
    valid = np.flatnonzero(obs["edge_valid_mask"])
    assert len(valid)
    _, _, terminated, truncated, step_info = env.step(int(valid[0]))
    assert step_info["time_of_day_hours"] == pytest.approx(22.0)
    if not (terminated or truncated):
        assert step_info["next_time_of_day_hours"] == pytest.approx(22.5)
        assert env.time_of_day_hours == pytest.approx(22.5)
    _, reset_info = env.reset(seed=23)
    assert reset_info["time_of_day_hours"] == pytest.approx(22.0)


def test_defence_profile_uses_relative_features_and_penalizes_backtracking():
    env = QKDRoutingEnv(use_case="defence", node_feature_mode="relative",
                        reward_protection_overrides={"step_cost": -0.05,
                            "revisit": -0.6, "switch": -0.2,
                            "security_penalty": -30.0, "depletion_penalty": -60.0,
                            "backtrack": -0.5})
    obs, _ = env.reset(seed=47)
    assert obs["node_features"][env.node_to_idx[env.destination], 3] == pytest.approx(0.0)
    neighbors = list(env.graph.neighbors(env._current_node))
    valid = [i for i in range(len(neighbors)) if obs["edge_valid_mask"][i]]
    backtracking = [i for i in valid if env._km_dist_to_dest(neighbors[i]) >
                    env._km_dist_to_dest(env._current_node)]
    if backtracking:
        _, _, _, _, info = env.step(backtracking[0])
        assert info["reward_terms"]["backtrack"] == -0.5
        assert info["reward_terms"]["margin"] <= 0.0
        assert info["reward_terms"]["skr"] <= 0.0


def test_fso_outages_are_correlated_within_each_detour():
    env = QKDRoutingEnv(season="monsoon")
    env.reset(seed=121)
    groups = {}
    for u, v, attrs in env.graph.edges(data=True):
        if attrs["link_type"] == "fso":
            groups.setdefault(attrs["detour_for"], []).append(
                env._edge_states[env._edge_key(u, v)]["outage"])
    assert groups
    assert all(len(set(outages)) == 1 for outages in groups.values())


def test_fso_detour_outage_is_stable_for_the_episode():
    env = QKDRoutingEnv(season="monsoon")
    env.reset(seed=712)

    def outage_by_detour():
        groups = {}
        for u, v, attrs in env.graph.edges(data=True):
            if attrs["link_type"] == "fso":
                groups.setdefault(attrs["detour_for"], set()).add(
                    env._edge_states[env._edge_key(u, v)]["outage"])
        assert all(len(states) == 1 for states in groups.values())
        return {key: next(iter(states)) for key, states in groups.items()}

    initial = outage_by_detour()
    for hour in (2.0, 14.0, 22.0):
        env.time_of_day_hours = hour
        env._refresh_link_states()
        assert outage_by_detour() == initial
    env.reset(seed=712)
    assert outage_by_detour() == initial


def test_actor_observes_pool_switch_and_revisit_state():
    env = QKDRoutingEnv()
    obs, _ = env.reset(seed=10)
    neighbors = list(env.graph.neighbors(env._current_node))
    slot = int(np.flatnonzero(obs["edge_valid_mask"])[0])
    candidate = neighbors[slot]
    edge_features = obs["edge_features"][slot]
    assert edge_features.shape == (9,)
    assert edge_features[5] == pytest.approx(env._key_pool[env._edge_key(env._current_node, candidate)])
    env._visited.add(candidate)
    env._previous_link_type = "fiber" if env.graph[env._current_node][candidate]["link_type"] == "fso" else "fso"
    updated = env._get_obs()["edge_features"][slot]
    assert updated[6] == 1.0 and updated[7] == 1.0


def test_actor_observes_destination_distance_progress():
    env = QKDRoutingEnv()
    obs, _ = env.reset(seed=13)
    current = env._current_node
    for slot, neighbor_idx in enumerate(obs["neighbor_indices"]):
        if neighbor_idx < 0:
            continue
        neighbor = env.node_names[int(neighbor_idx)]
        expected = (env._km_dist_to_dest(current) - env._km_dist_to_dest(neighbor)) / max(env.max_edge_km, 1.0)
        assert obs["edge_features"][slot, 8] == pytest.approx(expected)


def test_behavior_cloning_teacher_uses_hops_before_skr_for_equal_distance_routes():
    env = QKDRoutingEnv(season="normal", fiber_outage_prob=0.0)
    env.reset(seed=222)
    route = __import__("topology").shortest_fiber_route(env.graph)
    u, v = next((u, v) for u, v in zip(route, route[1:])
        if any(attrs["link_type"] == "fso" and
               set(attrs.get("detour_for", ())) == {u, v}
               for _, _, attrs in env.graph.edges(data=True)))
    env._current_node = u
    fso_neighbor = next(n for n in env.graph.neighbors(u)
                        if env.graph[u][n]["link_type"] == "fso" and
                        env.graph[u][n].get("detour_for") == frozenset((u, v)))
    fiber_key, fso_key = env._edge_key(u, v), env._edge_key(u, fso_neighbor)
    env._edge_states[fiber_key].update(qber=0.01, skr=0.02, outage=False)
    env._edge_states[fso_key].update(qber=0.01, skr=0.20, outage=False)
    # Construct an equal-distance choice: road-aligned FSO spans are slightly
    # shorter in straight-line distance than the winding fiber route geometry.
    env._destination_distances = {node: 1e9 for node in env.graph.nodes}
    env._destination_distances[v] = 100.0
    env._destination_distances[fso_neighbor] = (
        100.0 + env.graph[u][v]["distance_km"] -
        env.graph[u][fso_neighbor]["distance_km"])
    obs = env._get_obs()
    assert env.graph[u][v]["distance_km"] + env._km_dist_to_dest(v) == pytest.approx(
        env.graph[u][fso_neighbor]["distance_km"] +
        env._km_dist_to_dest(fso_neighbor), abs=1e-6)
    chosen = _teacher_slot(env, obs)
    assert env.node_names[int(obs["neighbor_indices"][chosen])] == v


def test_skr_reward_does_not_increase_when_one_span_is_split_into_segments():
    fiber_span = distance_normalized_skr_reward(0.20, 80.0, 80.0, 1.2)
    eight_fso_segments = sum(
        distance_normalized_skr_reward(0.20, 10.0, 80.0, 1.2)
        for _ in range(8))
    assert eight_fso_segments == pytest.approx(fiber_span)


def test_chain_parity_regression():
    value = chain_parity_error([0.09, 0.09, 0.02])
    assert value >= QBER_HARD
    assert 0.02 < QBER_HARD


@pytest.mark.parametrize("term", ["margin", "skr", "pool", "latency", "energy", "congestion"])
def test_reward_ablations_zero_only_selected_term(term):
    base, ablated = QKDRoutingEnv(), QKDRoutingEnv(disabled_reward_terms={term})
    base.reset(seed=47); ablated.reset(seed=47)
    # Select an outgoing valid slot at the same deterministic initial state.
    slot = int(np.flatnonzero(base._get_obs()["edge_valid_mask"])[0])
    _, _, _, _, info_a = base.step(slot)
    _, _, _, _, info_b = ablated.step(slot)
    if term == "pool":
        assert info_b["reward_terms"]["pool_level"] == 0.0
        assert info_b["reward_terms"]["pool_delta"] == 0.0
        assert info_a["reward_terms"]["pool_level"] != 0.0
        assert info_a["reward_terms"]["pool_delta"] != 0.0
    else:
        assert info_b["reward_terms"][term] == 0.0
        assert info_a["reward_terms"][term] != 0.0 or term == "congestion"
    assert info_b["reward_terms"]["step_cost"] == STEP_COST


def test_protected_penalties_not_ablatable():
    env = QKDRoutingEnv(disabled_reward_terms={"step_cost", "revisit", "switch",
                                               "security_penalty", "depletion_penalty"})
    assert env.disabled_reward_terms == set()
    assert STEP_COST != 0 and REVISIT_PENALTY != 0 and SWITCH_PENALTY != 0
    assert C_SEC != 0 and FAILURE_PENALTY != 0
    env.reset(seed=8)
    valid = int(np.flatnonzero(env._get_obs()["edge_valid_mask"])[0])
    _, _, _, _, info = env.step(valid)
    assert info["reward_terms"]["step_cost"] == STEP_COST


def test_invalid_chosen_neighbor_uses_shared_validity_fallback():
    env = QKDRoutingEnv()
    obs, _ = env.reset(seed=111)
    neighbors, _, mask, _ = env._candidate_mask(env._current_node)
    valid = np.flatnonzero(mask[:len(neighbors)])
    assert len(valid) >= 1
    chosen_valid = int(valid[0])
    invalid_candidates = [i for i in range(len(neighbors)) if i not in valid]
    if not invalid_candidates:
        invalid_candidates = [(chosen_valid + 1) % len(neighbors)]
    invalid = invalid_candidates[0]
    invalid_neighbor = neighbors[invalid]
    state = env._edge_states[env._edge_key(env._current_node, invalid_neighbor)]
    state["qber"], state["skr"] = QBER_HARD, 0.0
    assert not env._edge_valid((env._current_node, invalid_neighbor))
    expected_slot = min(valid, key=lambda i: env._km_dist_to_dest(neighbors[int(i)]))
    _, _, _, _, info = env.step(invalid)
    assert info["fallback"]
    assert info["qber"] < QBER_HARD
    assert info["node"] == neighbors[int(expected_slot)]


def test_security_and_pool_penalties_remain_active():
    env = QKDRoutingEnv(disabled_reward_terms={"security_penalty", "depletion_penalty"})
    env.reset(seed=18)
    obs = env._get_obs()
    valid_slots = np.flatnonzero(obs["edge_valid_mask"])
    slot = next(int(i) for i in valid_slots
                if env.graph.nodes[
                    env.node_names[int(obs["neighbor_indices"][i])]
                ].get("node_type") != "full_tn")
    neighbor_idx = int(obs["neighbor_indices"][slot])
    neighbor = env.node_names[neighbor_idx]
    edge_key = env._edge_key(env._current_node, neighbor)
    # Exercise the environment's safety backstop after an externally corrupted
    # policy state; normal masking would exclude an already unsafe projection.
    env._hop_qbers = [0.09]
    env._edge_states[edge_key]["qber"] = 0.09
    env._edge_states[edge_key]["skr"] = 0.1
    env._candidate_mask = lambda _u: (list(env.graph.neighbors(_u)),
        np.zeros((env.max_neighbors, 5), dtype=np.float32),
        np.ones(env.max_neighbors, dtype=np.int8), obs["neighbor_indices"])
    env._key_pool[edge_key] = 0.0
    _, _, terminated, _, info = env.step(slot)
    assert terminated
    assert info["reward_terms"]["security_penalty"] == C_SEC
    assert info["reward_terms"]["depletion_penalty"] != 0


def test_randomized_endpoints_seeded_pairs_and_distances():
    env = QKDRoutingEnv(randomize_endpoints=True)
    pairs = []
    for seed in (1, 2, 3, 4):
        obs, info = env.reset(seed=seed)
        pairs.append((info["source"], info["destination"]))
        assert obs["destination_node"] == env.node_to_idx[info["destination"]]
        assert obs["state_scalars"][0] == pytest.approx(
            env._km_dist_to_dest(env.source) / env.max_edge_km)
    assert len(set(pairs)) > 1
