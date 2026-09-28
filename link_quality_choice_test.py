"""Test policy decisions under device, hour, season and turbulence inputs.

This controlled counterfactual holds node embeddings and route progress fixed;
fiber/FSO QBER and SKR are recomputed from the active physics model. It is a
diagnostic of learned action preferences, not a route-level performance score.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from models import GNNActorCritic
from physics import (FIBER_DARK_COUNT_HZ, FIBER_DETECTOR_EFFICIENCY,
                     fso_viability_probability, sample_link_state)
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import shortest_fiber_route


def run_test(checkpoint: Path, samples_per_condition: int = 32) -> dict:
    torch.set_num_threads(8)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    env = QKDRoutingEnv(season="monsoon", use_case="defence",
        node_feature_mode="relative", time_of_day_hours=22.0,
        time_jitter_hours=0.0, fiber_outage_prob=0.0)
    env.reset(seed=314)
    route = shortest_fiber_route(env.graph)
    detoured_hop = next((u, v) for u, v in zip(route, route[1:])
        if any(attrs["link_type"] == "fso" and
               set(attrs.get("detour_for", ())) == {u, v}
               for _, _, attrs in env.graph.edges(data=True)))
    u, v = detoured_hop
    env._current_node = u
    env._visited = {u}
    env._hop_qbers = []
    env._previous_link_type = None
    observation = env._get_obs()
    neighbors = list(env.graph.neighbors(u))
    fiber_slot = next(i for i, n in enumerate(neighbors)
                      if env.graph[u][n]["link_type"] == "fiber" and n == v)
    fso_slot = next(i for i, n in enumerate(neighbors)
                    if env.graph[u][n]["link_type"] == "fso")
    same_neighbor = env.node_to_idx[v]

    def score_pair(fiber_state: dict, fso_state: dict) -> dict:
        obs = {key: np.array(value, copy=True) for key, value in observation.items()}
        obs["neighbor_indices"][:] = -1
        obs["neighbor_indices"][fiber_slot] = same_neighbor
        obs["neighbor_indices"][fso_slot] = same_neighbor
        obs["edge_valid_mask"][:] = 0
        obs["edge_features"][:] = 0.0
        common = [0.0, 0.0, 0.0, 0.125, 0.0, 1.0, 0.0, 0.0, 0.5]
        for slot, is_fso, quality in ((fiber_slot, 0.0, fiber_state),
                                      (fso_slot, 1.0, fso_state)):
            feature = list(common)
            feature[0] = float(quality["qber"])
            feature[1] = float(quality["skr"])
            feature[2] = is_fso
            feature[4] = float(quality["qber"])
            obs["edge_features"][slot] = feature
            obs["edge_valid_mask"][slot] = int(not quality["outage"] and quality["skr"] > 0)
        with torch.no_grad():
            logits, _ = model(obs)
            pair_logits = logits[0, [fiber_slot, fso_slot]]
            probabilities = torch.softmax(pair_logits, dim=0).cpu().tolist()
        return {"fiber_probability": probabilities[0], "fso_probability": probabilities[1],
                "selected": "fiber" if probabilities[0] >= probabilities[1] else "fso"}

    condition_results = {}
    seed = 70100
    for season in ("normal", "summer", "winter", "monsoon"):
        for hour in (2.0, 14.0, 22.0):
            trials = []
            for _ in range(samples_per_condition):
                seed += 1
                fiber = sample_link_state("fiber", 80.0, season, hour,
                    random.Random(seed), fiber_outage_prob=0.0)
                # Condition on a clear FSO path to measure quality preferences;
                # marginal FSO availability is reported separately.
                fso_rng = random.Random(seed + 100_000)
                fso = sample_link_state("fso", 10.0, season, hour, fso_rng,
                                        outage_uniform=0.0)
                decision = score_pair(fiber, fso)
                c = fso["conditions"]
                trials.append({
                    **decision,
                    "fiber_qber": fiber["qber"], "fiber_skr": fiber["skr"],
                    "fso_qber": fso["qber"], "fso_skr": fso["skr"],
                    "fso_transmittance": c["transmittance"],
                    "cn2": c["Cn2"], "effective_cn2": c["effective_Cn2"],
                    "rytov_variance": c["rytov_variance"],
                    "fso_better_skr": fso["skr"] > fiber["skr"],
                })
            key = f"{season}_{int(hour):02d}h"
            probabilities = np.asarray([t["fso_probability"] for t in trials])
            fso_rates = np.asarray([t["fso_skr"] for t in trials])
            policy_vs_skr_corr = (float(np.corrcoef(probabilities, fso_rates)[0, 1])
                if len(trials) >= 2 and np.std(probabilities) > 0 and np.std(fso_rates) > 0
                else None)
            condition_results[key] = {
                "season": season, "hour": hour,
                "fso_marginal_availability": fso_viability_probability(10.0, season),
                "fiber_dark_count_hz_upper_bound": FIBER_DARK_COUNT_HZ,
                "fiber_sde": FIBER_DETECTOR_EFFICIENCY,
                "trials_conditional_on_fso_clear": samples_per_condition,
                "mean_fso_probability": statistics.mean(t["fso_probability"] for t in trials),
                "pearson_correlation_fso_probability_vs_fso_skr_proxy": policy_vs_skr_corr,
                "fso_selected_rate": sum(t["selected"] == "fso" for t in trials) / len(trials),
                "fso_higher_skr_rate": sum(t["fso_better_skr"] for t in trials) / len(trials),
                "median_fso_skr_proxy": statistics.median(t["fso_skr"] for t in trials),
                "median_fiber_skr_proxy": statistics.median(t["fiber_skr"] for t in trials),
                "median_fso_transmittance": statistics.median(t["fso_transmittance"] for t in trials),
                "median_cn2": statistics.median(t["cn2"] for t in trials),
                "median_rytov_variance": statistics.median(t["rytov_variance"] for t in trials),
                "quality_samples": trials,
            }

    return {
        "checkpoint": str(checkpoint),
        "test": "matched-progress twin candidate counterfactual, 10 km FSO detour vs 80 km fiber",
        "shared_state": {"seasonal_outages_conditioned_clear": True,
                         "same_neighbor_embedding": True,
                         "same_progress_and_distance_features": True,
                         "fiber_detector_profile": "IDQ ID281 1550 nm characterized channel",
                         "fso_detector": "785 nm scenario assumption, not ID281 profile"},
        "conditions": condition_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-per-condition", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_test(args.checkpoint, args.samples_per_condition)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf8")
    print(rendered)


if __name__ == "__main__":
    main()
