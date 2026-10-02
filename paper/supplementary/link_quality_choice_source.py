"""Test policy decisions under device, hour, season and turbulence inputs.

This controlled counterfactual holds node embeddings and route progress fixed;
fiber/FSO QBER and SKR are recomputed from the active physics model. It is a
diagnostic of learned action preferences, not a route-level performance score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
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
    model.enable_encoder_cache(True)
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

    def counterfactual(fiber_rate: float, fso_rate: float,
                       fiber_qber: float, fso_qber: float,
                       fso_available: bool = True) -> dict:
        """Hold route context fixed while varying one policy input at a time."""
        obs = {key: np.array(value, copy=True) for key, value in observation.items()}
        obs["neighbor_indices"][:] = -1
        obs["neighbor_indices"][fiber_slot] = same_neighbor
        obs["neighbor_indices"][fso_slot] = same_neighbor
        obs["edge_valid_mask"][:] = 0
        obs["edge_features"][:] = 0.0
        for slot, is_fso, rate, qber in (
                (fiber_slot, 0.0, fiber_rate, fiber_qber),
                (fso_slot, 1.0, fso_rate, fso_qber)):
            feature = [qber, rate, is_fso, 0.125, qber, 1.0, 0.0, 0.0, 0.5]
            obs["edge_features"][slot] = feature
            available = (not is_fso or fso_available)
            obs["edge_valid_mask"][slot] = int(
                available and qber < env.qber_hard and rate > 0.0)
        with torch.no_grad():
            logits, _ = model(obs)
            pair_logits = logits[0, [fiber_slot, fso_slot]]
            probabilities = torch.softmax(pair_logits, dim=0).cpu().tolist()
        return {"fiber_probability": probabilities[0],
                "fso_probability": probabilities[1],
                "fiber_valid": bool(obs["edge_valid_mask"][fiber_slot]),
                "fso_valid": bool(obs["edge_valid_mask"][fso_slot])}

    rate_sensitivity = []
    base_fiber_rate = 0.05
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        rate_sensitivity.append({"fso_to_fiber_rate_ratio": ratio,
            **counterfactual(base_fiber_rate, base_fiber_rate * ratio, 0.02, 0.02)})
    qber_sensitivity = []
    for q in (0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.109, 0.111):
        qber_sensitivity.append({"fso_qber": q,
            **counterfactual(0.05, 0.05, 0.02, q)})
    outage_sensitivity = counterfactual(0.05, 0.05, 0.02, 0.02,
                                        fso_available=False)

    # Vary only the latent turbulence input while preserving the candidate
    # geometry, route context, atmospheric draw, fiber candidate, and clear
    # availability state. This isolates the link-quality pathway; it is not a
    # claim that these multipliers are calibrated weather distributions.
    fso_turbulence_sensitivity = []
    for scale in (0.01, 0.1, 1.0, 10.0, 100.0):
        trials = []
        for sample_index in range(samples_per_condition):
            sample_seed = 88000 + sample_index
            fiber = sample_link_state("fiber", 80.0, "monsoon", 22.0,
                random.Random(sample_seed), fiber_outage_prob=0.0)
            reference_fso = sample_link_state("fso", 10.0, "monsoon", 22.0,
                random.Random(sample_seed + 1), outage_uniform=0.0)
            fso_rng = random.Random(sample_seed + 1)
            fso_rng.lognormvariate(0.0, 0.5)  # consume the natural-Cn2 draw
            fso = sample_link_state("fso", 10.0, "monsoon", 22.0,
                fso_rng, outage_uniform=0.0,
                cn2_override=reference_fso["conditions"]["Cn2"] * scale)
            decision = score_pair(fiber, fso)
            trials.append({"fso_probability": decision["fso_probability"],
                "selected": decision["selected"], "qber": fso["qber"],
                "skr": fso["skr"],
                "valid": bool(fso["qber"] < env.qber_hard and fso["skr"] > 0.0),
                "cn2": fso["conditions"]["Cn2"],
                "rytov_variance": fso["conditions"]["rytov_variance"]})
        fso_turbulence_sensitivity.append({"cn2_multiplier": scale,
            "samples": len(trials),
            "mean_fso_probability": float(np.mean([t["fso_probability"] for t in trials])),
            "fso_selected_rate": float(np.mean([t["selected"] == "fso" for t in trials])),
            "valid_fso_candidate_rate": float(np.mean([t["valid"] for t in trials])),
            "median_qber": float(np.median([t["qber"] for t in trials])),
            "median_skr_proxy": float(np.median([t["skr"] for t in trials])),
            "median_rytov_variance": float(np.median([t["rytov_variance"] for t in trials]))})

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

    source_path = Path(__file__).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_path.parent,
        check=True, capture_output=True, text=True).stdout.strip()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source": source_path.name,
        "source_sha256": source_sha256,
        "git_commit": git_commit,
        "test": "matched-progress twin candidate counterfactual, 10 km FSO detour vs 80 km fiber",
        "shared_state": {"seasonal_outages_conditioned_clear": True,
                         "same_neighbor_embedding": True,
                         "same_progress_and_distance_features": True,
                         "fiber_detector_profile": "IDQ ID281 1550 nm characterized channel",
                         "fso_detector": "785 nm scenario assumption, not ID281 profile"},
        "conditions": condition_results,
        "controlled_input_sensitivity": {
            "shared_context": "same current/destination/neighbor embedding, progress, distance, pool, and link-switch features; one input varied at a time",
            "rate_ratio_sweep": rate_sensitivity,
            "qber_sweep": qber_sensitivity,
            "fso_outage_mask": outage_sensitivity,
            "fso_turbulence_cn2_multiplier_sweep": fso_turbulence_sensitivity,
            "unsupported_perturbations": [
                "pointing jitter parameter is fixed by the current physics model",
                "background scaling is fixed by the current physics model",
                "persistent outage bursts are not modeled independently of link-state availability"],
            "interpretation_note": "Synthetic policy-input counterfactuals isolate response of the current checkpoint. They do not represent measured links or end-to-end route success.",
        },
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
