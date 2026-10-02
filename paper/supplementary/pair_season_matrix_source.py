"""Run a paired route-policy matrix over non-training city pairs and weather times.

This is a cross-pair transfer diagnostic. The stability campaign's BC stage
randomizes endpoints, so the tested pairs are held out from PPO rollouts but
not guaranteed to be unseen during behavioral cloning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import BaselinePolicy
from evaluation import _run_episode
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES, build_topology

PAIRS = (("Delhi", "Mumbai"), ("Mumbai", "Chennai"), ("Kolkata", "Delhi"),
         ("Bangalore", "Hyderabad"), ("Jaipur", "Kolkata"), ("Hyderabad", "Mumbai"))
CONDITIONS = (("normal", 2.0), ("normal", 22.0), ("summer", 22.0),
              ("winter", 22.0), ("monsoon", 2.0), ("monsoon", 22.0))
METHODS = ("GNN-PPO", "Random", "Dijkstra-km", "BFS-hop", "Max-SKR")


def wilson(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    hops = [row["hops"] for row in rows if row["hops"] is not None]
    return {"episodes": n, "successes": successes, "success_rate": successes / n if n else 0.0,
        "success_rate_wilson_95pct": wilson(successes, n),
        "successful_mean_hops": float(np.mean(hops)) if hops else None,
        "invalid_actions": sum(row["invalid_action_count"] for row in rows),
        "no_valid_action_states": sum(row["no_valid_action_states"] for row in rows),
        "fallbacks": sum(row["fallback_count"] for row in rows),
        "revisits": sum(row["revisits"] for row in rows),
        "mean_reward": float(np.mean([row["reward"] for row in rows])) if rows else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "supplementary" /
                        "pair_season_matrix_20261002.json")
    parser.add_argument("--seeds-per-condition", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=144)
    args = parser.parse_args()
    if args.seeds_per_condition < 1:
        parser.error("--seeds-per-condition must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("model_name", "GNN") not in {"GNN", "GNNActorCritic"}:
        parser.error("This matrix runner currently supports GNN checkpoints only")
    torch.set_num_threads(4)
    model = GNNActorCritic(hidden_dim=int(metadata.get("hidden_dim", 64)),
                           dropedge_probability=0.05)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    graph = build_topology()
    if not all(city in graph for pair in PAIRS for city in pair):
        raise ValueError("endpoint matrix references cities missing from the topology")

    rows = []
    seed_base = 880_000
    for pair_index, (source, destination) in enumerate(PAIRS):
        if source == "Delhi" and destination == "Chennai":
            raise AssertionError("training endpoint pair must remain excluded")
        for condition_index, (season, hour) in enumerate(CONDITIONS):
            for replicate in range(args.seeds_per_condition):
                episode_seed = seed_base + pair_index * 10_000 + condition_index * 100 + replicate
                common = {"graph": graph, "source": source, "destination": destination,
                    "season": season, "use_case": "defence", "node_feature_mode": "relative",
                    "time_of_day_hours": hour, "time_jitter_hours": 0.0,
                    "fiber_outage_prob": 0.01, "qber_hard": 0.11,
                    "max_steps": args.max_steps}
                for method in METHODS:
                    env = QKDRoutingEnv(**common)
                    policy = model if method == "GNN-PPO" else BaselinePolicy(method, episode_seed)
                    result = _run_episode(policy, env, episode_seed)
                    rows.append({"method": method, "source": source,
                        "destination": destination, "season": season, "hour": hour,
                        "replicate": replicate, "seed": episode_seed, **result})

    grouped = defaultdict(list)
    by_method = defaultdict(list)
    by_pair = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["season"], row["hour"])].append(row)
        by_method[row["method"]].append(row)
        by_pair[(row["method"], row["source"], row["destination"])].append(row)
    report = {"status": "complete", "study": "cross-pair and season/diurnal transfer matrix",
        "scope_note": "Pairs are excluded from PPO's fixed-endpoint rollouts, but BC randomized endpoints; this is not a strict end-to-end held-out endpoint test. Simulator-derived only.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "protocol": {"pairs": [list(pair) for pair in PAIRS],
            "conditions": [{"season": season, "hour": hour} for season, hour in CONDITIONS],
            "seeds_per_condition": args.seeds_per_condition,
            "shared_environment_seeds_paired": True, "max_steps": args.max_steps,
            "training_pair_excluded": ["Delhi", "Chennai"],
            "methods": list(METHODS)},
        "summary_by_method": {method: summarize(by_method[method]) for method in METHODS},
        "summary_by_method_condition": {f"{method}|{season}|{int(hour):02d}h": summarize(group)
            for (method, season, hour), group in grouped.items()},
        "summary_by_method_pair": {f"{method}|{source}|{destination}": summarize(group)
            for (method, source, destination), group in by_pair.items()},
        "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "source", "destination", "season",
            "hour", "replicate", "seed", "success", "hops", "reward", "revisits",
            "invalid_action_count", "no_valid_action_states", "fallback_count",
            "qber_skr_validity", "route"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(row[key]) if isinstance(row[key], (dict, list))
                                   else row.get(key)) for key in writer.fieldnames})
    print(json.dumps(report["summary_by_method"], indent=2))


if __name__ == "__main__":
    main()
