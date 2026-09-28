"""Benchmark GNN inference latency for every ordered major-city pair."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES


CHECKPOINT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUTPUT = ROOT / "experiments" / "pair_inference_benchmark.json"


def main() -> None:
    torch.set_num_threads(8)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CHECKPOINT, model, map_location="cpu")
    model.eval()
    env_kwargs = {
        "season": "monsoon", "use_case": "defence", "node_feature_mode": "relative",
        "time_of_day_hours": 22.0, "time_jitter_hours": 1.0, "dt_seconds": 300.0,
        "max_steps": 400, "fiber_outage_prob": 0.01, "qber_hard": 0.11,
        "reward_protection_overrides": {"step_cost": -0.05, "revisit": -0.6,
            "switch": -0.2, "security_penalty": -30.0, "depletion_penalty": -60.0,
            "backtrack": -0.5},
    }
    results = []
    with torch.no_grad():
        for index, source in enumerate(CITIES):
            for offset, destination in enumerate(CITIES):
                if source == destination:
                    continue
                env = QKDRoutingEnv(source=source, destination=destination, **env_kwargs)
                obs, _ = env.reset(seed=91000 + index * 100 + offset)
                # Warm up model/runtime before timing this pair.
                model.act(obs, deterministic=True)
                decision_times = []
                route_start = time.perf_counter()
                route = [source]
                success = False
                for _ in range(env.max_steps):
                    start = time.perf_counter()
                    action, _, _, _ = model.act(obs, deterministic=True)
                    decision_times.append(time.perf_counter() - start)
                    obs, _, terminated, truncated, info = env.step(action)
                    if info.get("node"):
                        route.append(info["node"])
                    if info.get("success"):
                        success = True
                    if terminated or truncated:
                        break
                route_seconds = time.perf_counter() - route_start
                arr = np.asarray(decision_times, dtype=float)
                results.append({
                    "source": source, "destination": destination, "success": success,
                    "hops": len(route) - 1 if success else None,
                    "decisions": len(arr), "route_inference_seconds": route_seconds,
                    "decision_seconds_median": float(np.median(arr)),
                    "decision_seconds_mean": float(np.mean(arr)),
                    "decision_seconds_std": float(np.std(arr)),
                    "decision_seconds_min": float(np.min(arr)),
                    "decision_seconds_max": float(np.max(arr)),
                })
    all_decisions = np.asarray([r["decision_seconds_median"] for r in results])
    report = {"checkpoint": str(CHECKPOINT), "conditions": env_kwargs,
              "cities": list(CITIES), "ordered_pairs": len(results),
              "summary": {"decision_median_seconds": float(np.median(all_decisions)),
                          "decision_mean_seconds": float(np.mean(all_decisions)),
                          "route_median_seconds": float(np.median([r["route_inference_seconds"] for r in results])),
                          "successful_pairs": sum(r["success"] for r in results)},
              "pairs": results}
    OUTPUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
