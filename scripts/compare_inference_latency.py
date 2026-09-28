"""Compare GNN, BFS-hop, and Dijkstra-km latency across major-city pairs."""
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

from baselines import BaselinePolicy
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES


CHECKPOINT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUTPUT = ROOT / "experiments" / "inference_latency_comparison.json"
METHODS = ("GNN-200ep", "BFS-hop", "Dijkstra-km")


def main() -> None:
    torch.set_num_threads(8)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CHECKPOINT, model, map_location="cpu")
    model.eval()
    conditions = {"season": "monsoon", "use_case": "defence", "node_feature_mode": "relative",
        "time_of_day_hours": 22.0, "time_jitter_hours": 1.0, "dt_seconds": 300.0,
        "max_steps": 400, "fiber_outage_prob": 0.01, "qber_hard": 0.11,
        "reward_protection_overrides": {"step_cost": -0.05, "revisit": -0.6,
            "switch": -0.2, "security_penalty": -30.0, "depletion_penalty": -60.0,
            "backtrack": -0.5}}
    results = []
    for method_index, method in enumerate(METHODS):
        policy = None if method == "GNN-200ep" else BaselinePolicy(method, seed=93000 + method_index)
        for pair_index, source in enumerate(CITIES):
            for offset, destination in enumerate(CITIES):
                if source == destination:
                    continue
                env = QKDRoutingEnv(source=source, destination=destination, **conditions)
                seed = 91000 + pair_index * 100 + offset
                obs, _ = env.reset(seed=seed)
                if policy is None:
                    with torch.no_grad():
                        model.act(obs, deterministic=True)
                decision_times, route = [], [source]
                success = False
                route_start = time.perf_counter()
                for _ in range(env.max_steps):
                    start = time.perf_counter()
                    if policy is None:
                        with torch.no_grad():
                            action, _, _, _ = model.act(obs, deterministic=True)
                    else:
                        action = policy.act(obs, env)
                    decision_times.append(time.perf_counter() - start)
                    obs, _, terminated, truncated, info = env.step(action)
                    if info.get("node"):
                        route.append(info["node"])
                    if info.get("success"):
                        success = True
                    if terminated or truncated:
                        break
                arr = np.asarray(decision_times, dtype=float)
                results.append({"method": method, "source": source, "destination": destination,
                    "success": success, "hops": len(route) - 1 if success else None,
                    "decisions": len(arr), "route_seconds": time.perf_counter() - route_start,
                    "decision_median_seconds": float(np.median(arr)),
                    "decision_mean_seconds": float(np.mean(arr))})
    summary = {}
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        summary[method] = {"pairs": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "decision_median_seconds": float(np.median([r["decision_median_seconds"] for r in rows])),
            "decision_mean_seconds": float(np.mean([r["decision_median_seconds"] for r in rows])),
            "route_median_seconds": float(np.median([r["route_seconds"] for r in rows])),
            "route_mean_seconds": float(np.mean([r["route_seconds"] for r in rows]))}
    report = {"checkpoint": str(CHECKPOINT), "conditions": conditions,
              "summary": summary, "pairs": results}
    OUTPUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
