"""Collect paper-facing latency, entropy, reliability, and efficiency metrics."""
from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_engine import act, configure_cpu, entropy, localize_observation
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES, build_topology

DEFAULT_CKPT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
DEFAULT_OUT = ROOT / "experiments" / "paper_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    checkpoint, output = args.checkpoint, args.output
    configure_cpu(threads=4, interop_threads=2)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(checkpoint, model, map_location="cpu"); model.eval()
    graph = build_topology()
    rows = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source == destination:
                continue
            env = QKDRoutingEnv(source=source, destination=destination, season="monsoon",
                use_case="defence", node_feature_mode="relative", time_of_day_hours=22,
                time_jitter_hours=1, fiber_outage_prob=0.01, max_steps=400)
            obs, _ = env.reset(seed=91000 + i * 100 + j)
            decisions, entropies, normalized, qber_ok = [], [], [], []
            route = [source]; success = False
            route_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(env.max_steps):
                    infer_obs = localize_observation(obs, hops=2)
                    start = time.perf_counter(); action = act(model, infer_obs)
                    decisions.append(time.perf_counter() - start)
                    h, hn = entropy(model, infer_obs); entropies.append(h); normalized.append(hn)
                    obs, _, terminated, truncated, info = env.step(action)
                    if info.get("node"):
                        route.append(info["node"])
                    if "qber" in info and "skr" in info:
                        qber_ok.append(float(info["qber"]) < env.qber_hard and float(info["skr"]) > 0)
                    success = success or bool(info.get("success"))
                    if terminated or truncated:
                        break
            distance = sum(graph[u][v]["distance_km"] for u, v in zip(route, route[1:]))
            rows.append({"source": source, "destination": destination, "success": success,
                "hops": len(route) - 1 if success else None, "distance_km": distance,
                "route_seconds": time.perf_counter() - route_start,
                "decision_p50_ms": float(np.percentile(decisions, 50) * 1000),
                "decision_p95_ms": float(np.percentile(decisions, 95) * 1000),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_normalized_mean": float(np.mean(normalized)),
                "qber_skr_valid_rate": float(np.mean(qber_ok)) if qber_ok else None})
    decisions = np.asarray([r["decision_p50_ms"] for r in rows])
    report = {"checkpoint": str(checkpoint), "inference_mode": "local_2_hop_fp32",
        "conditions": {"season": "monsoon", "time": "22:00", "fiber_outage_prob": 0.01},
        "model": {"parameters": sum(p.numel() for p in model.parameters()),
                  "checkpoint_bytes": checkpoint.stat().st_size},
        "aggregate": {"pairs": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "decision_p50_ms": float(np.percentile(decisions, 50)),
            "decision_p95_ms": float(np.percentile(decisions, 95)),
            "route_p50_s": float(np.percentile([r["route_seconds"] for r in rows], 50)),
            "route_p95_s": float(np.percentile([r["route_seconds"] for r in rows], 95)),
            "entropy_mean": float(np.mean([r["entropy_mean"] for r in rows])),
            "normalized_entropy_mean": float(np.mean([r["entropy_normalized_mean"] for r in rows])),
            "qber_skr_valid_rate": float(np.mean([r["qber_skr_valid_rate"] for r in rows])),
            "median_hops": float(np.median([r["hops"] for r in rows if r["success"]])),
            "median_distance_km": float(np.median([r["distance_km"] for r in rows]))},
        "established_method_entropy": {"BFS-hop": 0.0, "Dijkstra-km": 0.0, "Max-SKR": 0.0},
        "pairs": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
