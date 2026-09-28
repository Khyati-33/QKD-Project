"""Benchmark inference-only local-subgraph GNN routing."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_engine import act, configure_cpu, localize_observation
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES

CKPT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUT = ROOT / "experiments" / "local_subgraph_benchmark.json"
CONDITIONS = {"season": "monsoon", "use_case": "defence", "node_feature_mode": "relative",
    "time_of_day_hours": 22.0, "time_jitter_hours": 1.0, "dt_seconds": 300.0,
    "max_steps": 400, "fiber_outage_prob": 0.01, "qber_hard": 0.11}


def run(model, hops: int | None) -> list[dict]:
    rows = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source == destination:
                continue
            env = QKDRoutingEnv(source=source, destination=destination, **CONDITIONS)
            obs, _ = env.reset(seed=91000 + i * 100 + j)
            timed, success, route = [], False, 0
            start_route = time.perf_counter()
            with torch.inference_mode():
                for _ in range(env.max_steps):
                    infer_obs = localize_observation(obs, hops=hops) if hops is not None else obs
                    start = time.perf_counter(); action = act(model, infer_obs)
                    timed.append(time.perf_counter() - start)
                    obs, _, terminated, truncated, info = env.step(action)
                    route += 1; success = success or bool(info.get("success"))
                    if terminated or truncated:
                        break
            rows.append({"source": source, "destination": destination, "success": success,
                "hops": route if success else None, "decision_median_seconds": float(np.median(timed)),
                "route_seconds": time.perf_counter() - start_route})
    return rows


def summary(rows: list[dict]) -> dict:
    return {"pairs": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
        "decision_median_seconds": float(np.median([r["decision_median_seconds"] for r in rows])),
        "route_median_seconds": float(np.median([r["route_seconds"] for r in rows])),
        "median_hops_success": float(np.median([r["hops"] for r in rows if r["success"]]))}


def main() -> None:
    configure_cpu(threads=4, interop_threads=2)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CKPT, model, map_location="cpu"); model.eval()
    full = run(model, None)
    local = run(model, 2)
    report = {"checkpoint": str(CKPT), "conditions": CONDITIONS,
        "full_graph": summary(full), "local_2_hop": summary(local),
        "pairs": [{"mode": "full_graph", **r} for r in full] +
                 [{"mode": "local_2_hop", **r} for r in local]}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"full_graph": report["full_graph"], "local_2_hop": report["local_2_hop"]}, indent=2))


if __name__ == "__main__":
    main()
