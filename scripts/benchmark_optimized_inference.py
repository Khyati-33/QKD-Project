"""Benchmark production-style optimized GNN inference against FP32."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_engine import act, batched_actions, configure_cpu, quantize_dynamic_int8
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES


CHECKPOINT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUTPUT = ROOT / "experiments" / "optimized_inference_benchmark.json"
CONDITIONS = {"season": "monsoon", "use_case": "defence", "node_feature_mode": "relative",
    "time_of_day_hours": 22.0, "time_jitter_hours": 1.0, "dt_seconds": 300.0,
    "max_steps": 400, "fiber_outage_prob": 0.01, "qber_hard": 0.11,
    "reward_protection_overrides": {"step_cost": -0.05, "revisit": -0.6,
        "switch": -0.2, "security_penalty": -30.0, "depletion_penalty": -60.0,
        "backtrack": -0.5}}


def new_model() -> GNNActorCritic:
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CHECKPOINT, model, map_location="cpu")
    return model.eval()


def route_benchmark(model, label: str) -> list[dict]:
    rows = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source == destination:
                continue
            env = QKDRoutingEnv(source=source, destination=destination, **CONDITIONS)
            obs, _ = env.reset(seed=91000 + i * 100 + j)
            act(model, obs)  # warm-up
            times, success, hops = [], False, 0
            start_route = time.perf_counter()
            with torch.inference_mode():
                for _ in range(env.max_steps):
                    start = time.perf_counter()
                    action = act(model, obs)
                    times.append(time.perf_counter() - start)
                    obs, _, terminated, truncated, info = env.step(action)
                    hops += 1
                    success = success or bool(info.get("success"))
                    if terminated or truncated:
                        break
            rows.append({"method": label, "source": source, "destination": destination,
                "success": success, "hops": hops if success else None,
                "decision_median_seconds": float(np.median(times)),
                "decision_mean_seconds": float(np.mean(times)),
                "route_seconds": time.perf_counter() - start_route})
    return rows


def batch_benchmark(model, batch_sizes=(1, 4, 8, 16, 32, 42)) -> dict:
    observations = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source == destination:
                continue
            env = QKDRoutingEnv(source=source, destination=destination, **CONDITIONS)
            obs, _ = env.reset(seed=91000 + i * 100 + j)
            observations.append(obs)
    result = {}
    for size in batch_sizes:
        sample = observations[:size]
        batched_actions(model, sample)
        samples = []
        for _ in range(5):
            start = time.perf_counter(); batched_actions(model, sample)
            samples.append(time.perf_counter() - start)
        result[str(size)] = {"batch_seconds_median": float(np.median(samples)),
            "per_decision_seconds": float(np.median(samples) / size)}
    return result


def summarize(rows: list[dict]) -> dict:
    return {"pairs": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
        "decision_median_seconds": float(np.median([r["decision_median_seconds"] for r in rows])),
        "decision_mean_seconds": float(np.mean([r["decision_median_seconds"] for r in rows])),
        "route_median_seconds": float(np.median([r["route_seconds"] for r in rows]))}


def main() -> None:
    configure_cpu(threads=4, interop_threads=2)
    fp32 = new_model()
    int8 = quantize_dynamic_int8(new_model())
    fp_rows = route_benchmark(fp32, "GNN-FP32-inference-mode")
    int_rows = route_benchmark(int8, "GNN-INT8-dynamic")
    report = {"checkpoint": str(CHECKPOINT), "conditions": CONDITIONS,
        "optimizations": ["torch.inference_mode", "dynamic INT8 Linear quantization",
                           "bounded CPU threads", "batched independent decisions"],
        "summary": {"FP32": summarize(fp_rows), "INT8": summarize(int_rows)},
        "batch_throughput": batch_benchmark(fp32), "pairs": fp_rows + int_rows}
    OUTPUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(json.dumps({"summary": report["summary"], "batch_throughput": report["batch_throughput"]}, indent=2))


if __name__ == "__main__":
    main()
