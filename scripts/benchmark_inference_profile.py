"""Profile per-decision CPU latency and process memory across thread counts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES, build_topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "supplementary" /
                        "inference_profile_20261002.json")
    parser.add_argument("--repeats-per-pair", type=int, default=20)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    if args.repeats_per_pair < 5:
        parser.error("--repeats-per-pair must be at least 5 for tail quantiles")
    if any(thread < 1 for thread in args.threads):
        parser.error("thread counts must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    model = GNNActorCritic(hidden_dim=int(metadata.get("hidden_dim", 64)),
                           dropedge_probability=0.05)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    # Disable observation memoization so each timed call includes the full
    # encoder, attention head, action mask, and categorical action selection.
    model.enable_encoder_cache(False)

    graph = build_topology()
    observations = []
    pair_seeds = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source == destination:
                continue
            env = QKDRoutingEnv(graph=graph, source=source, destination=destination,
                season="monsoon", use_case="defence", node_feature_mode="relative",
                time_of_day_hours=22.0, time_jitter_hours=1.0, dt_seconds=300,
                max_steps=144, fiber_outage_prob=0.01, qber_hard=0.11)
            seed = 962000 + i * 100 + j
            observations.append(env.reset(seed=seed)[0])
            pair_seeds.append({"source": source, "destination": destination, "seed": seed})

    process = psutil.Process()
    background_processes = []
    for candidate in psutil.process_iter(["pid", "name"]):
        try:
            if candidate.info["pid"] != process.pid and str(candidate.info["name"]).lower().startswith("python"):
                background_processes.append({"pid": candidate.info["pid"],
                    "name": candidate.info["name"],
                    "cpu_seconds_at_start": sum(candidate.cpu_times()[:2])})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    memory_before = process.memory_info().rss
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    by_thread = {}
    with torch.no_grad():
        for threads in args.threads:
            torch.set_num_threads(threads)
            for obs in observations[: min(5, len(observations))]:
                model.act(obs, deterministic=True)
            samples = []
            for repeat in range(args.repeats_per_pair):
                for obs in observations:
                    start = time.perf_counter_ns()
                    model.act(obs, deterministic=True)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
            values = np.asarray(samples, dtype=float)
            by_thread[str(threads)] = {
                "decisions": int(values.size),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "mean_ms": float(np.mean(values)),
                "max_ms": float(np.max(values)),
                "std_ms": float(np.std(values, ddof=1)),
            }
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "protocol": {"ordered_city_pairs": len(observations),
            "repeats_per_pair": args.repeats_per_pair,
            "thread_counts": args.threads, "season": "monsoon",
            "time_of_day_hours": 22.0, "feature_mode": "relative",
            "fiber_outage_prob": 0.01, "qber_hard": 0.11,
            "max_steps": 144, "encoder_cache": "disabled",
            "timed_operation": "one deterministic model.act decision",
            "note": "One machine and one fixed simulator topology; no graph-size scaling claim."},
        "host": {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "torch_version": torch.__version__},
        "memory": {"process_rss_before_bytes": int(memory_before),
            "process_rss_after_bytes": int(process.memory_info().rss),
            "model_parameter_bytes": int(parameter_bytes)},
        "measurement_context": {"other_python_processes_at_start": background_processes,
            "concurrent_load_note": "Other Python processes may affect latency; inspect this list before comparing runs."},
        "summary_by_threads": by_thread,
        "pair_seeds": pair_seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_by_threads": by_thread, "memory": report["memory"]}, indent=2))


if __name__ == "__main__":
    main()
