"""Benchmark exact LSTM encoder memoization on the frozen matched holdout matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GNNActorCritic, LSTMActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology

REPORT = ROOT / "experiments" / "final_gnn_lstm_holdout_20261007.json"
GNN_REPORT = ROOT / "experiments" / "final_holdout_20261007.json"
OUT = ROOT / "experiments" / "lstm_encoder_cache_benchmark_20261112.json"
PAIRS = (("Delhi", "Mumbai"), ("Mumbai", "Chennai"), ("Kolkata", "Delhi"),
         ("Bangalore", "Hyderabad"), ("Jaipur", "Kolkata"), ("Hyderabad", "Mumbai"))
CONDITIONS = (("normal", 2.0), ("normal", 22.0), ("summer", 2.0),
              ("summer", 22.0), ("winter", 2.0), ("winter", 22.0))
THREADS = (1, 4, 8)
REPEATS = 10
LATENCY_WARMUP = 100
LATENCY_CALLS = 1000


def percentile_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {"calls": int(values.size), "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99))}


def finite_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compare finite logits without turning identical masked -inf entries into NaN."""
    if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
        return float("inf")
    if not torch.equal(torch.isposinf(left), torch.isposinf(right)):
        return float("inf")
    if not torch.equal(torch.isneginf(left), torch.isneginf(right)):
        return float("inf")
    finite = torch.isfinite(left) & torch.isfinite(right)
    return float((left[finite] - right[finite]).abs().max().item()) if bool(finite.any()) else 0.0


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path) -> LSTMActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(payload.get("metadata", {}).get("hidden_dim", 64))
    model = LSTMActorCritic(hidden_dim=hidden)
    load_checkpoint(path, model, map_location="cpu")
    model.eval()
    return model


def load_gnn_policy(path: Path) -> GNNActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(payload.get("metadata", {}).get("hidden_dim", 64))
    model = GNNActorCritic(hidden_dim=hidden, dropedge_probability=0.05)
    load_checkpoint(path, model, map_location="cpu")
    model.eval()
    model.enable_encoder_cache(False)
    return model


def build_observations() -> list[dict]:
    graph = build_topology()
    observations = []
    for pair_index, (source, destination) in enumerate(PAIRS):
        for condition_index, (season, hour) in enumerate(CONDITIONS):
            for replicate in range(3):
                seed = 700000 + 100 * pair_index + 10 * condition_index + replicate
                env = QKDRoutingEnv(graph=graph, source=source, destination=destination,
                    season=season, use_case="defence", node_feature_mode="relative",
                    time_of_day_hours=hour, time_jitter_hours=0.0,
                    fiber_outage_prob=0.01, qber_hard=0.11, max_steps=144)
                observation, _ = env.reset(seed=seed)
                observations.append({"scenario_id": f"{pair_index}:{condition_index}:{replicate}",
                                     "observation": observation})
    return observations


def measure(model: LSTMActorCritic, observations: list[dict], threads: int) -> dict:
    torch.set_num_threads(threads)
    obs = [row["observation"] for row in observations]

    # Verify the optimization is output-equivalent on all 108 frozen reset states.
    model.enable_encoder_cache(False)
    with torch.inference_mode():
        reference = [(model(ob)[0].clone(), model(ob)[1].clone(),
                      model.act(ob, deterministic=True)[0]) for ob in obs]
    model.enable_encoder_cache(True)
    max_logit_diff = 0.0
    max_value_diff = 0.0
    action_mismatches = 0
    with torch.inference_mode():
        for ob, (expected_logits, expected_value, expected_action) in zip(obs, reference):
            logits, value = model(ob)
            action = model.act(ob, deterministic=True)[0]
            max_logit_diff = max(max_logit_diff,
                finite_abs_difference(logits, expected_logits))
            max_value_diff = max(max_value_diff,
                float((value - expected_value).abs().max().item()))
            action_mismatches += int(action != expected_action)

    # Run warm-up and timed calls across the same 108 scenarios. Node-feature
    # matrices are destination-conditioned and static for this fixed topology;
    # the cache remains bounded at 32 entries and only exact-byte matches hit.
    results = {}
    for enabled in (False, True):
        model.enable_encoder_cache(enabled)
        with torch.inference_mode():
            for ob in obs:
                model.act(ob, deterministic=True)
            samples = []
            for _ in range(REPEATS):
                for ob in obs:
                    start = time.perf_counter_ns()
                    model.act(ob, deterministic=True)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
        results["cache_on" if enabled else "cache_off"] = percentile_summary(samples)
        if enabled:
            results["cache_entries_after_warmup"] = len(model._encoder_cache)

    off = results["cache_off"]
    on = results["cache_on"]
    results["p95_reduction_percent"] = 100.0 * (off["p95_ms"] - on["p95_ms"]) / off["p95_ms"]
    results["p50_reduction_percent"] = 100.0 * (off["p50_ms"] - on["p50_ms"]) / off["p50_ms"]
    results["equivalence"] = {"observations": len(obs),
        "max_logit_abs_difference": max_logit_diff,
        "max_value_abs_difference": max_value_diff,
        "logits_exact_including_mask": all(torch.equal(model(ob)[0], expected[0])
                                            for ob, expected in zip(obs, reference)),
        "deterministic_action_mismatches": action_mismatches}
    return results


def measure_original_latency_protocol(model: LSTMActorCritic, observation: dict,
                                      threads: int) -> dict:
    """Repeat the manuscript's one-observation, 100-warmup/1000-call protocol."""
    torch.set_num_threads(threads)
    results = {}
    outputs = {}
    for enabled in (False, True):
        model.enable_encoder_cache(enabled)
        with torch.no_grad():
            for _ in range(LATENCY_WARMUP):
                model.act(observation, deterministic=True)
            logits, value = model(observation)
            outputs["cache_on" if enabled else "cache_off"] = (
                logits.clone(), value.clone(), model.act(observation, deterministic=True)[0])
            samples = []
            for _ in range(LATENCY_CALLS):
                start = time.perf_counter_ns()
                model.act(observation, deterministic=True)
                samples.append((time.perf_counter_ns() - start) / 1e6)
        results["cache_on" if enabled else "cache_off"] = percentile_summary(samples)
    off, on = outputs["cache_off"], outputs["cache_on"]
    results["equivalence"] = {
        "max_logit_abs_difference": finite_abs_difference(off[0], on[0]),
        "logits_exact_including_mask": torch.equal(off[0], on[0]),
        "max_value_abs_difference": float((off[1] - on[1]).abs().max().item()),
        "deterministic_action_mismatch": int(off[2] != on[2])}
    results["p95_reduction_percent"] = 100.0 * (
        results["cache_off"]["p95_ms"] - results["cache_on"]["p95_ms"]
    ) / results["cache_off"]["p95_ms"]
    return results


def measure_uncached_policy(model: torch.nn.Module, observation: dict,
                            threads: int) -> dict[str, float]:
    """Measure the common deterministic actor-call protocol with encoder caching off."""
    torch.set_num_threads(threads)
    if hasattr(model, "enable_encoder_cache"):
        model.enable_encoder_cache(False)
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            model.act(observation, deterministic=True)
        samples = []
        for _ in range(LATENCY_CALLS):
            start = time.perf_counter_ns()
            model.act(observation, deterministic=True)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return percentile_summary(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-protocol-only", action="store_true",
        help="run only the manuscript's fixed-observation latency replication and append it to the saved report")
    parser.add_argument("--gnn-comparison-only", action="store_true",
        help="measure the five matched GNN checkpoints on the manuscript's fixed-observation latency protocol")
    args = parser.parse_args()
    torch.set_num_threads(4)
    parent = psutil.Process()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    observations = build_observations()
    latency_env = QKDRoutingEnv(graph=build_topology(), source="Delhi", destination="Mumbai",
        season="normal", use_case="defence", node_feature_mode="relative",
        time_of_day_hours=2.0, time_jitter_hours=0.0, fiber_outage_prob=0.01,
        qber_hard=0.11, max_steps=144)
    latency_observation, _ = latency_env.reset(seed=700000)

    if args.gnn_comparison_only:
        output = json.loads(OUT.read_text(encoding="utf-8"))
        gnn_report = json.loads(GNN_REPORT.read_text(encoding="utf-8"))
        gnn_rows = []
        for item in gnn_report["trained_checkpoint_manifest"]:
            if item["architecture"] != "GNN":
                continue
            checkpoint = ROOT / item["checkpoint"]
            model = load_gnn_policy(checkpoint)
            seed_row = {"training_seed": int(item["training_seed"]),
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "checkpoint_sha256": hash_file(checkpoint), "threads": {}}
            for threads in (1, 4):
                seed_row["threads"][str(threads)] = measure_uncached_policy(
                    model, latency_observation, threads)
            gnn_rows.append(seed_row)
        architecture_comparison = {}
        for threads in (1, 4):
            gnn_p95 = [r["threads"][str(threads)]["p95_ms"] for r in gnn_rows]
            lstm_off = [r["original_latency_protocol"][str(threads)]["cache_off"]["p95_ms"]
                        for r in output["per_checkpoint"]]
            lstm_on = [r["original_latency_protocol"][str(threads)]["cache_on"]["p95_ms"]
                       for r in output["per_checkpoint"]]
            gnn_mean = float(np.mean(gnn_p95))
            lstm_off_mean = float(np.mean(lstm_off))
            lstm_on_mean = float(np.mean(lstm_on))
            architecture_comparison[str(threads)] = {
                "gnn_uncached_mean_checkpoint_p95_ms": gnn_mean,
                "gnn_uncached_sd_checkpoint_p95_ms": float(np.std(gnn_p95, ddof=1)),
                "lstm_uncached_mean_checkpoint_p95_ms": lstm_off_mean,
                "lstm_cached_mean_checkpoint_p95_ms": lstm_on_mean,
                "cached_lstm_p95_reduction_vs_gnn_percent":
                    100.0 * (gnn_mean - lstm_on_mean) / gnn_mean,
                "cached_lstm_p95_reduction_vs_uncached_lstm_percent":
                    100.0 * (lstm_off_mean - lstm_on_mean) / lstm_off_mean}
        output["gnn_per_checkpoint_original_latency_protocol"] = gnn_rows
        output["matched_architecture_latency_comparison"] = architecture_comparison
        output["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"matched_architecture_latency_comparison": architecture_comparison,
                          "output": str(OUT)}, indent=2))
        return

    if args.original_protocol_only:
        output = json.loads(OUT.read_text(encoding="utf-8"))
        for row in output["per_checkpoint"]:
            checkpoint = ROOT / row["checkpoint"]
            model = load_policy(checkpoint)
            row["original_latency_protocol"] = {str(threads):
                measure_original_latency_protocol(model, latency_observation, threads)
                for threads in THREADS}
        original_protocol_aggregate = {}
        for threads in THREADS:
            off = [row["original_latency_protocol"][str(threads)]["cache_off"]
                   for row in output["per_checkpoint"]]
            on = [row["original_latency_protocol"][str(threads)]["cache_on"]
                  for row in output["per_checkpoint"]]
            reductions = [row["original_latency_protocol"][str(threads)][
                "p95_reduction_percent"] for row in output["per_checkpoint"]]
            original_protocol_aggregate[str(threads)] = {
                "cache_off_mean_checkpoint_p50_ms": float(np.mean([x["p50_ms"] for x in off])),
                "cache_off_mean_checkpoint_p95_ms": float(np.mean([x["p95_ms"] for x in off])),
                "cache_off_sd_checkpoint_p95_ms": float(np.std([x["p95_ms"] for x in off], ddof=1)),
                "cache_on_mean_checkpoint_p50_ms": float(np.mean([x["p50_ms"] for x in on])),
                "cache_on_mean_checkpoint_p95_ms": float(np.mean([x["p95_ms"] for x in on])),
                "cache_on_sd_checkpoint_p95_ms": float(np.std([x["p95_ms"] for x in on], ddof=1)),
                "mean_checkpoint_p95_reduction_percent": float(np.mean(reductions)),
                "all_outputs_exact": all(
                    row["original_latency_protocol"][str(threads)]["equivalence"][
                        "max_logit_abs_difference"] == 0.0
                    and row["original_latency_protocol"][str(threads)]["equivalence"][
                        "logits_exact_including_mask"]
                    and row["original_latency_protocol"][str(threads)]["equivalence"][
                        "max_value_abs_difference"] == 0.0
                    and row["original_latency_protocol"][str(threads)]["equivalence"][
                        "deterministic_action_mismatch"] == 0
                    for row in output["per_checkpoint"])}
        output["original_latency_protocol_aggregate_by_threads"] = original_protocol_aggregate
        output["protocol"]["original_latency_protocol_replication"] = {
            "observation": "Delhi-to-Mumbai, normal, 02:00, seed 700000",
            "warmup_calls": LATENCY_WARMUP, "measured_calls": LATENCY_CALLS}
        output["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"original_latency_protocol_aggregate_by_threads":
                          original_protocol_aggregate, "output": str(OUT)}, indent=2))
        return

    rows = []
    for item in report["checkpoint_manifest"]:
        if item["architecture"] != "LSTM":
            continue
        checkpoint = ROOT / item["checkpoint"]
        model = load_policy(checkpoint)
        seed_row = {"training_seed": int(item["training_seed"]),
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": hash_file(checkpoint), "threads": {}}
        for threads in THREADS:
            seed_row["threads"][str(threads)] = measure(model, observations, threads)
        seed_row["original_latency_protocol"] = {str(threads):
            measure_original_latency_protocol(model, latency_observation, threads)
            for threads in THREADS}
        rows.append(seed_row)

    aggregate = {}
    for threads in THREADS:
        aggregate[str(threads)] = {}
        for mode in ("cache_off", "cache_on"):
            values = [row["threads"][str(threads)][mode] for row in rows]
            aggregate[str(threads)][mode] = {
                "checkpoint_mean_p50_ms": float(np.mean([x["p50_ms"] for x in values])),
                "checkpoint_sd_p50_ms": float(np.std([x["p50_ms"] for x in values], ddof=1)),
                "checkpoint_mean_p95_ms": float(np.mean([x["p95_ms"] for x in values])),
                "checkpoint_sd_p95_ms": float(np.std([x["p95_ms"] for x in values], ddof=1)),
                "checkpoint_mean_p99_ms": float(np.mean([x["p99_ms"] for x in values])),
            }
        aggregate[str(threads)]["mean_checkpoint_p95_reduction_percent"] = float(np.mean([
            row["threads"][str(threads)]["p95_reduction_percent"] for row in rows]))
        aggregate[str(threads)]["all_outputs_exact"] = all(
            row["threads"][str(threads)]["equivalence"]["max_logit_abs_difference"] == 0.0
            and row["threads"][str(threads)]["equivalence"]["logits_exact_including_mask"]
            and row["threads"][str(threads)]["equivalence"]["max_value_abs_difference"] == 0.0
            and row["threads"][str(threads)]["equivalence"]["deterministic_action_mismatches"] == 0
            for row in rows)

    original_protocol_aggregate = {}
    for threads in THREADS:
        off = [row["original_latency_protocol"][str(threads)]["cache_off"] for row in rows]
        on = [row["original_latency_protocol"][str(threads)]["cache_on"] for row in rows]
        reductions = [row["original_latency_protocol"][str(threads)]["p95_reduction_percent"]
                      for row in rows]
        original_protocol_aggregate[str(threads)] = {
            "cache_off_mean_checkpoint_p50_ms": float(np.mean([x["p50_ms"] for x in off])),
            "cache_off_mean_checkpoint_p95_ms": float(np.mean([x["p95_ms"] for x in off])),
            "cache_off_sd_checkpoint_p95_ms": float(np.std([x["p95_ms"] for x in off], ddof=1)),
            "cache_on_mean_checkpoint_p50_ms": float(np.mean([x["p50_ms"] for x in on])),
            "cache_on_mean_checkpoint_p95_ms": float(np.mean([x["p95_ms"] for x in on])),
            "cache_on_sd_checkpoint_p95_ms": float(np.std([x["p95_ms"] for x in on], ddof=1)),
            "mean_checkpoint_p95_reduction_percent": float(np.mean(reductions)),
            "all_outputs_exact": all(
                row["original_latency_protocol"][str(threads)]["equivalence"][
                    "max_logit_abs_difference"] == 0.0
                and row["original_latency_protocol"][str(threads)]["equivalence"][
                    "logits_exact_including_mask"]
                and row["original_latency_protocol"][str(threads)]["equivalence"][
                    "max_value_abs_difference"] == 0.0
                and row["original_latency_protocol"][str(threads)]["equivalence"][
                    "deterministic_action_mismatch"] == 0
                for row in rows)}

    output = {"status": "complete", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": str(REPORT.relative_to(ROOT)),
        "protocol": {"checkpoints": len(rows), "matched_training_seeds": [r["training_seed"] for r in rows],
            "scenarios_per_checkpoint": len(observations), "thread_counts": list(THREADS),
            "measured_repeats_per_scenario": REPEATS,
            "original_latency_protocol_replication": {"observation": "Delhi-to-Mumbai, normal, 02:00, seed 700000",
                "warmup_calls": LATENCY_WARMUP, "measured_calls": LATENCY_CALLS},
            "timed_operation": "one deterministic LSTM-PPO actor call (forward, masked distribution, argmax)",
            "warmup": "one pass through the 108 held-out reset observations per cache mode",
            "cache": "exact-byte node-feature key; max 32 embeddings; encoder-only memoization",
            "environment": "same frozen 108-scenario matrix, CPU, no environment stepping timed",
            "note": "Warm-cache latency applies after a feature matrix has been encoded; cold miss cost is not included in the reported warm-call quantiles."},
        "host": {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False), "torch_version": torch.__version__,
            "other_python_processes": [{"pid": p.pid, "name": p.info.get("name")}
                for p in psutil.process_iter(["name"]) if p.pid != parent.pid and
                (p.info.get("name") or "").lower().startswith("python")]},
        "aggregate_by_threads": aggregate,
        "original_latency_protocol_aggregate_by_threads": original_protocol_aggregate,
        "per_checkpoint": rows}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate_by_threads": aggregate,
        "original_latency_protocol_aggregate_by_threads": original_protocol_aggregate,
        "equivalence": {str(t): [r["threads"][str(t)]["equivalence"] for r in rows]
                        for t in THREADS},
        "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
