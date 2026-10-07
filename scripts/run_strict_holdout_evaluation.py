"""Tune weighted-SP on fixed validation episodes and evaluate frozen holdout policies."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import BASELINE_NAMES, BaselinePolicy
from evaluation import _run_episode
from models import GNNActorCritic, MLPActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology

PAIRS = (("Delhi", "Mumbai"), ("Mumbai", "Chennai"), ("Kolkata", "Delhi"),
         ("Bangalore", "Hyderabad"), ("Jaipur", "Kolkata"), ("Hyderabad", "Mumbai"))
CONDITIONS = (("normal", 2.0), ("normal", 22.0), ("summer", 2.0),
              ("summer", 22.0), ("winter", 2.0), ("winter", 22.0))
REPLICATES = 3
WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)
TUNING_EPISODES = len(WEIGHT_GRID) ** 3 * 4
METHOD_COUNT = 5 + 2 * 5
SCENARIO_COUNT = len(PAIRS) * len(CONDITIONS) * REPLICATES
EVALUATION_EPISODES = SCENARIO_COUNT * METHOD_COUNT * (1 + 4)
TOTAL_EPISODES = TUNING_EPISODES + EVALUATION_EPISODES
TOTAL_UNITS = TOTAL_EPISODES + (2 * 5) + 1  # episodes, latency per checkpoint, final report


class ProgressJournal:
    """Append-only episode checkpoints plus an atomic live progress snapshot."""

    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.status_path = path.with_suffix(".status.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        valid_lines = []
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                valid_lines.append(line)
                if record.get("kind") != "metadata":
                    self.records[record["id"]] = record
            if len(valid_lines) != len(lines):
                path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""),
                                encoding="utf-8")
        expected = {"kind": "metadata", **metadata}
        if valid_lines:
            existing = json.loads(valid_lines[0])
            if existing != expected:
                raise RuntimeError(f"progress journal metadata differs: {path}")
        else:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(expected, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.completed_episodes = sum(
            record.get("kind") in {"tuning_episode", "evaluation_episode"}
            for record in self.records.values())
        self.completed_latency_models = sum(
            record.get("kind") == "latency_checkpoint" for record in self.records.values())
        self.report_complete = any(
            record.get("kind") == "report_complete" for record in self.records.values())
        self.write_status("starting", None)

    def write_status(self, stage: str, current_id: str | None):
        status = {
            "stage": stage,
            "current_unit": current_id,
            "completed_episodes": self.completed_episodes,
            "total_episodes": TOTAL_EPISODES,
            "completed_latency_models": self.completed_latency_models,
            "total_latency_models": 10,
            "report_complete": self.report_complete,
            "completed_units": (self.completed_episodes + self.completed_latency_models
                                + int(self.report_complete)),
            "total_units": TOTAL_UNITS,
            "percent": round(100 * (self.completed_episodes + self.completed_latency_models
                                     + int(self.report_complete)) / TOTAL_UNITS, 2),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        filled = round(status["percent"] / 10)
        status["progress_bar"] = "[" + "█" * filled + "░" * (10 - filled) + "]"
        # Append-only snapshots avoid Windows replace/locking failures and keep
        # a small history of the exact stage and active unit.
        with self.status_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status, separators=(",", ":")) + "\n")
            stream.flush()

    def append(self, record: dict):
        if record["id"] in self.records:
            return self.records[record["id"]]
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[record["id"]] = record
        if record.get("kind") in {"tuning_episode", "evaluation_episode"}:
            self.completed_episodes += 1
        elif record.get("kind") == "latency_checkpoint":
            self.completed_latency_models += 1
        elif record.get("kind") == "report_complete":
            self.report_complete = True
        self.write_status(record.get("stage", "unknown"), record["id"])
        if (record.get("kind") in {"tuning_episode", "evaluation_episode"}
                and (self.completed_episodes % 10 == 0
                     or self.completed_episodes == TOTAL_EPISODES)):
            print(f"[PROGRESS] {self.completed_episodes}/{TOTAL_EPISODES} episodes "
                  f"({100*self.completed_episodes/TOTAL_EPISODES:.1f}%) "
                  f"stage={record.get('stage')} last={record['id']}", flush=True)
        return record

    def set_running(self, stage: str, current_id: str):
        self.write_status(stage, current_id)
        if stage != getattr(self, "last_printed_stage", None) or self.completed_episodes % 10 == 0:
            self.last_printed_stage = stage
            completed_units = self.completed_episodes + self.completed_latency_models
            print(f"[RUNNING] stage={stage} unit={current_id} "
                  f"progress={completed_units}/{TOTAL_UNITS} "
                  f"episodes={self.completed_episodes}/{TOTAL_EPISODES} "
                  f"({100*completed_units/TOTAL_UNITS:.1f}%)", flush=True)


def make_env(graph, source, destination, season, hour, seed, fiber_outage_prob=0.01):
    env = QKDRoutingEnv(graph=graph, source=source, destination=destination,
        season=season, use_case="defence", node_feature_mode="relative",
        time_of_day_hours=hour, time_jitter_hours=0.0, fiber_outage_prob=fiber_outage_prob,
        qber_hard=0.11, max_steps=144)
    # _run_episode resets with this seed immediately before the first action.
    # Resetting here too repeated the full graph-state initialization twice.
    return env


def tune_weighted_sp(graph, journal: ProgressJournal):
    candidates = list(itertools.product(WEIGHT_GRID, repeat=3))
    rows = {weights: [] for weights in candidates}
    costs = {weights: [] for weights in candidates}
    for seed in (11001, 11002, 11003, 11004):
        for weights in candidates:
            weights_key = ",".join(map(str, weights))
            episode_id = f"tune|seed={seed}|weights={weights_key}"
            saved = journal.records.get(episode_id)
            if saved is None:
                journal.set_running("weighted_sp_validation", episode_id)
                env = make_env(graph, "Delhi", "Chennai", "monsoon", 22.0, seed)
                policy = BaselinePolicy("Weighted-SP", weighted_sp_weights=weights)
                episode = _run_episode(policy, env, seed)
                cost = float(getattr(env, "_weighted_sp_realized_cost", 0.0))
                saved = journal.append({"id": episode_id, "kind": "tuning_episode",
                    "stage": "weighted_sp_validation", "seed": seed,
                    "weights": list(weights), "episode": episode, "cost": cost})
            rows[weights].append(saved["episode"])
            costs[weights].append(float(saved["cost"]))
    def score(weights):
        episodes = rows[weights]
        success = sum(r["success"] for r in episodes) / len(episodes)
        hops = [r["hops"] for r in episodes if r["success"]]
        return (-success, statistics.mean(hops) if hops else float("inf"),
                statistics.mean(costs[weights]), weights)
    selected = min(candidates, key=score)
    tuning = [{"weights_q_rate_pool": list(weights),
               "success_rate": sum(r["success"] for r in rows[weights]) / len(rows[weights]),
               "mean_realized_weighted_cost": statistics.mean(costs[weights]),
               "successful_mean_hops": (float(np.mean([r["hops"] for r in rows[weights]
                   if r["success"]])) if any(r["success"] for r in rows[weights]) else None)}
              for weights in candidates]
    return selected, tuning


def load_policy(path: Path, architecture: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    hidden = int(metadata.get("hidden_dim", 64))
    if architecture == "GNN":
        model = GNNActorCritic(hidden_dim=hidden, dropedge_probability=0.05)
    else:
        model = MLPActorCritic(hidden_dim=hidden)
    load_checkpoint(path, model, map_location="cpu")
    model.eval()
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows):
    n = len(rows)
    success = sum(bool(r["success"]) for r in rows)
    hops = [int(r["hops"]) for r in rows if r["success"]]
    return {"episodes": n, "successes": success,
            "success_rate": success / n if n else None,
            "successful_mean_hops": float(np.mean(hops)) if hops else None,
            "successful_median_hops": float(np.median(hops)) if hops else None,
            "mean_reward": float(np.mean([r["reward"] for r in rows])) if rows else None,
            "invalid_actions": sum(r["invalid_action_count"] for r in rows),
            "fallbacks": sum(r["fallback_count"] for r in rows),
            "no_valid_action_states": sum(r["no_valid_action_states"] for r in rows)}


def scenario_bootstrap_success(rows, repeats=2000, seed=810027):
    """Scenario-resampling CI for one fixed checkpoint; training seeds stay separate."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(bool(row["success"]))
    per_scenario = np.asarray([np.mean(v) for v in groups.values()], dtype=float)
    if not len(per_scenario):
        return {"scenarios": 0, "success_rate": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    samples = rng.choice(per_scenario, size=(repeats, len(per_scenario)), replace=True).mean(axis=1)
    return {"scenarios": int(len(per_scenario)),
            "success_rate": float(per_scenario.mean()),
            "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])]}


def nested_training_scenario_bootstrap(seed_groups, repeats=2000, seed=810028):
    """Resample training seeds outside and paired scenario clusters inside."""
    training_seeds = sorted(seed_groups)
    if not training_seeds:
        return {"training_seeds": 0, "ci95": [None, None]}
    per_seed = {}
    for training_seed in training_seeds:
        scenarios = defaultdict(list)
        for row in seed_groups[training_seed]:
            scenarios[row["scenario_id"]].append(bool(row["success"]))
        per_seed[training_seed] = np.asarray([np.mean(v) for v in scenarios.values()])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        outer = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        means = []
        for training_seed in outer:
            values = per_seed[int(training_seed)]
            means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        samples[i] = np.mean(means)
    point = np.mean([v.mean() for v in per_seed.values()])
    return {"training_seeds": len(training_seeds),
            "scenarios_per_seed": min(len(v) for v in per_seed.values()),
            "success_rate": float(point),
            "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "method": f"{repeats} replicates; training seed outer, scenario cluster inner"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-campaign", type=Path, required=True)
    parser.add_argument("--mlp-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "final_holdout_20261007.json")
    parser.add_argument("--progress-file", type=Path,
                        help="Append-only resume journal (defaults beside --output).")
    args = parser.parse_args()
    torch.set_num_threads(4)
    graph = build_topology()
    learned = []
    checkpoint_manifest = []
    for architecture, campaign in (("GNN", args.gnn_campaign), ("MLP", args.mlp_campaign)):
        for seed in (20261002, 20261003, 20261004, 20261005, 20261006):
            run_info = json.loads((campaign / "seed_runs.json").read_text(encoding="utf-8"))
            matches = [r for r in run_info["runs"] if r["training_seed"] == seed]
            if len(matches) != 1 or matches[0]["status"] != "complete":
                raise RuntimeError(f"missing completed {architecture} run for seed {seed}")
            run_dir = ROOT / matches[0]["run_dir"]
            checkpoint = run_dir / "checkpoints" / architecture / "best_by_success_rate.pt"
            if not checkpoint.exists():
                checkpoint = run_dir / "checkpoints" / architecture / "latest.pt"
            learned.append((architecture, seed, checkpoint, load_policy(checkpoint, architecture)))
            checkpoint_manifest.append({"architecture": architecture, "training_seed": seed,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint)})

    baseline_names = ("Random", "Dijkstra-km", "BFS-hop", "Max-SKR", "Weighted-SP")
    protocol_path = ROOT / "paper/supplementary/final_experiment_protocol_20261003.md"
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    progress_path = args.progress_file or args.output.with_suffix(".progress.jsonl")
    journal = ProgressJournal(progress_path, {
        "protocol_sha256": protocol_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "scenarios": SCENARIO_COUNT,
        "methods_per_scenario": METHOD_COUNT,
        "tuning_episodes": TUNING_EPISODES,
        "outage_levels": [0.0, 0.01, 0.05, 0.10],
    })
    weights, tuning = tune_weighted_sp(graph, journal)

    rows = []
    for pair_index, (source, destination) in enumerate(PAIRS):
        for condition_index, (season, hour) in enumerate(CONDITIONS):
            for replicate in range(REPLICATES):
                scenario_id = f"{pair_index}:{condition_index}:{replicate}"
                seed = 700000 + 100 * pair_index + 10 * condition_index + replicate
                methods = [(name, None, BaselinePolicy(name, seed=seed + 917))
                           for name in baseline_names if name != "Weighted-SP"]
                methods.append(("Weighted-SP", None,
                    BaselinePolicy("Weighted-SP", weighted_sp_weights=weights)))
                methods.extend((f"{architecture}-PPO", training_seed, policy)
                    for architecture, training_seed, checkpoint, policy in learned)
                for method, training_seed, policy in methods:
                    episode_id = f"heldout|{scenario_id}|{method}|seed={training_seed}"
                    saved = journal.records.get(episode_id)
                    if saved is None:
                        journal.set_running("heldout_matrix", episode_id)
                        env = make_env(graph, source, destination, season, hour, seed)
                        result = _run_episode(policy, env, seed)
                        row = {"method": method, "architecture": method.split("-")[0],
                            "training_seed": training_seed, "scenario_id": scenario_id,
                            "source": source, "destination": destination, "season": season,
                            "hour": hour, "replicate": replicate, "seed": seed, **result}
                        saved = journal.append({"id": episode_id,
                            "kind": "evaluation_episode", "stage": "heldout_matrix",
                            "episode": row})
                    rows.append(saved["episode"])

    by_method = defaultdict(list)
    by_seed = defaultdict(list)
    by_condition = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        if row["training_seed"] is not None:
            by_seed[(row["method"], row["training_seed"])].append(row)
        by_condition[(row["method"], row["season"], int(row["hour"]))].append(row)

    stress_rows = []
    for outage_index, outage_probability in enumerate((0.0, 0.01, 0.05, 0.10)):
        for pair_index, (source, destination) in enumerate(PAIRS):
            for condition_index, (season, hour) in enumerate(CONDITIONS):
                for replicate in range(REPLICATES):
                    scenario_id = f"{pair_index}:{condition_index}:{replicate}"
                    # Hold IDs and RNG seeds fixed across outage probabilities (paired stress).
                    seed = 800000 + 100*pair_index + 10*condition_index + replicate
                    methods = [(name, None, BaselinePolicy(name, seed=seed + 917))
                               for name in baseline_names if name != "Weighted-SP"]
                    methods.append(("Weighted-SP", None,
                        BaselinePolicy("Weighted-SP", weighted_sp_weights=weights)))
                    methods.extend((f"{architecture}-PPO", training_seed, policy)
                        for architecture, training_seed, checkpoint, policy in learned)
                    for method, training_seed, policy in methods:
                        episode_id = (f"stress|outage={outage_probability:.2f}|"
                                      f"{scenario_id}|{method}|seed={training_seed}")
                        saved = journal.records.get(episode_id)
                        if saved is None:
                            journal.set_running(f"outage_{outage_probability:.2f}", episode_id)
                            env = make_env(graph, source, destination, season, hour, seed,
                                           fiber_outage_prob=outage_probability)
                            result = _run_episode(policy, env, seed)
                            row = {"method": method,
                                "training_seed": training_seed, "scenario_id": scenario_id,
                                "fiber_outage_probability": outage_probability,
                                "source": source, "destination": destination,
                                "season": season, "hour": hour, "replicate": replicate,
                                "seed": seed, **result}
                            saved = journal.append({"id": episode_id,
                                "kind": "evaluation_episode",
                                "stage": f"outage_{outage_probability:.2f}",
                                "episode": row})
                        stress_rows.append(saved["episode"])
        print(f"Completed fiber-outage stress probability {outage_probability:.2f}", flush=True)

    stress_by_method_outage = defaultdict(list)
    stress_by_seed_outage = defaultdict(list)
    for row in stress_rows:
        stress_by_method_outage[(row["method"], row["fiber_outage_probability"])].append(row)
        if row["training_seed"] is not None:
            stress_by_seed_outage[(row["method"], row["fiber_outage_probability"],
                                   row["training_seed"])].append(row)

    # Actual deterministic actor-call latency, cache disabled, on a held-out observation.
    import platform
    import time
    latency = {}
    latency_env = make_env(graph, "Delhi", "Mumbai", "normal", 2.0, 700000)
    latency_obs, _ = latency_env.reset(seed=700000)
    for architecture, training_seed, checkpoint, policy in learned:
        latency_key = f"{architecture}|{training_seed}"
        latency_id = f"latency|{latency_key}"
        saved = journal.records.get(latency_id)
        if saved is not None:
            latency[latency_key] = saved["latency"]
            continue
        journal.set_running("inference_latency", latency_id)
        if hasattr(policy, "enable_encoder_cache"):
            policy.enable_encoder_cache(False)
        policy.eval()
        with torch.no_grad():
            for _ in range(100):
                policy.act(latency_obs, deterministic=True)
            elapsed_ms = []
            for _ in range(1000):
                started = time.perf_counter_ns()
                policy.act(latency_obs, deterministic=True)
                elapsed_ms.append((time.perf_counter_ns()-started)/1e6)
        latency[latency_key] = {
            "host": platform.platform(), "device": "CPU",
            "torch_threads": torch.get_num_threads(), "warmup_calls": 100,
            "measured_calls": 1000, "encoder_cache": "disabled",
            "mean_ms": float(np.mean(elapsed_ms)),
            "p50_ms": float(np.quantile(elapsed_ms, 0.50)),
            "p95_ms": float(np.quantile(elapsed_ms, 0.95)),
            "p99_ms": float(np.quantile(elapsed_ms, 0.99))}
        journal.append({"id": latency_id, "kind": "latency_checkpoint",
            "stage": "inference_latency", "latency": latency[latency_key]})

    baseline_episodes = {name: by_method[name] for name in baseline_names}
    environment_uncertainty = {f"{m}|training_seed={s}": scenario_bootstrap_success(v)
        for (m, s), v in by_seed.items()}
    training_seed_rates = defaultdict(list)
    for (method, training_seed), episodes in by_seed.items():
        training_seed_rates[method].append(summarize(episodes)["success_rate"])
    seed_rows = defaultdict(dict)
    for (method, training_seed), episodes in by_seed.items():
        seed_rows[method][training_seed] = episodes
    report = {"status": "complete", "protocol": "final_experiment_protocol_20261003.md",
        "protocol_sha256": protocol_sha256,
        "progress_journal": str(progress_path),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "endpoint_pairs": [list(p) for p in PAIRS],
        "trained_checkpoint_manifest": checkpoint_manifest,
        "conditions": [{"season": s, "hour": h} for s, h in CONDITIONS],
        "unique_scenario_count": len(PAIRS) * len(CONDITIONS) * REPLICATES,
        "weighted_sp_weights_q_rate_pool": list(weights), "weighted_sp_validation_grid": tuning,
        "baseline_summary": {k: summarize(v) for k, v in baseline_episodes.items()},
        "baseline_scenario_bootstrap_success_rate_ci95": {
            method: scenario_bootstrap_success(rows_for_method)
            for method, rows_for_method in baseline_episodes.items()},
        "summary_by_method_training_seed": {f"{m}|{s}": summarize(v)
            for (m, s), v in by_seed.items()},
        "environment_seed_uncertainty_success_rate_ci95": environment_uncertainty,
        "nested_training_and_scenario_bootstrap_success_rate_ci95": {
            method: nested_training_scenario_bootstrap(groups)
            for method, groups in seed_rows.items()},
        "training_seed_uncertainty_success_rate": {method: {
            "training_seeds": len(values), "mean": float(np.mean(values)),
            "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": values} for method, values in training_seed_rates.items()},
        "summary_by_method_condition": {f"{m}|{season}|{hour:02d}h": summarize(v)
            for (m, season, hour), v in by_condition.items()},
        "fiber_outage_stress_levels": [0.0, 0.01, 0.05, 0.10],
        "fiber_outage_stress_summary": {f"{m}|outage={p:.2f}": summarize(v)
            for (m, p), v in stress_by_method_outage.items()},
        "fiber_outage_stress_summary_by_training_seed": {
            f"{m}|outage={p:.2f}|seed={s}": summarize(v)
            for (m, p, s), v in stress_by_seed_outage.items()},
        "fiber_outage_stress_episodes": stress_rows,
        "inference_latency": latency,
        "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal.set_running("report_assembly", "aggregate_and_write_final_json")
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp_output.replace(args.output)
    journal.append({"id": "report|complete", "kind": "report_complete",
        "stage": "complete", "output": str(args.output)})
    print(json.dumps({"output": str(args.output),
        "summary_by_method_training_seed": report["summary_by_method_training_seed"]}, indent=2))


if __name__ == "__main__":
    main()
