"""Evaluate five trained LSTM-PPO checkpoints on the frozen GNN holdout scenarios."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import _run_episode
from models import LSTMActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology

SEEDS = (20261002, 20261003, 20261004, 20261005, 20261006)
PAIRS = (("Delhi", "Mumbai"), ("Mumbai", "Chennai"), ("Kolkata", "Delhi"),
         ("Bangalore", "Hyderabad"), ("Jaipur", "Kolkata"), ("Hyderabad", "Mumbai"))
CONDITIONS = (("normal", 2.0), ("normal", 22.0), ("summer", 2.0),
              ("summer", 22.0), ("winter", 2.0), ("winter", 22.0))
REPLICATES = 3
SCENARIO_COUNT = len(PAIRS) * len(CONDITIONS) * REPLICATES
TOTAL_UNITS = SCENARIO_COUNT * len(SEEDS) + len(SEEDS) + 1


def make_env(graph, source, destination, season, hour):
    return QKDRoutingEnv(graph=graph, source=source, destination=destination,
        season=season, use_case="defence", node_feature_mode="relative",
        time_of_day_hours=hour, time_jitter_hours=0.0, fiber_outage_prob=0.01,
        qber_hard=0.11, max_steps=144)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(payload.get("metadata", {}).get("hidden_dim", 64))
    model = LSTMActorCritic(hidden_dim=hidden)
    load_checkpoint(path, model, map_location="cpu")
    model.eval()
    # Node features are constant within a fixed-destination episode; cache only
    # their deterministic sequence embeddings while scoring every action anew.
    model.enable_encoder_cache(True)
    return model


def summarize(rows):
    successes = [r for r in rows if r["success"]]
    hops = [int(r["hops"]) for r in successes]
    return {"episodes": len(rows), "successes": len(successes),
            "success_rate": len(successes) / len(rows) if rows else None,
            "successful_mean_hops": float(np.mean(hops)) if hops else None,
            "successful_median_hops": float(np.median(hops)) if hops else None,
            "invalid_actions": sum(r["invalid_action_count"] for r in rows),
            "fallbacks": sum(r["fallback_count"] for r in rows),
            "no_valid_action_states": sum(r["no_valid_action_states"] for r in rows)}


def nested_bootstrap(seed_groups, repeats=10000, seed=810029):
    per_seed = {}
    for train_seed, rows in seed_groups.items():
        by_scenario = defaultdict(list)
        for row in rows:
            by_scenario[row["scenario_id"]].append(float(bool(row["success"])))
        per_seed[train_seed] = np.asarray([np.mean(v) for v in by_scenario.values()])
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    train_seeds = sorted(per_seed)
    for i in range(repeats):
        selected = rng.choice(train_seeds, size=len(train_seeds), replace=True)
        values = []
        for train_seed in selected:
            scenarios = per_seed[int(train_seed)]
            values.append(float(rng.choice(scenarios, size=len(scenarios), replace=True).mean()))
        draws[i] = np.mean(values)
    point = float(np.mean([x.mean() for x in per_seed.values()]))
    return {"training_seeds": len(train_seeds), "scenarios_per_seed": SCENARIO_COUNT,
            "mean": point, "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "bootstrap_replicates": repeats}


def paired_nested_difference(gnn_groups, lstm_groups, repeats=10000, seed=810030):
    differences = {}
    for training_seed in SEEDS:
        gnn = {r["scenario_id"]: float(bool(r["success"])) for r in gnn_groups[training_seed]}
        lstm = {r["scenario_id"]: float(bool(r["success"])) for r in lstm_groups[training_seed]}
        if set(gnn) != set(lstm):
            raise RuntimeError(f"GNN/LSTM held-out scenarios differ for seed {training_seed}")
        differences[training_seed] = np.asarray([lstm[s] - gnn[s] for s in sorted(gnn)])
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for i in range(repeats):
        selected = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        means = []
        for training_seed in selected:
            values = differences[int(training_seed)]
            means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        draws[i] = np.mean(means)
    point = float(np.mean([x.mean() for x in differences.values()]))
    return {"difference_definition": "LSTM-PPO success minus GNN-PPO success",
            "mean_difference": point,
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "bootstrap_replicates": repeats,
            "method": "paired scenario resampling within independently resampled training seeds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path,
        default=ROOT / "experiments" / "final_holdout_lstm_20261007")
    parser.add_argument("--gnn-report", type=Path,
        default=ROOT / "experiments" / "final_holdout_20261007.json")
    parser.add_argument("--output", type=Path,
        default=ROOT / "experiments" / "final_gnn_lstm_holdout_20261007.json")
    parser.add_argument("--journal", type=Path,
        default=ROOT / "experiments" / "final_gnn_lstm_holdout_20261007.progress.jsonl")
    args = parser.parse_args()
    campaign = args.campaign_dir if args.campaign_dir.is_absolute() else ROOT / args.campaign_dir
    gnn_path = args.gnn_report if args.gnn_report.is_absolute() else ROOT / args.gnn_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    journal_path = args.journal if args.journal.is_absolute() else ROOT / args.journal
    status_path = journal_path.with_suffix(".status.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    gnn_report = json.loads(gnn_path.read_text(encoding="utf-8"))
    gnn_groups = defaultdict(list)
    for row in gnn_report["episodes"]:
        if row["method"] == "GNN-PPO":
            gnn_groups[int(row["training_seed"])].append(row)
    if set(gnn_groups) != set(SEEDS) or any(len(v) != SCENARIO_COUNT for v in gnn_groups.values()):
        raise RuntimeError("The archived GNN report does not contain the expected five matched checkpoints")

    seed_report = json.loads((campaign / "seed_runs.json").read_text(encoding="utf-8"))
    learned = []
    manifest = []
    by_training_seed = {int(r["training_seed"]): r for r in seed_report.get("runs", [])}
    for training_seed in SEEDS:
        record = by_training_seed.get(training_seed)
        if not record or record.get("status") != "complete":
            raise RuntimeError(f"Missing completed LSTM training run for seed {training_seed}")
        run_dir = ROOT / record["run_dir"]
        checkpoint = run_dir / "checkpoints" / "LSTM" / "best_by_success_rate.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "checkpoints" / "LSTM" / "latest.pt"
        learned.append((training_seed, checkpoint, load_policy(checkpoint)))
        manifest.append({"architecture": "LSTM", "training_seed": training_seed,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint)})

    journal_rows = {}
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            if rec.get("kind") == "episode":
                journal_rows[rec["id"]] = rec["row"]
            elif rec.get("kind") == "latency":
                journal_rows[rec["id"]] = rec["latency"]
            elif rec.get("kind") == "report_complete":
                journal_rows[rec["id"]] = {"status": "complete"}

    completed = sum(not key.startswith("latency|") and key != "report|complete"
                    for key in journal_rows)

    def status(stage, unit):
        nonlocal completed
        record = {"stage": stage, "current_unit": unit,
            "completed_episodes": completed, "total_episodes": SCENARIO_COUNT * len(SEEDS),
            "completed_checkpoints": sum(k.startswith("latency|") for k in journal_rows),
            "total_checkpoints": len(SEEDS),
            "percent": round(100 * (completed + sum(k.startswith("latency|") for k in journal_rows)
                                  + int("report|complete" in journal_rows)) / TOTAL_UNITS, 2),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        filled = round(record["percent"] / 10)
        # Use ASCII so progress logging also works in Windows cp1252 consoles.
        record["progress_bar"] = "[" + "#" * filled + "." * (10-filled) + "]"
        with status_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
        print(f"{record['progress_bar']} {record['percent']:.1f}% "
              f"episodes={completed}/{SCENARIO_COUNT*len(SEEDS)} stage={stage} unit={unit}",
              flush=True)

    rows = []
    graph = build_topology()
    for training_seed, checkpoint, policy in learned:
        for pair_index, (source, destination) in enumerate(PAIRS):
            for condition_index, (season, hour) in enumerate(CONDITIONS):
                for replicate in range(REPLICATES):
                    scenario_id = f"{pair_index}:{condition_index}:{replicate}"
                    seed = 700000 + 100 * pair_index + 10 * condition_index + replicate
                    episode_id = f"{scenario_id}|LSTM-PPO|seed={training_seed}"
                    row = journal_rows.get(episode_id)
                    if row is None:
                        status("heldout_matrix", episode_id)
                        env = make_env(graph, source, destination, season, hour)
                        result = _run_episode(policy, env, seed)
                        row = {"method": "LSTM-PPO", "architecture": "LSTM",
                            "training_seed": training_seed, "scenario_id": scenario_id,
                            "source": source, "destination": destination, "season": season,
                            "hour": hour, "replicate": replicate, "seed": seed, **result}
                        with journal_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"kind": "episode", "id": episode_id,
                                                "row": row}, separators=(",", ":")) + "\n")
                            f.flush()
                        journal_rows[episode_id] = row
                        completed += 1
                        if completed % 10 == 0 or completed == SCENARIO_COUNT * len(SEEDS):
                            status("heldout_matrix", episode_id)
                    rows.append(row)

    lstm_groups = defaultdict(list)
    for row in rows:
        lstm_groups[int(row["training_seed"])].append(row)
    gnn_all = [r for seed in SEEDS for r in gnn_groups[seed]]
    lstm_all = rows
    by_checkpoint = {str(seed): summarize(lstm_groups[seed]) for seed in SEEDS}
    gnn_summary = summarize(gnn_all)
    lstm_summary = summarize(lstm_all)

    # Paired training seeds and identical held-out scenario IDs preserve the comparison.
    scenario_differences = []
    paired_by_seed = {}
    discordant_outcomes = {"GNN_only_success": 0, "LSTM_only_success": 0,
                           "both_success": 0, "both_fail": 0}
    paired_hop_differences = []
    for training_seed in SEEDS:
        gnn = {r["scenario_id"]: r for r in gnn_groups[training_seed]}
        lstm = {r["scenario_id"]: r for r in lstm_groups[training_seed]}
        pairs = []
        for scenario_id in sorted(gnn):
            if scenario_id not in lstm or gnn[scenario_id]["seed"] != lstm[scenario_id]["seed"]:
                raise RuntimeError("GNN/LSTM comparisons are not paired on the frozen scenario seeds")
            pairs.append({"scenario_id": scenario_id,
                "gnn_success": bool(gnn[scenario_id]["success"]),
                "lstm_success": bool(lstm[scenario_id]["success"]),
                "success_difference_lstm_minus_gnn": int(lstm[scenario_id]["success"])
                    - int(gnn[scenario_id]["success"]),
                "gnn_hops": gnn[scenario_id]["hops"] if gnn[scenario_id]["success"] else None,
                "lstm_hops": lstm[scenario_id]["hops"] if lstm[scenario_id]["success"] else None})
            gnn_ok, lstm_ok = bool(gnn[scenario_id]["success"]), bool(lstm[scenario_id]["success"])
            if gnn_ok and lstm_ok:
                discordant_outcomes["both_success"] += 1
                paired_hop_differences.append(int(lstm[scenario_id]["hops"])
                                              - int(gnn[scenario_id]["hops"]))
            elif gnn_ok:
                discordant_outcomes["GNN_only_success"] += 1
            elif lstm_ok:
                discordant_outcomes["LSTM_only_success"] += 1
            else:
                discordant_outcomes["both_fail"] += 1
        paired_by_seed[training_seed] = pairs
        scenario_differences.extend(pairs)

    latency = {key: value for key, value in gnn_report.get("inference_latency", {}).items()
               if key.startswith("GNN|")}
    env = make_env(graph, "Delhi", "Mumbai", "normal", 2.0)
    obs, _ = env.reset(seed=700000)
    for training_seed, checkpoint, policy in learned:
        key = f"LSTM|{training_seed}"
        journal_key = f"latency|{key}"
        if journal_key in journal_rows:
            latency[key] = journal_rows[journal_key]
            continue
        status("inference_latency", journal_key)
        if hasattr(policy, "enable_encoder_cache"):
            policy.enable_encoder_cache(False)
        with torch.no_grad():
            for _ in range(100):
                policy.act(obs, deterministic=True)
            times = []
            for _ in range(1000):
                start = time.perf_counter_ns()
                policy.act(obs, deterministic=True)
                times.append((time.perf_counter_ns()-start)/1e6)
        latency[key] = {"host": platform.platform(), "device": "CPU",
            "torch_threads": torch.get_num_threads(), "warmup_calls": 100,
            "measured_calls": 1000, "encoder_cache": "disabled",
            "mean_ms": float(np.mean(times)), "p50_ms": float(np.quantile(times, .5)),
            "p95_ms": float(np.quantile(times, .95)), "p99_ms": float(np.quantile(times, .99))}
        journal_rows[journal_key] = latency[key]
        with journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "latency", "id": journal_key,
                                "latency": latency[key]}, separators=(",", ":")) + "\n")

    protocol_path = ROOT / "paper/supplementary/final_gnn_lstm_protocol_20261007.md"
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in (
        "scripts/run_lstm_heldout_evaluation.py", "scripts/run_final_model_campaign.py",
        "scripts/plot_gnn_lstm_comparison.py", "scripts/run_gnn_lstm_reward_audit.py",
        "models.py", "qkd_env.py", "ppo.py", "configs/final_holdout_50ep.yaml")}
    report = {"status": "complete", "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "source_sha256": source_hashes,
        "comparison": "five independently trained GNN-PPO checkpoints vs five matched LSTM-PPO checkpoints",
        "scenario_count_per_checkpoint": SCENARIO_COUNT,
        "matched_scenario_protocol": {"endpoint_pairs": [list(x) for x in PAIRS],
            "conditions": [{"season": s, "hour": h} for s, h in CONDITIONS],
            "replicates_per_pair_condition": REPLICATES,
            "scenario_seed_formula": "700000 + 100*pair_index + 10*condition_index + replicate"},
        "checkpoint_manifest": manifest,
        "summary": {"GNN-PPO": gnn_summary, "LSTM-PPO": lstm_summary},
        "nested_training_scenario_ci95": {"GNN-PPO": nested_bootstrap(gnn_groups),
            "LSTM-PPO": nested_bootstrap(lstm_groups, seed=810031)},
        "paired_success_difference": paired_nested_difference(gnn_groups, lstm_groups),
        "paired_outcome_counts": discordant_outcomes,
        "paired_hops_on_joint_success": {"episodes": len(paired_hop_differences),
            "mean_LSTM_minus_GNN": float(np.mean(paired_hop_differences))
                if paired_hop_differences else None,
            "median_LSTM_minus_GNN": float(np.median(paired_hop_differences))
                if paired_hop_differences else None},
        "per_training_seed": {"GNN-PPO": {str(s): summarize(gnn_groups[s]) for s in SEEDS},
            "LSTM-PPO": by_checkpoint},
        "paired_scenario_outcomes_by_training_seed": {str(s): paired_by_seed[s] for s in SEEDS},
        "inference_latency": latency,
        "episodes": gnn_all + lstm_all}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    journal_rows["report|complete"] = {"status": "complete"}
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "report_complete", "id": "report|complete"}) + "\n")
    status("complete", "report|complete")
    print(json.dumps({"output": str(output), "summary": report["summary"],
                      "paired_success_difference": report["paired_success_difference"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
