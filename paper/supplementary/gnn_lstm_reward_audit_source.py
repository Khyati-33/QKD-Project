"""Compare shaped-reward components for one matched GNN-PPO/LSTM-PPO seed."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GNNActorCritic, LSTMActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology
from scripts.run_strict_holdout_evaluation import PAIRS, CONDITIONS, REPLICATES

SEED = 20261002
TERMS = ("progress", "margin", "skr", "pool_level", "pool_delta", "latency",
         "energy", "congestion", "step_cost", "revisit", "switch", "backtrack",
         "arrival", "security_penalty", "depletion_penalty", "failure_penalty")
LABELS = {"progress":"Progress", "margin":"QBER margin", "skr":"Rate proxy",
          "pool_level":"Key-pool level", "pool_delta":"Key-pool change",
          "latency":"Latency", "energy":"Energy", "congestion":"Congestion",
          "step_cost":"Step cost", "revisit":"Revisit", "switch":"Link switch",
          "backtrack":"Backtrack", "arrival":"Arrival", "security_penalty":"Security",
          "depletion_penalty":"Depletion", "failure_penalty":"No valid action"}


def sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(checkpoint: Path, architecture: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(payload.get("metadata", {}).get("hidden_dim", 64))
    model = (GNNActorCritic(hidden_dim=hidden, dropedge_probability=0.05)
             if architecture == "GNN" else LSTMActorCritic(hidden_dim=hidden))
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    if architecture == "LSTM":
        model.enable_encoder_cache(True)
    return model


def run_episode(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    totals = defaultdict(float)
    reward_total = 0.0
    success = False
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            action = policy.act(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        steps += 1
        terms = info.get("reward_terms", {})
        for name, value in terms.items():
            totals[name] += float(value)
        if not terms and reward:
            totals["failure_penalty"] += float(reward)
        success = bool(info.get("success", False))
    return {"success": success, "steps": steps, "reward": reward_total,
            **{term: float(totals[term]) for term in TERMS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-report", type=Path, default=Path("experiments/final_holdout_20261007.json"))
    parser.add_argument("--lstm-campaign", type=Path, default=Path("experiments/final_holdout_lstm_20261007"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/final_gnn_lstm_reward_audit_20261007"))
    args = parser.parse_args()
    gnn_report = args.gnn_report if args.gnn_report.is_absolute() else ROOT / args.gnn_report
    campaign = args.lstm_campaign if args.lstm_campaign.is_absolute() else ROOT / args.lstm_campaign
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    gnn_manifest = json.loads(gnn_report.read_text(encoding="utf-8"))["trained_checkpoint_manifest"]
    gnn_record = next(r for r in gnn_manifest if r["architecture"] == "GNN" and
                      int(r["training_seed"]) == SEED)
    gnn_checkpoint = ROOT / gnn_record["checkpoint"]
    lstm_runs = json.loads((campaign / "seed_runs.json").read_text(encoding="utf-8"))["runs"]
    lstm_record = next(r for r in lstm_runs if int(r["training_seed"]) == SEED)
    if lstm_record["status"] != "complete":
        raise RuntimeError("The selected LSTM seed is not complete")
    lstm_dir = ROOT / lstm_record["run_dir"]
    lstm_checkpoint = lstm_dir / "checkpoints" / "LSTM" / "best_by_success_rate.pt"
    if not lstm_checkpoint.exists():
        lstm_checkpoint = lstm_dir / "checkpoints" / "LSTM" / "latest.pt"

    policies = {"GNN-PPO": load_policy(gnn_checkpoint, "GNN"),
                "LSTM-PPO": load_policy(lstm_checkpoint, "LSTM")}
    journal = output / "episodes.jsonl"
    saved = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            saved[row["id"]] = row["episode"]

    graph = build_topology()
    rows = []
    total = len(PAIRS) * len(CONDITIONS) * REPLICATES * len(policies)
    for pair_index, (source, destination) in enumerate(PAIRS):
        for condition_index, (season, hour) in enumerate(CONDITIONS):
            for replicate in range(REPLICATES):
                scenario_id = f"{pair_index}:{condition_index}:{replicate}"
                seed = 700000 + 100 * pair_index + 10 * condition_index + replicate
                for method, policy in policies.items():
                    key = f"{scenario_id}|{method}"
                    row = saved.get(key)
                    if row is None:
                        env = QKDRoutingEnv(graph=graph, source=source, destination=destination,
                            season=season, use_case="defence", node_feature_mode="relative",
                            time_of_day_hours=hour, time_jitter_hours=0.0,
                            fiber_outage_prob=0.01, qber_hard=0.11, max_steps=144)
                        result = run_episode(policy, env, seed)
                        row = {"method": method, "training_seed": SEED,
                            "scenario_id": scenario_id, "source": source,
                            "destination": destination, "season": season,
                            "hour": hour, "replicate": replicate, "seed": seed, **result}
                        with journal.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"id": key, "episode": row}, separators=(",", ":")) + "\n")
                            f.flush()
                        saved[key] = row
                        if len(saved) % 10 == 0 or len(saved) == total:
                            print(f"[{len(saved)}/{total}] {100*len(saved)/total:.1f}% "
                                  f"reward audit unit={key}", flush=True)
                    rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summaries = {method: {"episodes": len(values),
        "success_rate": float(np.mean([r["success"] for r in values])),
        "successful_mean_hops": float(np.mean([r["steps"] for r in values if r["success"]]))
            if any(r["success"] for r in values) else None,
        **{f"mean_{term}": float(np.mean([r[term] for r in values])) for term in TERMS}}
        for method, values in groups.items()}
    deltas = {term: summaries["GNN-PPO"][f"mean_{term}"] - summaries["LSTM-PPO"][f"mean_{term}"]
              for term in TERMS}
    fig, ax = plt.subplots(figsize=(3.48, 4.55))
    vals = [deltas[term] for term in TERMS]
    y = np.arange(len(TERMS))
    ax.barh(y, vals,
            color=["#2878B5" if value >= 0 else "#D9534F" for value in vals],
            edgecolor="black", linewidth=.4)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y, [LABELS[t] for t in TERMS], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cumulative difference per episode", fontsize=7)
    ax.set_title("GNN-PPO minus LSTM-PPO reward", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linestyle="--", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "gnn_lstm_reward_components.pdf", bbox_inches="tight")
    fig.savefig(output / "gnn_lstm_reward_components.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (output / "episode_reward_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol_path = ROOT / "paper/supplementary/final_gnn_lstm_protocol_20261007.md"
    report = {"status": "complete", "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in (
            "scripts/run_gnn_lstm_reward_audit.py", "models.py", "qkd_env.py",
            "ppo.py", "configs/final_holdout_50ep.yaml")},
        "training_seed": SEED,
        "checkpoints": {"GNN-PPO": str(gnn_checkpoint.relative_to(ROOT)),
            "LSTM-PPO": str(lstm_checkpoint.relative_to(ROOT))},
        "scenario_count_per_policy": len(PAIRS) * len(CONDITIONS) * REPLICATES,
        "paired_methods": ["GNN-PPO", "LSTM-PPO"], "summary": summaries,
        "mean_component_difference_gnn_minus_lstm": deltas, "episodes": rows,
        "interpretation": "Descriptive cumulative reward components on paired scenarios; route lengths can differ and these terms are not a resource-service metric."}
    (output / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "summary": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
