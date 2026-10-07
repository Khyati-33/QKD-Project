"""Repeat the matched reward-term decomposition for frozen final GNN/MLP seeds."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
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
from baselines import BaselinePolicy
from models import GNNActorCritic, MLPActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology
from scripts.run_strict_holdout_evaluation import PAIRS, CONDITIONS, REPLICATES

TERMS = ("progress", "margin", "skr", "pool_level", "pool_delta", "latency",
         "energy", "congestion", "step_cost", "revisit", "switch", "backtrack",
         "arrival", "security_penalty", "depletion_penalty", "failure_penalty")
LABELS = {"progress":"Progress", "margin":"QBER margin", "skr":"Rate proxy",
          "pool_level":"Key-pool level", "pool_delta":"Key-pool change",
          "latency":"Latency", "energy":"Energy", "congestion":"Congestion",
          "step_cost":"Step cost", "revisit":"Revisit", "switch":"Link switch",
          "backtrack":"Backtrack", "arrival":"Arrival", "security_penalty":"Security",
          "depletion_penalty":"Depletion", "failure_penalty":"No valid action"}


def load_final_policy(campaign: Path, architecture: str, seed: int):
    report = json.loads((campaign / "seed_runs.json").read_text(encoding="utf-8"))
    record = next(r for r in report["runs"] if int(r["training_seed"]) == seed)
    if record["status"] != "complete":
        raise RuntimeError(f"{architecture} seed {seed} is not complete")
    run_dir = ROOT / record["run_dir"]
    checkpoint = run_dir / "checkpoints" / architecture / "best_by_success_rate.pt"
    if not checkpoint.exists(): checkpoint = run_dir / "checkpoints" / architecture / "latest.pt"
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(data.get("metadata", {}).get("hidden_dim", 64))
    model = (GNNActorCritic(hidden_dim=hidden, dropedge_probability=0.05)
             if architecture == "GNN" else MLPActorCritic(hidden_dim=hidden))
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    return model, checkpoint


def run_episode(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    totals = defaultdict(float); reward_total = 0.0; success = False; steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if isinstance(policy, BaselinePolicy):
            action = policy.act(obs, env)
        else:
            with torch.no_grad(): action = policy.act(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward); steps += 1
        terms = info.get("reward_terms", {})
        for name, value in terms.items(): totals[name] += float(value)
        if not terms and reward: totals["failure_penalty"] += float(reward)
        success = bool(info.get("success", False))
    return {"success": success, "steps": steps, "reward": reward_total,
            **{term: totals[term] for term in TERMS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-campaign", type=Path, required=True)
    parser.add_argument("--mlp-campaign", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=20261002)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"experiments"/"final_reward_audit_20261007")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_topology()
    policies = {}
    checkpoint_paths = {}
    for arch, campaign in (("GNN", args.gnn_campaign), ("MLP", args.mlp_campaign)):
        policies[f"{arch}-PPO"], checkpoint_paths[arch] = load_final_policy(campaign, arch, args.training_seed)
    journal_path = args.output_dir / "progress.jsonl"
    status_path = args.output_dir / "progress_status.jsonl"
    total_episodes = len(PAIRS) * len(CONDITIONS) * REPLICATES * 3
    protocol_path = ROOT / "paper/supplementary/final_experiment_protocol_20261003.md"
    metadata = {"kind": "metadata", "training_seed": args.training_seed,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "checkpoints": {k: str(v.relative_to(ROOT)) for k, v in checkpoint_paths.items()}}
    saved_rows = {}
    if journal_path.exists():
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        parsed = []
        for line in lines:
            try: parsed.append(json.loads(line))
            except json.JSONDecodeError: break
        if len(parsed) != len(lines):
            journal_path.write_text("\n".join(json.dumps(row) for row in parsed) +
                                    ("\n" if parsed else ""), encoding="utf-8")
        if parsed and parsed[0] != metadata:
            raise RuntimeError(f"reward-audit journal does not match this run: {journal_path}")
        saved_rows = {row["id"]: row["episode"] for row in parsed[1:]}
    else:
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    completed = len(saved_rows)

    def write_progress(stage, unit):
        nonlocal completed
        progress = {"stage": stage, "current_unit": unit,
            "completed_episodes": completed, "total_episodes": total_episodes,
            "percent": round(100 * completed / total_episodes, 2),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        with status_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(progress, separators=(",", ":")) + "\n")
            stream.flush()

    rows = []
    for pair_index, (source, destination) in enumerate(PAIRS):
        for condition_index, (season, hour) in enumerate(CONDITIONS):
            for replicate in range(REPLICATES):
                seed = 700000 + 100*pair_index + 10*condition_index + replicate
                scenario_id = f"{pair_index}:{condition_index}:{replicate}"
                methods = list(policies.items()) + [("BFS-hop", BaselinePolicy("BFS-hop", seed=seed+917))]
                for method, policy in methods:
                    episode_id = f"{scenario_id}|{method}"
                    row = saved_rows.get(episode_id)
                    if row is None:
                        write_progress("reward_component_audit", episode_id)
                        env = QKDRoutingEnv(graph=graph, source=source, destination=destination,
                            season=season, use_case="defence", node_feature_mode="relative",
                            time_of_day_hours=hour, time_jitter_hours=0.0,
                            fiber_outage_prob=0.01, qber_hard=0.11, max_steps=144)
                        result = run_episode(policy, env, seed)
                        row = {"method": method, "training_seed": args.training_seed,
                            "scenario_id": scenario_id, "source": source, "destination": destination,
                            "season": season, "hour": hour, "replicate": replicate, "seed": seed,
                            **result}
                        with journal_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"id": episode_id,
                                "episode": row}, separators=(",", ":")) + "\n")
                            stream.flush(); os.fsync(stream.fileno())
                        saved_rows[episode_id] = row
                        completed += 1
                        write_progress("reward_component_audit", episode_id)
                        if completed % 5 == 0 or completed == total_episodes:
                            print(f"[PROGRESS] reward audit {completed}/{total_episodes} "
                                  f"({100*completed/total_episodes:.1f}%) last={episode_id}",
                                  flush=True)
                    rows.append(row)
        print(f"Completed reward audit pair {source}-{destination}", flush=True)
    fields = list(rows[0])
    with (args.output_dir/"episode_reward_components.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for row in rows: groups[row["method"]].append(row)
    summaries={m:{"episodes":len(rs), "success_rate":float(np.mean([r["success"] for r in rs])),
        "mean_successful_hops":float(np.mean([r["steps"] for r in rs if r["success"]])) if any(r["success"] for r in rs) else None,
        **{f"mean_{t}":float(np.mean([r[t] for r in rs])) for t in TERMS}}
        for m,rs in groups.items()}
    deltas={arch:{term:float(np.mean([r[term] for r in groups[f"{arch}-PPO"]])-
                                  np.mean([r[term] for r in groups["BFS-hop"]]))
                  for term in TERMS} for arch in ("GNN","MLP")}
    fig, axes = plt.subplots(1,2,figsize=(10,6),sharey=True)
    x=np.arange(len(TERMS)); width=.38
    for ax, arch in zip(axes,("GNN","MLP")):
        vals=[deltas[arch][t] for t in TERMS]
        ax.bar(x,vals,width,color=["#2878B5" if v>=0 else "#D9534F" for v in vals],edgecolor="black",linewidth=.45)
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_xticks(x,[LABELS[t] for t in TERMS],rotation=55,ha="right",fontsize=9)
        ax.set_title(f"{arch}-PPO minus BFS-hop")
        ax.grid(axis="y",linestyle="--",alpha=.3)
        ax.set_xlabel("Cumulative reward term")
    axes[0].set_ylabel("Paired mean contribution difference per episode")
    fig.suptitle("Reward decomposition on the frozen 108-scenario test set")
    fig.tight_layout()
    for suffix, kw in (("pdf",{}),("png",{"dpi":400})):
        fig.savefig(args.output_dir/f"final_reward_components.{suffix}",bbox_inches="tight",**kw)
    plt.close(fig)
    report={"status":"complete","protocol":"final_experiment_protocol_20261003.md",
        "training_seed":args.training_seed,
        "checkpoint_paths":{k:str(v.relative_to(ROOT)) for k,v in checkpoint_paths.items()},
        "scenario_count_per_policy":len(PAIRS)*len(CONDITIONS)*REPLICATES,
        "paired_methods":["GNN-PPO","MLP-PPO","BFS-hop"],
        "reward_term_summary":summaries,"agent_minus_bfs_component_difference":deltas,
        "episodes":rows,
        "interpretation":"Descriptive cumulative reward components on paired scenarios; route visits can diverge and do not estimate network service capacity."}
    # ``args.output_dir`` may be relative to the caller's working directory;
    # normalize before making a repository-relative report path on Windows.
    report["progress_journal"] = str(journal_path.resolve().relative_to(ROOT))
    report_tmp = args.output_dir/"audit_summary.json.tmp"
    report_tmp.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    report_tmp.replace(args.output_dir/"audit_summary.json")
    write_progress("complete", "audit_summary_and_reward_component_plots")
    print(json.dumps({"output_dir":str(args.output_dir),"summary":summaries},indent=2),flush=True)


if __name__=="__main__": main()
