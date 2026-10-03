"""Paired reward-component audit across the manuscript scenario matrix.

Runs the current GNN-PPO checkpoint and every implemented routing baseline on
the same endpoint/condition/seed combinations, records every environment
reward term, and writes four publication-ready diagnostic figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import BASELINE_NAMES, BaselinePolicy
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv


ENDPOINT_PAIRS = [
    ("Delhi", "Mumbai"), ("Mumbai", "Chennai"), ("Kolkata", "Delhi"),
    ("Bangalore", "Hyderabad"), ("Jaipur", "Kolkata"),
    ("Hyderabad", "Mumbai"),
]
# Summer/winter have no hour in the supplied protocol; hold them at the
# checkpoint's configured hour (22:00). Named day/night conditions are exact.
CONDITIONS = {
    "normal_0200": ("normal", 2.0),
    "normal_2200": ("normal", 22.0),
    "summer": ("summer", 22.0),
    "winter": ("winter", 22.0),
    "monsoon_0200": ("monsoon", 2.0),
    "monsoon_2200": ("monsoon", 22.0),
}
POLICY_ORDER = ["GNN-PPO", *BASELINE_NAMES]
TERMS = ["progress", "margin", "skr", "pool_level", "pool_delta",
         "latency", "energy", "congestion", "step_cost", "revisit",
         "switch", "backtrack", "arrival", "security_penalty",
         "depletion_penalty", "failure_penalty"]
TERM_LABELS = {
    "progress": "Progress", "margin": "QBER margin", "skr": "SKR",
    "pool_level": "Key pool level", "pool_delta": "Key pool change",
    "latency": "Latency", "energy": "Energy", "congestion": "Congestion",
    "step_cost": "Step cost", "revisit": "Revisit", "switch": "Link switch",
    "backtrack": "Backtrack", "arrival": "Arrival",
    "security_penalty": "Security violation", "depletion_penalty": "Pool depletion",
    "failure_penalty": "No valid neighbor",
}
UTILITY_TERMS = ["progress", "margin", "skr", "pool_level", "pool_delta",
                 "latency", "energy", "congestion", "arrival"]
PROTECTION_TERMS = ["step_cost", "revisit", "switch", "backtrack",
                    "security_penalty", "depletion_penalty", "failure_penalty"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_episode(policy, env: QKDRoutingEnv, seed: int, policy_name: str,
                source: str, destination: str, condition: str) -> dict:
    obs, _ = env.reset(seed=seed)
    totals: defaultdict[str, float] = defaultdict(float)
    reward_sum = 0.0
    steps = 0
    success = False
    failure = ""
    terminated = truncated = False
    while not (terminated or truncated):
        if isinstance(policy, BaselinePolicy):
            action = policy.act(obs, env)
        else:
            action = policy.act(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        steps += 1
        terms = info.get("reward_terms", {})
        for term, value in terms.items():
            totals[term] += float(value)
        # The environment uses this terminal penalty for a dead-end state but
        # currently reports an empty reward_terms dictionary on that branch.
        if not terms and reward:
            totals["failure_penalty"] += float(reward)
        success = bool(info.get("success", False))
        failure = str(info.get("failure", ""))
    if not success and not failure:
        if totals["security_penalty"] < 0:
            failure = "security_breach"
        elif totals["depletion_penalty"] < 0:
            failure = "pool_depleted"
        elif totals["failure_penalty"] < 0:
            failure = "no_valid_neighbor"
        elif truncated:
            failure = "max_steps"
        else:
            failure = "terminated_without_arrival"
    row = {
        "policy": policy_name, "condition": condition, "season": env.season,
        "time_of_day_hours": env.start_time_of_day_hours, "seed": seed,
        "source": source, "destination": destination, "success": success,
        "hops": steps if success else "", "steps": steps, "failure": failure,
        "total_reward": reward_sum,
    }
    row.update({term: totals[term] for term in TERMS})
    return row


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key] or 0.0) for row in rows])) if rows else float("nan")


def save_plot(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_component_plot(rows, terms, path, title):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(terms))
    width = 0.82 / len(POLICY_ORDER)
    colors = plt.get_cmap("tab10").colors
    for i, policy in enumerate(POLICY_ORDER):
        subset = [r for r in rows if r["policy"] == policy]
        vals = [mean(subset, t) for t in terms]
        ax.bar(x + (i - (len(POLICY_ORDER) - 1) / 2) * width, vals,
               width, label=policy, color=colors[i], edgecolor="black", linewidth=0.45)
    ax.axhline(0, color="black", linewidth=0.85)
    ax.set_xticks(x, [TERM_LABELS[t] for t in terms], rotation=28, ha="right")
    ax.set_ylabel("Mean cumulative reward contribution per episode")
    ax.set_xlabel("Reward component")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    save_plot(fig, path)


def heatmap(values, row_labels, col_labels, path, title, colorbar_label,
            *, percent=False, blank_nan=False):
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    masked = np.ma.masked_invalid(data) if blank_nan else data
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0,
                   vmax=100 if percent else None)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(colorbar_label)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            label = "—" if np.isnan(val) else (f"{val:.0f}%" if percent else f"{val:.1f}")
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="black" if np.isnan(val) or val < (60 if percent else 0.55 * np.nanmax(data)) else "white")
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel("Routing policy")
    fig.tight_layout()
    save_plot(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" /
                        "defence_monsoon_night_50ep_idq.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments" /
                        "runs" / "defence_monsoon_night_50ep_idq_20260928T044715Z_c84be8c5" /
                        "checkpoints" / "GNN" / "latest.pt")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    env_cfg = config.get("environment", {})
    torch.set_num_threads(max(1, int(config.get("cpu_threads", 8))))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("metadata", {}).get("hidden_dim", config.get("hidden_dim", 64)))
    model = GNNActorCritic(hidden_dim=hidden_dim,
                           dropedge_probability=float(config.get("model", {}).get(
                               "dropedge_probability", 0.0)))
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    if hasattr(model, "enable_encoder_cache"):
        model.enable_encoder_cache(True)

    output = args.output or ROOT / "experiments" / "runs" / (
        "reward_component_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    plots_dir = output / "plots"
    plots_dir.mkdir()

    base_kwargs = {
        "use_case": env_cfg.get("use_case", "defence"),
        "node_feature_mode": env_cfg.get("node_feature_mode", "relative"),
        "dt_seconds": float(env_cfg.get("dt_seconds", 300.0)),
        "time_jitter_hours": 0.0,
        "fiber_outage_prob": float(env_cfg.get("fiber_outage_prob", 0.01)),
        "qber_hard": float(env_cfg.get("qber_hard", 0.11)),
        "max_steps": int(env_cfg.get("max_steps", 400)),
        "disabled_reward_terms": set(env_cfg.get("disabled_reward_terms", [])),
        "reward_protection_overrides": dict(env_cfg.get("reward_protection_overrides", {})),
    }
    policies = {"GNN-PPO": model}
    all_rows = []
    scenarios = []
    for pair_index, (source, destination) in enumerate(ENDPOINT_PAIRS):
        for condition, (season, hour) in CONDITIONS.items():
            for seed_offset in range(args.seed_count):
                seed = args.seed_base + seed_offset
                scenarios.append({"source": source, "destination": destination,
                                  "condition": condition, "season": season,
                                  "time_of_day_hours": hour, "seed": seed})
                for policy_name in POLICY_ORDER:
                    if policy_name == "GNN-PPO":
                        policy = model
                    else:
                        policy = BaselinePolicy(policy_name, seed=seed + pair_index * 1000)
                    env = QKDRoutingEnv(source=source, destination=destination,
                                        season=season, time_of_day_hours=hour,
                                        **base_kwargs)
                    all_rows.append(run_episode(policy, env, seed, policy_name,
                                                source, destination, condition))
            print(f"Completed {source}–{destination}: {condition}", flush=True)

    fieldnames = ["policy", "condition", "season", "time_of_day_hours", "seed",
                  "source", "destination", "success", "hops", "steps", "failure",
                  "total_reward", *TERMS]
    with (output / "episode_reward_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for policy_name in POLICY_ORDER:
        subset = [r for r in all_rows if r["policy"] == policy_name]
        successes = [r for r in subset if r["success"]]
        row = {"policy": policy_name, "episodes": len(subset),
               "success_rate": len(successes) / len(subset),
               "mean_hops_successful": mean(successes, "hops") if successes else "",
               "mean_total_reward": mean(subset, "total_reward")}
        row.update({f"mean_{t}": mean(subset, t) for t in TERMS})
        summary_rows.append(row)
    with (output / "policy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Graph 1: objective/utility terms, including destination-arrival reward.
    grouped_component_plot(all_rows, UTILITY_TERMS, plots_dir / "01_reward_utility_components",
                           "Mean cumulative utility contributions by policy")
    # Graph 2: operational and safety costs, shown on a separate scale.
    grouped_component_plot(all_rows, PROTECTION_TERMS, plots_dir / "02_reward_protection_costs",
                           "Mean cumulative protection and failure costs by policy")

    # Graph 3: component-level advantage of the learned agent over hop-optimal BFS.
    gnn_rows = [r for r in all_rows if r["policy"] == "GNN-PPO"]
    bfs_rows = [r for r in all_rows if r["policy"] == "BFS-hop"]
    labels = ["Progress", "QBER margin", "SKR", "Key pool", "Latency",
              "Energy", "Congestion", "Step cost", "Arrival"]
    term_sets = [["progress"], ["margin"], ["skr"], ["pool_level", "pool_delta"],
                 ["latency"], ["energy"], ["congestion"], ["step_cost"], ["arrival"]]
    delta = [sum(mean(gnn_rows, t) - mean(bfs_rows, t) for t in group)
             for group in term_sets]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = ["#2878B5" if v >= 0 else "#D9534F" for v in delta]
    ax.bar(np.arange(len(delta)), delta, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(delta)), labels, rotation=25, ha="right")
    ax.set_ylabel("Mean cumulative reward difference per episode")
    ax.set_xlabel("Component (GNN-PPO minus BFS-hop)")
    ax.set_title("Paired reward-component difference: GNN-PPO − BFS-hop")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    fig.tight_layout()
    save_plot(fig, plots_dir / "03_gnn_minus_bfs_components")

    # Graph 4 is a two-panel outcome figure: completion rates and hops on
    # successful episodes, each broken down by the same six conditions.
    success_values, hop_values = [], []
    for policy_name in POLICY_ORDER:
        success_row, hop_row = [], []
        for condition in CONDITIONS:
            group = [r for r in all_rows if r["policy"] == policy_name and r["condition"] == condition]
            ok = [r for r in group if r["success"]]
            success_row.append(100.0 * len(ok) / len(group))
            hop_row.append(mean(ok, "hops") if ok else float("nan"))
        success_values.append(success_row)
        hop_values.append(hop_row)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, vals, title, cbar, percent, blank in [
        (axes[0], success_values, "Completion rate", "Successful episodes (%)", True, False),
        (axes[1], hop_values, "Successful-route hop count", "Mean hops, successes only", False, True),
    ]:
        data = np.asarray(vals, dtype=float)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(np.ma.masked_invalid(data) if blank else data, cmap=cmap,
                       aspect="auto", vmin=0, vmax=100 if percent else None)
        ax.set_xticks(np.arange(len(CONDITIONS)), list(CONDITIONS), rotation=28, ha="right")
        ax.set_yticks(np.arange(len(POLICY_ORDER)), POLICY_ORDER)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, pad=0.025)
        cb.set_label(cbar)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                label = "—" if np.isnan(val) else (f"{val:.0f}%" if percent else f"{val:.1f}")
                threshold = 60 if percent else (0.55 * np.nanmax(data) if np.isfinite(data).any() else 0)
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="black" if np.isnan(val) or val < threshold else "white")
    axes[0].set_ylabel("Routing policy")
    axes[1].set_ylabel("Routing policy")
    fig.suptitle("Paired evaluation outcomes across conditions", fontsize=14)
    save_plot(fig, plots_dir / "04_policy_outcomes")

    report = {
        "status": "paired_reward_component_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(ROOT)),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "scenario_count_per_policy": len(scenarios),
        "episodes_per_policy": len(scenarios),
        "policies": POLICY_ORDER,
        "implemented_baselines_only": True,
        "weighted_shortest_path_note": "Not included; no Weighted-SP implementation exists in this repository.",
        "scenarios": {"endpoint_pairs": ENDPOINT_PAIRS, "conditions": CONDITIONS,
                      "seeds": list(range(args.seed_base, args.seed_base + args.seed_count))},
        "condition_hour_note": "summer and winter are evaluated at 22:00 because the supplied condition labels specify no hour; time jitter is disabled so condition hours remain matched.",
        "interpretation_note": "Environment seeds and starting conditions are paired; later channel states can diverge because each policy visits a different route. Results are simulated, route-level measurements.",
        "scope_note": "No network-wide demand blocking, key-pool load balancing, deployed service tests, or field validation are measured.",
        "summary": summary_rows,
    }
    (output / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Paired reward-component audit\n\n"
        f"Evaluated {len(scenarios)} matched endpoint/condition/seed scenarios per policy "
        f"({len(POLICY_ORDER)} policies; {len(all_rows)} episodes total). Every cumulative "
        "reward term is stored in `episode_reward_components.csv`; `policy_summary.csv` "
        "contains per-policy means. Four figures are saved in `plots/` as both PNG and PDF.\n\n"
        "The reward decomposition uses the environment's existing `info['reward_terms']`; "
        "reward equations were not changed. Conditions without a specified hour use the "
        "checkpoint's 22:00 operating point. This is a simulator-only route audit.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "episode_count": len(all_rows),
                      "summary": summary_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
