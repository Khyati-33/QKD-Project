"""Paired-seed comparison of the current GNN checkpoint and route baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import BASELINE_NAMES, BaselinePolicy
from evaluation import _run_episode
from models import GNNActorCritic
from physics import SEASONS
from qkd_env import QKDRoutingEnv
from ppo import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0:
        return [None, None]
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half = z * ((p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) ** 0.5) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" /
                        "defence_monsoon_night_50ep_idq.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments" /
                        "runs" / "defence_monsoon_night_50ep_idq_20260928T044715Z_c84be8c5" /
                        "checkpoints" / "GNN" / "latest.pt")
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--seeds-per-season", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment = config.get("environment", {})
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hidden_dim = int(checkpoint_payload.get("metadata", {}).get("hidden_dim", 64))
    model = GNNActorCritic(hidden_dim=hidden_dim,
                           dropedge_probability=float(config.get("model", {}).get(
                               "dropedge_probability", 0.0)))
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    model.enable_encoder_cache(True)

    env_kwargs = {
        "source": environment.get("source", "Delhi"),
        "destination": environment.get("destination", "Chennai"),
        "use_case": environment.get("use_case", "standard"),
        "node_feature_mode": environment.get("node_feature_mode", "geographic"),
        "time_of_day_hours": float(environment.get("time_of_day_hours", 12.0)),
        "time_jitter_hours": float(environment.get("time_jitter_hours", 0.0)),
        "fiber_outage_prob": float(environment.get("fiber_outage_prob", 0.01)),
        "qber_hard": float(environment.get("qber_hard", 0.11)),
        "max_steps": int(environment.get("max_steps", 400)),
    }
    policies = {"GNN-PPO": model}
    policies.update({name: BaselinePolicy(name, seed=args.seed_base)
                     for name in BASELINE_NAMES})
    rows = []
    for season_index, season in enumerate(SEASONS):
        for episode_index in range(args.seeds_per_season):
            seed = args.seed_base + season_index * 10_000 + episode_index
            # Every policy gets a fresh environment with the same inputs and
            # seed; this pairs the sampled outage/channel realization.
            for policy_name, policy in policies.items():
                if isinstance(policy, BaselinePolicy):
                    policy.rng = random.Random(args.seed_base + episode_index)
                env = QKDRoutingEnv(season=season, **env_kwargs)
                result = _run_episode(policy, env, seed)
                rows.append({"policy": policy_name, "season": season,
                             "seed": seed, "source": result["source"],
                             "destination": result["destination"],
                             "success": result["success"], "hops": result["hops"],
                             "reward": result["reward"],
                             "revisits": result["revisits"],
                             "checked_edges": result["qber_skr_validity"]["checked_edges"],
                             "valid_edges": result["qber_skr_validity"]["valid_edges"],
                             "invalid_actions": result["invalid_action_count"],
                             "no_valid_action_states": result["no_valid_action_states"],
                             "fallbacks": result["fallback_count"],
                             "fiber_edges": result["link_type_counts"]["fiber"],
                             "fso_edges": result["link_type_counts"]["fso"],
                             "route": json.dumps(result["route"], ensure_ascii=False)})

    summaries = {}
    for name in policies:
        group = [row for row in rows if row["policy"] == name]
        successful = [row for row in group if row["success"]]
        summaries[name] = {
            "episodes": len(group),
            "successes": len(successful),
            "success_rate": len(successful) / len(group),
            "success_rate_wilson_95pct": _wilson_interval(len(successful), len(group)),
            "successful_mean_hops": (sum(row["hops"] for row in successful) /
                                      len(successful) if successful else None),
            "successful_median_hops": (statistics.median(row["hops"] for row in successful)
                                       if successful else None),
            "checked_edges": sum(row["checked_edges"] for row in group),
            "invalid_actions": sum(row["invalid_actions"] for row in group),
            "no_valid_action_states": sum(row["no_valid_action_states"] for row in group),
            "fallbacks": sum(row["fallbacks"] for row in group),
            "revisits": sum(row["revisits"] for row in group),
            "fso_edges": sum(row["fso_edges"] for row in group),
        }

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "experiments" / "runs" / f"corrected_route_comparison_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "status": "paired_route_level_simulation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_status": subprocess.run(["git", "status", "--short"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.splitlines(),
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": _sha256(args.config),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "conditions": {"source": env_kwargs["source"],
            "destination": env_kwargs["destination"], "seasons": list(SEASONS),
            "seeds_per_season": args.seeds_per_season,
            "seed_base": args.seed_base, "shared_environment_seeds": True,
            "shared_valid_action_mask": True, "max_steps": env_kwargs["max_steps"]},
        "scope_note": "Single-route simulator comparison; not demand blocking, QKP load balancing, or service throughput.",
        "summary": summaries,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Corrected paired-seed route comparison\n\n"
        "Each method receives the same season and environment seed for every episode. "
        "The evaluator records invalid action proposals and environment fallbacks. "
        "This remains a route-level comparison and does not measure dynamic-demand "
        "blocking or network-wide key-pool utilization.\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
