"""Run independent 50-epoch GNN policies on a shared held-out evaluation set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (20261002, 20261003, 20261004, 20261005, 20261006)
BASELINES = ("Random", "Dijkstra-km", "BFS-hop", "Max-SKR")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # Two-sided 95% Student-t critical value for 4 degrees of freedom.
    t_critical = 2.7764451051977987 if len(values) == 5 else None
    margin = t_critical * sd / (len(values) ** 0.5) if t_critical else None
    return {"mean_across_training_seeds": mean,
            "sample_sd_across_training_seeds": sd,
            "min": min(values), "max": max(values),
            "training_seed_95pct_t_interval": ([mean - margin, mean + margin]
                                                if margin is not None else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" /
                        "seed_stability_50ep.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    raw_base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    campaign_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.output_dir or ROOT / "experiments" / "runs" / \
        f"training_seed_stability_{campaign_stamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "configs").mkdir()

    # Import only after CLI/config parsing; each call resets Python, NumPy and
    # Torch RNGs from its own training seed.
    from run_experiment import run_experiment

    run_records = []
    per_seed_rows = []
    for seed in args.seeds:
        config = dict(raw_base)
        config["experiment_name"] = f"qkd_gnn_seed_stability_{seed}"
        config["seed"] = int(seed)
        config_path = campaign / "configs" / f"seed_{seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(config_path)
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        final = max(evaluation, key=lambda row: int(row["ppo_epoch"]))
        if int(final["ppo_epoch"]) != int(config["training"]["ppo_epochs"]):
            raise RuntimeError(f"seed {seed} did not complete the configured PPO budget")
        run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        record = {"training_seed": int(seed), "run_dir": str(run_dir.relative_to(ROOT)),
                  "run_status": run_summary["status"],
                  "git_commit": run_summary["git_commit"],
                  "final_ppo_epoch": int(final["ppo_epoch"]),
                  "policy": final["policy"], "baselines": final["baselines"]}
        run_records.append(record)
        for method, metrics in [("GNN-PPO", record["policy"]),
                                *[(name, record["baselines"][name]) for name in BASELINES]]:
            per_seed_rows.append({"training_seed": int(seed), "method": method,
                "success_rate": metrics["success_rate"],
                "avg_hops_success": metrics["avg_hops_success"],
                "overall_reward": metrics["overall_reward"],
                "qber_skr_valid_rate": metrics["qber_skr_validity"]["valid_rate"],
                "invalid_actions": metrics["qber_skr_validity"].get("invalid_action_count", 0),
                "fallback_count": metrics["qber_skr_validity"].get("fallback_count", 0)})

    aggregate = {}
    for method in ("GNN-PPO", *BASELINES):
        rows = [r for r in per_seed_rows if r["method"] == method]
        aggregate[method] = {}
        for metric in ("success_rate", "avg_hops_success", "overall_reward",
                       "qber_skr_valid_rate"):
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            aggregate[method][metric] = mean_summary(values) if values else None

    with (campaign / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0]))
        writer.writeheader(); writer.writerows(per_seed_rows)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256(args.config),
        "independent_training_seeds": [int(seed) for seed in args.seeds],
        "training_seed_count": len(args.seeds),
        "fixed_training_budget": {"bc_epochs": int(raw_base["training"]["bc_epochs"]),
            "bc_episodes_per_epoch": int(raw_base["training"]["bc_episodes_per_epoch"]),
            "ppo_epochs": int(raw_base["training"]["ppo_epochs"]),
            "steps_per_epoch": int(raw_base["training"]["steps_per_epoch"])},
        "shared_evaluation_protocol": {"seasons": raw_base["evaluation"]["seasons"],
            "episodes_per_season": int(raw_base["evaluation"]["eval_seeds_per_season"]),
            "endpoint_pair": raw_base["protected_conditions"]["source_destination"]["value"],
            "baseline_environment_seeds_paired": True},
        "campaign_dir": str(campaign.relative_to(ROOT)),
        "per_seed": run_records,
        "aggregate_across_independent_training_seeds": aggregate,
        "interpretation": "Training-seed intervals quantify variability across five fitted policies on the fixed simulator evaluation protocol. They do not imply external validity, QKD security, or network-service performance.",
    }
    (campaign / "campaign_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"campaign_dir": str(campaign),
                      "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
