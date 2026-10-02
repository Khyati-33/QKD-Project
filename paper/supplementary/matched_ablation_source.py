"""Run matched-budget GNN feature, reward, regularization, and width ablations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "baseline": {},
    "geographic_features": {"node_feature_mode": "geographic"},
    "no_skr_reward": {"disabled_reward_terms": ["skr"]},
    "no_dropedge": {"dropedge_probability": 0.0},
    "hidden_dim_32": {"hidden_dim": 32},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" /
                        "seed_stability_50ep.yaml")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=list(VARIANTS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    base["cpu_threads"] = args.cpu_threads
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.output_dir or ROOT / "experiments" / "runs" / f"matched_ablations_{stamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "configs").mkdir()

    from run_experiment import run_experiment

    records = []
    for variant in args.variants:
        override = VARIANTS[variant]
        config = copy.deepcopy(base)
        config["experiment_name"] = f"matched_ablation_{variant}_{args.seed}"
        config["seed"] = args.seed
        if "hidden_dim" in override:
            config["hidden_dim"] = override["hidden_dim"]
        if "dropedge_probability" in override:
            config.setdefault("model", {})["dropedge_probability"] = override["dropedge_probability"]
        if "node_feature_mode" in override:
            config.setdefault("environment", {})["node_feature_mode"] = override["node_feature_mode"]
        if "disabled_reward_terms" in override:
            config.setdefault("environment", {})["disabled_reward_terms"] = override["disabled_reward_terms"]
        config_path = campaign / "configs" / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(config_path)
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        final = max(evaluation, key=lambda row: int(row["ppo_epoch"]))
        if int(final["ppo_epoch"]) != int(config["training"]["ppo_epochs"]):
            raise RuntimeError(f"{variant} stopped before the matched PPO budget")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records.append({"variant": variant, "run_dir": str(run_dir.relative_to(ROOT)),
            "seed": args.seed, "git_commit": summary["git_commit"],
            "config_sha256": digest(config_path), "final_ppo_epoch": final["ppo_epoch"],
            "policy": final["policy"], "baselines": final["baselines"]})
        # Persist after each run so interrupted multi-hour campaigns retain evidence.
        (campaign / "progress.json").write_text(json.dumps({"status": "in_progress",
            "completed_variants": records, "planned_variants": args.variants}, indent=2),
            encoding="utf-8")

    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(args.config.relative_to(ROOT)), "source_config_sha256": digest(args.config),
        "shared_training_seed": args.seed, "matched_training_budget": base["training"],
        "cpu_threads_per_variant": args.cpu_threads,
        "shared_evaluation_protocol": base["evaluation"], "variants": records,
        "scope_note": "Single-seed matched-budget ablations diagnose sensitivity; they are not independent-training-seed estimates.",
        "campaign_dir": str(campaign.relative_to(ROOT))}
    output = ROOT / "paper" / "supplementary" / "matched_budget_ablations_20261002.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (campaign / "campaign_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "variants": len(records)}, indent=2))


if __name__ == "__main__":
    main()
