"""Validate and archive a completed five-seed campaign into paper evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "paper" / "supplementary"
EXPECTED_SEEDS = [20261002, 20261003, 20261004, 20261005, 20261006]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    report_path = campaign / "campaign_summary.json"
    metrics_path = campaign / "per_seed_metrics.csv"
    if not report_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Campaign summary or per-seed CSV is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("independent_training_seeds") != EXPECTED_SEEDS:
        raise ValueError("Campaign is not the complete, expected five-seed protocol")
    expected_epochs = int(report["fixed_training_budget"]["ppo_epochs"])
    if expected_epochs != 50 or len(report.get("per_seed", [])) != 5:
        raise ValueError("Campaign does not contain five policies at the fixed 50-epoch budget")
    artifacts = []
    for record in report["per_seed"]:
        seed = int(record["training_seed"])
        run_dir = ROOT / record["run_dir"]
        run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        if run_summary.get("status") != "complete" or int(record["final_ppo_epoch"]) != expected_epochs:
            raise ValueError(f"Seed {seed} did not finish the full training budget")
        if not evaluation or max(int(row["ppo_epoch"]) for row in evaluation) != expected_epochs:
            raise ValueError(f"Seed {seed} is missing final evaluation evidence")
        checkpoint = run_dir / "checkpoints" / "GNN" / "latest.pt"
        resolved_config = run_dir / "configs" / "resolved.yaml"
        if not checkpoint.exists() or not resolved_config.exists():
            raise FileNotFoundError(f"Seed {seed} checkpoint/config snapshot is missing")
        checkpoint_target = SUPPLEMENT / f"checkpoint_seed_{seed}_50ep.pt"
        config_target = SUPPLEMENT / f"seed_{seed}_resolved.yaml"
        shutil.copy2(checkpoint, checkpoint_target)
        shutil.copy2(resolved_config, config_target)
        artifacts.extend([{"path": checkpoint_target.name, "sha256": digest(checkpoint_target),
                           "training_seed": seed},
                          {"path": config_target.name, "sha256": digest(config_target),
                           "training_seed": seed}])
    shutil.copy2(report_path, SUPPLEMENT / "training_seed_stability_20261002.json")
    shutil.copy2(metrics_path, SUPPLEMENT / "training_seed_stability_20261002.csv")
    manifest = {"campaign_summary_sha256": digest(SUPPLEMENT / "training_seed_stability_20261002.json"),
        "per_seed_metrics_sha256": digest(SUPPLEMENT / "training_seed_stability_20261002.csv"),
        "artifacts": artifacts,
        "validation": "all five runs completed 50 PPO epochs and contain a final shared-protocol evaluation"}
    (SUPPLEMENT / "training_seed_stability_artifacts_20261002.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archived_seeds": EXPECTED_SEEDS,
                      "supplementary_dir": str(SUPPLEMENT)}, indent=2))


if __name__ == "__main__":
    main()
