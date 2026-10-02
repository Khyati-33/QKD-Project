"""Refresh a compact status record from the live multi-seed run directories."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
OUTPUT = ROOT / "paper" / "supplementary" / "training_seed_stability_progress_20261002.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_training_seed_stability import SEEDS as PLANNED_SEEDS


def main() -> None:
    campaigns = sorted(RUNS.glob("training_seed_stability_*/configs/seed_*.yaml"),
                       key=lambda path: path.stat().st_mtime, reverse=True)
    if not campaigns:
        raise FileNotFoundError("No training seed campaign configs were found")
    config_dir = campaigns[0].parent
    campaign_dir = config_dir.parent
    configured_seeds = {int(match.group(1)) for path in config_dir.glob("seed_*.yaml")
                        if (match := re.fullmatch(r"seed_(\d+)\.yaml", path.name))}
    seeds = list(PLANNED_SEEDS)
    if not configured_seeds.issubset(set(seeds)):
        raise ValueError(f"campaign contains unplanned seeds: {configured_seeds - set(seeds)}")
    completed = []
    active = []
    for seed in seeds:
        candidates = sorted(RUNS.glob(f"qkd_gnn_seed_stability_{seed}_*"),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        run_dir = next((path for path in candidates if (path / "summary.json").exists()), None)
        if run_dir is not None:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") == "complete" and (run_dir / "evaluation.json").exists():
                evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
                final = max(evaluation, key=lambda row: int(row["ppo_epoch"]))
                completed.append({"training_seed": seed, "run_dir": str(run_dir.relative_to(ROOT)),
                    "final_ppo_epoch": int(final["ppo_epoch"]),
                    "success_rate": final["policy"]["success_rate"],
                    "avg_hops_success": final["policy"]["avg_hops_success"]})
                continue
        run_dir = next((path for path in candidates if path.is_dir()), None)
        active.append({"training_seed": seed,
                       "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else None,
                       "checkpoint_updated_utc": (datetime.fromtimestamp(
                           (run_dir / "checkpoints" / "GNN" / "latest.pt").stat().st_mtime,
                           tz=timezone.utc).isoformat()
                           if run_dir and (run_dir / "checkpoints" / "GNN" / "latest.pt").exists()
                           else None),
                       "status": "no final summary yet"})
    report = {"status": "complete" if len(completed) == len(seeds) else "in_progress",
              "refreshed_utc": datetime.now(timezone.utc).isoformat(),
              "campaign_dir": str(campaign_dir.relative_to(ROOT)),
              "planned_seeds": seeds,
              "completed_seed_count": len(completed),
              "completed": completed,
              "not_yet_complete": active,
              "interpretation": "Progress inventory only. Aggregate training-seed statistics are valid only after every planned seed has a final evaluation.",
              "source": "per-run summary.json, evaluation.json, and run-directory checkpoint timestamps"}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed": len(completed),
                      "planned": len(seeds), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
