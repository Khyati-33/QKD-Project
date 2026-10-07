"""Train one final architecture on frozen held-out partitions across five seeds."""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_experiment import run_experiment

SEEDS = (20261002, 20261003, 20261004, 20261005, 20261006)


def write_report(path: Path, model: str, records: list[dict], config_path: Path):
    payload = {"status": ("complete" if len(records) == len(SEEDS) and
                           all(r["status"] == "complete" for r in records)
                           else "in_progress"), "model": model, "seeds": list(SEEDS),
               "source_config": str(config_path.relative_to(ROOT)),
               "updated_utc": datetime.now(timezone.utc).isoformat(), "runs": records}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("GNN", "MLP"), required=True)
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs" / "final_holdout_50ep.yaml")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_dir if args.campaign_dir.is_absolute() else ROOT / args.campaign_dir
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "configs").mkdir(exist_ok=True)
    records_path = campaign / "seed_runs.json"
    records_by_seed = {}
    if records_path.exists():
        old = json.loads(records_path.read_text(encoding="utf-8"))
        records_by_seed = {int(r["training_seed"]): r for r in old.get("runs", [])}
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for seed in SEEDS:
        existing = records_by_seed.get(seed)
        if existing and existing["status"] == "complete":
            continue
        config = dict(base)
        config["models"] = [args.model]
        config["seed"] = int(seed)
        config["experiment_name"] = f"final_holdout_{args.model.lower()}_{seed}"
        config_path = campaign / "configs" / f"seed_{seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        prior_run = None
        if existing and existing.get("run_dir"):
            prior_run = ROOT / existing["run_dir"]
        if prior_run is None:
            candidates = sorted((ROOT / "experiments" / "runs").glob(
                f"final_holdout_{args.model.lower()}_{seed}_*"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            prior_run = next((p for p in candidates if
                (p / "checkpoints" / args.model / "latest.pt").exists()), None)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        if prior_run:
            run_dir = prior_run
            resumed = run_experiment(config_path, resume=str(run_dir))
        else:
            resumed = run_experiment(config_path)
        summary = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
        checkpoint = resumed / "checkpoints" / args.model / "best_by_success_rate.pt"
        if not checkpoint.exists():
            checkpoint = resumed / "checkpoints" / args.model / "latest.pt"
        completed_at = datetime.now(timezone.utc)
        records_by_seed[seed] = {"training_seed": seed,
            "run_dir": str(resumed.relative_to(ROOT)), "status": summary["status"],
            "checkpoint_selection": "best_by_success_rate on fixed monsoon validation seeds",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "resolved_config": str((resumed / "configs" / "resolved.yaml").relative_to(ROOT)),
            "device": config.get("device", "cpu"),
            "cpu_threads": int(config.get("cpu_threads", 4)),
            "wall_seconds_this_campaign_process": time.perf_counter() - started,
            "started_utc": started_at.isoformat(),
            "completed_utc": completed_at.isoformat()}
        records = [records_by_seed[s] for s in sorted(records_by_seed)]
        write_report(records_path, args.model, records, args.config)
        print(json.dumps({"model": args.model, "seed": seed,
                          "status": summary["status"], "run_dir": str(resumed)}), flush=True)
    write_report(records_path, args.model,
                 [records_by_seed[s] for s in sorted(records_by_seed)], args.config)


if __name__ == "__main__":
    main()
