"""Inspect checkpoints and refresh progress for the matched-ablation campaign."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PLANNED = ["baseline", "geographic_features", "no_skr_reward", "no_dropedge", "hidden_dim_32"]


def main() -> None:
    campaigns = sorted((ROOT / "experiments" / "runs").glob("matched_ablations_*"),
                       key=lambda path: path.stat().st_mtime)
    if not campaigns:
        raise FileNotFoundError("no matched-ablation campaign directory exists")
    campaign = campaigns[-1]
    progress_path = campaign / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    completed = progress.get("completed_variants", [])
    done_names = {row["variant"] for row in completed}
    running = next((name for name in PLANNED if name not in done_names), None)
    state = None
    run_dir = None
    if running:
        matches = sorted((ROOT / "experiments" / "runs").glob(
            f"matched_ablation_{running}_*"), key=lambda path: path.stat().st_mtime)
        if matches:
            run_dir = matches[-1]
            checkpoint = run_dir / "checkpoints" / "GNN" / "latest.pt"
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = payload.get("training_state", {})
    progress.update({"status": "complete" if len(done_names) == len(PLANNED) else "in_progress",
        "campaign_dir": str(campaign.relative_to(ROOT)), "completed_variants": completed,
        "planned_variants": PLANNED, "running_variant": running,
        "running_run_dir": str(run_dir.relative_to(ROOT)) if run_dir else None,
        "training_state": ({"phase": state.get("phase"), "bc_epoch": state.get("bc_epoch"),
            "ppo_epoch": state.get("ppo_epoch"), "target_ppo_epochs": 50} if state else None)})
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    (ROOT / "paper" / "supplementary" / "matched_budget_ablations_progress_20261002.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
