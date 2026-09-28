"""Subprocess integration check for termination and checkpoint resume."""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_success_checkpoint_uses_reward_then_hops_as_tiebreakers():
    sys.path.insert(0, str(ROOT))
    from train import _checkpoint_scores, _is_better_success_checkpoint

    rank = _is_better_success_checkpoint
    assert rank(1.0, 9.0, 24.0, best_success=0.9, best_reward=100.0,
                best_avg_hops=23.0)
    assert rank(1.0, 10.0, 25.0, best_success=1.0, best_reward=9.0,
                best_avg_hops=23.0)
    assert rank(1.0, 10.0, 22.0, best_success=1.0, best_reward=10.0,
                best_avg_hops=23.0)
    assert not rank(1.0, 10.0, 24.0, best_success=1.0, best_reward=10.0,
                    best_avg_hops=23.0)
    rollout = {"rollout_reward": 65.0, "rollout_successes": 4.0}
    assert _checkpoint_scores({}, rollout, has_epoch_callback=True) is None
    assert _checkpoint_scores({"overall_reward": 10.0, "success_rate": 1.0},
                              rollout, has_epoch_callback=True) == (10.0, 1.0, None)


def test_process_termination_resumes_from_last_epoch(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "smoke_test.yaml").read_text(encoding="utf8"))
    config["experiment_name"] = "kill_resume_smoke"
    config["models"] = ["LSTM"]
    config["training"].update(ppo_epochs=3, steps_per_epoch=144,
                              update_epochs=1, batch_size=32)
    config["evaluation"]["interval"] = 3
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")

    previous = {path.name for path in (ROOT / "experiments" / "runs").glob("*") if path.is_dir()}
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "run_experiment.py"), "--config", str(config_path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run_dir = None
    reached_ppo_checkpoint = False
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        created = [p for p in (ROOT / "experiments" / "runs").glob("kill_resume_smoke_*")
                   if p.is_dir() and p.name not in previous]
        if created:
            run_dir = created[0]
            checkpoint = run_dir / "checkpoints" / "LSTM" / "latest.pt"
            if checkpoint.exists():
                try:
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    state = payload.get("training_state", {})
                    if state.get("phase") == "ppo" and state.get("ppo_epoch", -1) >= 0:
                        reached_ppo_checkpoint = True
                        break
                except (EOFError, RuntimeError, OSError):
                    pass  # wait until the checkpoint write has completed
        time.sleep(0.05)

    if reached_ppo_checkpoint:
        process.terminate()
    stdout, stderr = process.communicate(timeout=20)
    assert reached_ppo_checkpoint, f"process did not reach PPO checkpoint\n{stdout}\n{stderr}"
    assert process.returncode != 0, "test process should have been terminated mid-run"
    assert run_dir is not None
    checkpoint = run_dir / "checkpoints" / "LSTM" / "latest.pt"
    interrupted_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["training_state"]
    interrupted_epoch = interrupted_state["ppo_epoch"]

    resumed = subprocess.run([sys.executable, str(ROOT / "run_experiment.py"),
                               "--resume", str(run_dir)], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, f"resume failed\n{resumed.stdout}\n{resumed.stderr}"
    final_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    final_state = final_payload["training_state"]
    ppo_epochs = [row["epoch"] for row in final_state["history"] if row.get("phase") == "ppo"]
    assert ppo_epochs == [1, 2, 3]
    assert final_state["ppo_epoch"] == 2
    assert interrupted_epoch >= 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf8"))
    assert summary["status"] == "complete"
