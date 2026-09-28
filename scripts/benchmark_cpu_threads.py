"""Find the fastest CPU thread count for one-step GNN inference."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_engine import act
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import CITIES


CKPT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUT = ROOT / "experiments" / "cpu_thread_sweep.json"


def main() -> None:
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CKPT, model, map_location="cpu")
    model.eval()
    observations = []
    for i, source in enumerate(CITIES):
        for j, destination in enumerate(CITIES):
            if source != destination:
                env = QKDRoutingEnv(source=source, destination=destination, season="monsoon",
                    use_case="defence", node_feature_mode="relative", time_of_day_hours=22,
                    time_jitter_hours=1, fiber_outage_prob=0.01)
                observations.append(env.reset(seed=91000 + i * 100 + j)[0])
    report = {}
    for threads in (1, 2, 4, 8):
        torch.set_num_threads(threads)
        for obs in observations[:2]:
            act(model, obs)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            for obs in observations:
                act(model, obs)
            samples.append(time.perf_counter() - start)
        report[str(threads)] = {"batch_of_42_seconds_median": float(np.median(samples)),
            "per_decision_milliseconds": float(np.median(samples) / len(observations) * 1000)}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
