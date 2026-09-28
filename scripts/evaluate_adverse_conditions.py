"""Compare the 200-epoch agent and baselines under adverse conditions."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import BASELINE_NAMES, BaselinePolicy
from evaluation import evaluate_policy
from models import GNNActorCritic
from ppo import load_checkpoint


CHECKPOINT = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"
OUTPUT = ROOT / "experiments" / "adverse_conditions_comparison.json"

BASE = {
    "source": "Mumbai", "destination": "Kolkata", "use_case": "defence",
    "node_feature_mode": "relative", "time_of_day_hours": 22.0,
    "time_jitter_hours": 1.0, "dt_seconds": 300.0, "max_steps": 400,
    "fiber_outage_prob": 0.01, "qber_hard": 0.11,
    "reward_protection_overrides": {"step_cost": -0.05, "revisit": -0.6,
        "switch": -0.2, "security_penalty": -30.0, "depletion_penalty": -60.0,
        "backtrack": -0.5},
}

SCENARIOS = {
    "fixed_monsoon_night": {"season": "monsoon"},
    "random_endpoints_all_seasons": {"season": "monsoon", "randomize_endpoints": True},
    "high_fiber_outage": {"season": "monsoon", "fiber_outage_prob": 0.10},
    "severe_time_jitter": {"season": "monsoon", "time_jitter_hours": 6.0},
    "all_seasons": {"season": "monsoon", "seasons": ("normal", "summer", "winter", "monsoon")},
    "high_outage_and_time_jitter": {"season": "monsoon", "fiber_outage_prob": 0.10,
                                     "time_jitter_hours": 6.0, "randomize_endpoints": True},
}


def main() -> None:
    torch.set_num_threads(8)
    random.seed(20260928); np.random.seed(20260928); torch.manual_seed(20260928)
    model = GNNActorCritic(hidden_dim=64, dropedge_probability=0.05)
    load_checkpoint(CHECKPOINT, model, map_location="cpu")
    model.eval()
    methods = {"GNN-200ep": model}
    for index, name in enumerate(BASELINE_NAMES, start=1):
        methods[name] = BaselinePolicy(name, seed=71000 + index)
    report = {"checkpoint": str(CHECKPOINT), "seeds_per_method": 3, "scenarios": {}}
    for scenario, overrides in SCENARIOS.items():
        kwargs = dict(BASE); kwargs.update(overrides)
        seasons = kwargs.pop("seasons", ("monsoon",))
        kwargs.pop("season", None)
        report["scenarios"][scenario] = {name: evaluate_policy(
            policy, seasons=seasons, eval_seeds_per_season=3,
            env_kwargs=kwargs, seed_base=88000)
            for name, policy in methods.items()}
    OUTPUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    for scenario, results in report["scenarios"].items():
        print(scenario)
        for name, result in results.items():
            print(name, result["success_rate"], result["avg_hops_success"],
                  round(result["overall_reward"], 3))


if __name__ == "__main__":
    main()
