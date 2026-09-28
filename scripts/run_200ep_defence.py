"""Extend the Mumbai-to-Kolkata defence run to 200 PPO epochs."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import BASELINE_NAMES, BaselinePolicy
from evaluation import evaluate_policy
from models import GNNActorCritic
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv
from topology import build_topology
from train import train_model


CONFIG_PATH = ROOT / "configs" / "defence_monsoon_night_200ep_mumbai_kolkata.yaml"
OUTPUT_DIR = ROOT / "experiments" / "defence_monsoon_night_200ep_mumbai_kolkata"
RESUME = ROOT / "experiments" / "defence_monsoon_night_100ep_mumbai_kolkata" / "checkpoints" / "GNN" / "latest.pt"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8"))
    torch.set_num_threads(int(cfg["cpu_threads"]))
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    e = cfg["environment"]
    kwargs = {
        "source": e["source"], "destination": e["destination"],
        "season": e["season"], "use_case": e["use_case"],
        "node_feature_mode": e["node_feature_mode"],
        "time_of_day_hours": e["time_of_day_hours"],
        "time_jitter_hours": e["time_jitter_hours"],
        "dt_seconds": e["dt_seconds"], "max_steps": e["max_steps"],
        "fiber_outage_prob": e["fiber_outage_prob"], "qber_hard": e["qber_hard"],
        "reward_protection_overrides": e["reward_protection_overrides"],
    }
    env = QKDRoutingEnv(**kwargs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    trained = train_model("GNN", env, cfg["training"], OUTPUT_DIR,
                          resume_from=RESUME,
                          dropedge_probability=float(cfg["model"]["dropedge_probability"]))
    checkpoint = Path(trained["checkpoints"]["best_by_success_rate"])
    if not checkpoint.exists():
        checkpoint = Path(trained["checkpoints"]["latest"])
    model = GNNActorCritic(hidden_dim=cfg["hidden_dim"],
                           dropedge_probability=cfg["model"]["dropedge_probability"])
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    eval_kwargs = dict(kwargs)
    eval_kwargs.pop("season", None)
    comparison = {"GNN-200ep": evaluate_policy(model, seasons=("monsoon",),
                                                  eval_seeds_per_season=10,
                                                  env_kwargs=eval_kwargs, seed_base=70000)}
    for index, name in enumerate(BASELINE_NAMES, start=1):
        comparison[name] = evaluate_policy(BaselinePolicy(name, seed=71000 + index),
                                            seasons=("monsoon",), eval_seeds_per_season=10,
                                            env_kwargs=eval_kwargs, seed_base=70000)
    graph = build_topology()
    fiber = nx.Graph()
    fiber.add_nodes_from(graph.nodes)
    fiber.add_edges_from((u, v, a) for u, v, a in graph.edges(data=True) if a["link_type"] == "fiber")
    path = nx.shortest_path(fiber, "Mumbai", "Kolkata", weight="distance_km")
    report = {"experiment": cfg["experiment_name"], "resumed_from": str(RESUME),
              "checkpoint": str(checkpoint), "conditions": kwargs,
              "fiber_only_shortest_path_hops": len(path) - 1,
              "fiber_only_shortest_path_km": nx.path_weight(fiber, path, "distance_km"),
              "comparison": comparison}
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(json.dumps({name: {"success_rate": r["success_rate"],
                             "reward": r["overall_reward"],
                             "avg_hops_success": r["avg_hops_success"]}
                      for name, r in comparison.items()}, indent=2))


if __name__ == "__main__":
    main()
