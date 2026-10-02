"""Compare a trained policy with exhaustive simple-path reward optimization.

The generated graphs are deliberately tiny and keep channels static within an
episode. The exhaustive result is exact only for this declared toy protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StaticEpisodeEnv:
    """Freeze the sampled link state after reset for exhaustive comparison."""

    def __init__(self, *args, **kwargs):
        from qkd_env import QKDRoutingEnv

        class _Env(QKDRoutingEnv):
            def __init__(self, *inner_args, **inner_kwargs):
                self._freeze_link_refresh = False
                super().__init__(*inner_args, **inner_kwargs)

            def _refresh_link_states(self):
                if not self._freeze_link_refresh:
                    super()._refresh_link_states()

            def reset(self, **reset_kwargs):
                self._freeze_link_refresh = False
                result = super().reset(**reset_kwargs)
                self._freeze_link_refresh = True
                return result

        self.env = _Env(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)


def generated_graph(seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.Graph()
    for index, city in enumerate(("Delhi", "Mumbai", "Chennai", "Kolkata",
                                  "Bangalore", "Jaipur", "Hyderabad")):
        graph.add_node(city, city=True, node_type="full_tn",
                       pos=(float(index % 3), float(index // 3)))
    cities = list(graph.nodes)
    # A random spanning cycle guarantees connectivity; random chords create
    # competing routes and vary the exact reward-optimal path by graph seed.
    edge_pairs = [(cities[i], cities[(i + 1) % len(cities)])
                  for i in range(len(cities))]
    ring_edges = {frozenset(edge) for edge in edge_pairs}
    possible = [(u, v) for i, u in enumerate(cities) for v in cities[i + 1:]
                if frozenset((u, v)) not in ring_edges]
    edge_pairs += rng.sample(possible, k=4)
    for u, v in edge_pairs:
        # Keep every link in a physically feasible fiber range so differences
        # measure path utility rather than a trivially disconnected graph.
        distance = rng.uniform(10.0, 75.0)
        graph.add_edge(u, v, link_type="fiber", distance_km=distance,
                       backbone=True, corridor=(u, v), corridor_id=f"toy-{seed}")
    return graph


def run_path(env, path: list[str], seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    actions = 0
    fallback_count = 0
    success = False
    for target in path[1:]:
        neighbors = list(env.graph.neighbors(env._current_node))
        action = neighbors.index(target)
        if not obs["edge_valid_mask"][action]:
            return {"feasible": False, "success": False, "reward": total_reward,
                    "route": path[:actions + 1], "actions": actions,
                    "fallback_count": fallback_count}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        actions += 1
        fallback_count += int(bool(info.get("fallback")))
        if info.get("success"):
            success = True
        if terminated or truncated:
            break
    return {"feasible": success and actions == len(path) - 1,
            "success": success, "reward": total_reward, "route": path,
            "actions": actions, "fallback_count": fallback_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "supplementary" /
                        "small_graph_oracle_20261002.json")
    parser.add_argument("--graph-seeds", type=int, nargs="*", default=[4101, 4102, 4103, 4104, 4105])
    parser.add_argument("--channel-seeds", type=int, nargs="*", default=[81, 82, 83, 84])
    args = parser.parse_args()

    from models import GNNActorCritic
    from ppo import load_checkpoint

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("model_name", "GNN") not in {"GNN", "GNNActorCritic"}:
        raise ValueError("This oracle runner currently supports GNN checkpoints only")
    model = GNNActorCritic(hidden_dim=int(metadata.get("hidden_dim", 64)))
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    rows = []
    for graph_seed in args.graph_seeds:
        graph = generated_graph(graph_seed)
        for channel_seed in args.channel_seeds:
            env = StaticEpisodeEnv(graph=graph, source="Delhi", destination="Chennai",
                season="normal", use_case="defence", node_feature_mode="relative",
                fiber_outage_prob=0.05, max_steps=len(graph.nodes), time_of_day_hours=12.0,
                time_jitter_hours=0.0)
            # Every simple path is executed from the same channel seed. The
            # environment subclass freezes sampled edge states after reset.
            paths = list(nx.all_simple_paths(graph, "Delhi", "Chennai",
                                             cutoff=len(graph.nodes) - 1))
            candidates = [run_path(env, path, channel_seed) for path in paths]
            feasible = [(path, outcome) for path, outcome in zip(paths, candidates)
                        if outcome["feasible"] and outcome["fallback_count"] == 0]
            if not feasible:
                rows.append({"graph_seed": graph_seed, "channel_seed": channel_seed,
                    "feasible_simple_paths": 0, "oracle_success": False,
                    "policy_success": False, "failure": "no feasible simple path"})
                continue
            oracle_path, oracle = max(feasible, key=lambda item: item[1]["reward"])

            obs, _ = env.reset(seed=channel_seed)
            policy_reward = 0.0
            policy_route = ["Delhi"]
            policy_fallbacks = 0
            policy_success = False
            for _ in range(env.max_steps):
                with torch.inference_mode():
                    action, _, _, _ = model.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                policy_reward += float(reward)
                if info.get("node"):
                    policy_route.append(info["node"])
                policy_fallbacks += int(bool(info.get("fallback")))
                policy_success = policy_success or bool(info.get("success"))
                if terminated or truncated:
                    break
            rows.append({"graph_seed": graph_seed, "channel_seed": channel_seed,
                "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                "enumerated_simple_paths": len(paths), "feasible_simple_paths": len(feasible),
                "oracle_success": True, "oracle_route": oracle_path,
                "oracle_reward": oracle["reward"], "policy_success": policy_success,
                "policy_route": policy_route, "policy_reward": policy_reward,
                "policy_fallbacks": policy_fallbacks,
                "reward_regret": oracle["reward"] - policy_reward,
                "policy_route_simple": len(policy_route) == len(set(policy_route)),
                "oracle_rank_of_policy_reward": (1 + sum(o["reward"] > policy_reward
                    for _, o in feasible))})
    valid_rows = [row for row in rows if row.get("oracle_success")]
    successful_rows = [row for row in valid_rows if row["policy_success"]]
    report = {"protocol": "exact exhaustive simple-path reward oracle; static episode channel states",
        "scope_note": "Toy generated graphs only. This is not an oracle for the full dynamic network or a globally optimal real QKD route.",
        "channel_protocol": {"link_type": "fiber-only", "independent_per_edge_episode_outage_probability": 0.05,
            "channels_are_frozen_after_reset": True},
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source": Path(__file__).name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "graph_seeds": args.graph_seeds, "channel_seeds": args.channel_seeds,
        "summary": {"episodes": len(rows), "with_feasible_oracle": len(valid_rows),
            "policy_success_rate": (sum(r["policy_success"] for r in valid_rows) / len(valid_rows)
                                    if valid_rows else None),
            "mean_reward_regret": (sum(r["reward_regret"] for r in valid_rows) / len(valid_rows)
                                   if valid_rows else None),
            "median_reward_regret": (statistics.median(r["reward_regret"] for r in valid_rows)
                                     if valid_rows else None),
            "successful_policy_episodes": len(successful_rows),
            "successful_policy_mean_reward_regret": (statistics.mean(
                r["reward_regret"] for r in successful_rows) if successful_rows else None),
            "successful_policy_median_reward_regret": (statistics.median(
                r["reward_regret"] for r in successful_rows) if successful_rows else None),
            "successful_policy_reward_optimal_fraction": (sum(abs(r["reward_regret"]) < 1e-9
                for r in successful_rows) / len(successful_rows) if successful_rows else None),
            "policy_reward_optimal_fraction": (sum(abs(r["reward_regret"]) < 1e-9
                for r in valid_rows) / len(valid_rows) if valid_rows else None)},
        "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
