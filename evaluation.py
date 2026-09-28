"""Seeded median evaluation, probabilistic reachability, and inference timing."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from baselines import BASELINE_NAMES, BaselinePolicy
from models import GNNActorCritic, LSTMActorCritic
from physics import FIBER_OUTAGE_PROB, QBER_HARD, SEASONS, sample_link_state
from ppo import load_checkpoint
from qkd_env import QKDRoutingEnv


def _policy_action(policy, observation, env):
    if hasattr(policy, "act") and isinstance(policy, BaselinePolicy):
        return policy.act(observation, env)
    with torch.no_grad():
        action, _, _, _ = policy.act(observation, deterministic=True)
    return action


def _run_episode(policy, env: QKDRoutingEnv, seed: int, max_steps: int | None = None):
    obs, _ = env.reset(seed=seed)
    route = [env.source]
    reward = 0.0
    success = False
    physical_checks = []
    fallback_count = 0
    link_type_counts = {"fiber": 0, "fso": 0}
    for _ in range(max_steps or env.max_steps):
        action = _policy_action(policy, obs, env)
        obs, r, terminated, truncated, info = env.step(action)
        reward += float(r)
        if info.get("node"):
            route.append(info["node"])
        if "qber" in info and "skr" in info:
            physical_checks.append(float(info["qber"]) < env.qber_hard and float(info["skr"]) > 0.0)
            if info.get("link_type") in link_type_counts:
                link_type_counts[info["link_type"]] += 1
        fallback_count += int(bool(info.get("fallback", False)))
        if info.get("success"):
            success = True
        if terminated or truncated:
            break
    return {"reward": reward, "success": success,
            "source": env.source, "destination": env.destination,
            "hops": len(route) - 1 if success else None,
            "route": route, "revisits": len(route) - len(set(route)),
            "qber_skr_validity": {"checked_edges": len(physical_checks),
                "valid_edges": sum(physical_checks), "invalid_edges": len(physical_checks) - sum(physical_checks),
                "valid_rate": float(np.mean(physical_checks)) if physical_checks else None},
            "fallback_count": fallback_count,
            "link_type_counts": link_type_counts}


def evaluate_policy(policy, *, seasons: tuple[str, ...] = SEASONS,
                    eval_seeds_per_season: int = 4,
                    env_kwargs: dict[str, Any] | None = None,
                    seed_base: int = 50_000) -> dict[str, Any]:
    """Evaluate each season with multiple distinct seeds and median rewards."""
    if eval_seeds_per_season < 1:
        raise ValueError("eval_seeds_per_season must be positive")
    env_kwargs = dict(env_kwargs or {})
    was_training = getattr(policy, "training", None)
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "enable_encoder_cache"):
        policy.enable_encoder_cache(True)
    per_season: dict[str, list[dict[str, Any]]] = {}
    all_results = []
    representative = None
    zero_success_epochs = 0
    for season_index, season in enumerate(seasons):
        if season not in SEASONS:
            raise ValueError(f"invalid season {season!r}")
        season_runs = []
        for seed_index in range(eval_seeds_per_season):
            seed = seed_base + season_index * 10_000 + seed_index
            env = QKDRoutingEnv(season=season, **env_kwargs)
            result = _run_episode(policy, env, seed)
            result.update(season=season, seed=seed)
            season_runs.append(result)
            all_results.append(result)
            if representative is None:
                representative = result
        per_season.setdefault(season, []).extend(season_runs)
        if not any(run["success"] for run in season_runs):
            zero_success_epochs += 1
    summaries = {}
    for season, runs in per_season.items():
        successes = [r for r in runs if r["success"]]
        summaries[season] = {
            "reward_median": float(statistics.median(r["reward"] for r in runs)),
            "reward_std": float(np.std([r["reward"] for r in runs])),
            "seed_rewards": [{"seed": r["seed"], "source": r["source"],
                              "destination": r["destination"], "reward": r["reward"],
                              "success": r["success"], "hops": r["hops"]} for r in runs],
            "success_rate": len(successes) / len(runs),
            "avg_hops_success": float(np.mean([r["hops"] for r in successes])) if successes else None,
            "hop_ratio_to_optimal": (float(np.mean([r["hops"] for r in successes])) / 23.0
                                     if successes else None),
            "episodes": len(runs),
            "sample_path": runs[0]["route"],
            "sample_revisits": runs[0]["revisits"],
            "qber_skr_validity": _combine_validity(runs),
        }
    successes = [r for r in all_results if r["success"]]
    summary = {
        "overall_reward": float(statistics.median(r["reward"] for r in all_results)),
        "seed_reward_std": float(np.std([r["reward"] for r in all_results])),
        "per_season": summaries,
        "success_rate": len(successes) / len(all_results),
        "avg_hops_success": float(np.mean([r["hops"] for r in successes])) if successes else None,
        "successful_hop_counts": [r["hops"] for r in successes],
        "optimal_hops": 23,
        "avg_hop_ratio_to_optimal": (float(np.mean([r["hops"] for r in successes])) / 23.0
                                      if successes else None),
        "sample_path": representative["route"] if representative else [],
            "sample_path_revisits": representative["revisits"] if representative else 0,
        "sample_link_type_counts": (representative["link_type_counts"] if representative else {}),
        "zero_success_epochs": zero_success_epochs,
        "qber_skr_validity": _combine_validity(all_results),
        "episodes": len(all_results),
    }
    if hasattr(policy, "enable_encoder_cache"):
        policy.enable_encoder_cache(False)
    if was_training and hasattr(policy, "train"):
        policy.train()
    return summary


def _combine_validity(results: list[dict[str, Any]]) -> dict[str, Any]:
    checked = sum(r["qber_skr_validity"]["checked_edges"] for r in results)
    valid = sum(r["qber_skr_validity"]["valid_edges"] for r in results)
    invalid = sum(r["qber_skr_validity"]["invalid_edges"] for r in results)
    return {"checked_edges": checked, "valid_edges": valid, "invalid_edges": invalid,
            "valid_rate": (valid / checked) if checked else None,
            "fallback_count": sum(r.get("fallback_count", 0) for r in results)}


def evaluate_all_baselines(*, seasons=SEASONS, eval_seeds_per_season=4,
                           env_kwargs=None, seed_base=60_000,
                           baseline_names: tuple[str, ...] = BASELINE_NAMES):
    return {name: evaluate_policy(BaselinePolicy(name, seed_base), seasons=tuple(seasons),
                                  eval_seeds_per_season=eval_seeds_per_season,
                                  env_kwargs=env_kwargs, seed_base=seed_base + i * 100_000)
            for i, name in enumerate(baseline_names)}


def probabilistic_reachability_check(*, trials_per_hour: int = 10,
                                     seasons: tuple[str, ...] = SEASONS,
                                     threshold: float = 0.02,
                                     seed: int = 8080) -> dict[str, Any]:
    """Sample independent link states for each season/hour and test connectivity."""
    graph = QKDRoutingEnv().graph
    rng = random.Random(seed)
    report = {}
    for season in seasons:
        unreachable = total = 0
        for hour in range(24):
            for _ in range(trials_per_hour):
                sampled = nx.Graph()
                sampled.add_nodes_from(graph.nodes)
                # FSO segments within a single detour share a weather corridor.
                detour_draws = {}
                for _, _, attrs in graph.edges(data=True):
                    if attrs["link_type"] == "fso":
                        detour = attrs.get("detour_for")
                        if detour not in detour_draws:
                            detour_draws[detour] = rng.random()
                for u, v, attrs in graph.edges(data=True):
                    state = sample_link_state(attrs["link_type"], attrs["distance_km"],
                                              season, hour, rng,
                                              fiber_outage_prob=FIBER_OUTAGE_PROB,
                                              outage_uniform=(detour_draws.get(attrs.get("detour_for"))
                                                  if attrs["link_type"] == "fso" else None),
                                              outage_distance_km=(attrs.get("outage_reference_distance_km")
                                                  if attrs["link_type"] == "fso" else None))
                    if state["qber"] < QBER_HARD and state["skr"] > 0:
                        sampled.add_edge(u, v)
                total += 1
                if not nx.has_path(sampled, "Delhi", "Chennai"):
                    unreachable += 1
        fraction = unreachable / total
        report[season] = {"trials": total, "unreachable": unreachable,
                          "unreachable_fraction": fraction}
        if fraction >= threshold:
            raise AssertionError(f"{season}: unreachable draw fraction {fraction:.4f} >= {threshold}")
    return {"by_season": report, "threshold": threshold,
            "overall_unreachable_fraction": sum(v["unreachable"] for v in report.values()) /
                sum(v["trials"] for v in report.values())}


def time_inference(checkpoint: str | Path, *, model_name: str | None = None,
                   decisions: int = 100, season: str = "normal", seed: int = 919):
    """Measure checkpoint load separately from isolated decisions and full route."""
    if decisions < 1:
        raise ValueError("decisions must be positive")
    checkpoint = Path(checkpoint)
    start_load = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("metadata", {})
    inferred = model_name or meta.get("model_name", "LSTMActorCritic")
    cls = GNNActorCritic if inferred in {"GNN", "GNNActorCritic"} else LSTMActorCritic
    model = cls(hidden_dim=int(meta.get("hidden_dim", 64)))
    load_checkpoint(checkpoint, model)
    model.eval()
    load_seconds = time.perf_counter() - start_load
    env = QKDRoutingEnv(season=season)
    obs, _ = env.reset(seed=seed)
    samples = []
    with torch.no_grad():
        for index in range(decisions):
            start = time.perf_counter()
            action, _, _, _ = model.act(obs, deterministic=True)
            samples.append(time.perf_counter() - start)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + index + 1)

    # Time one complete environment episode separately from the isolated
    # decisions above; a failed route still runs to its terminal failure.
    obs, _ = env.reset(seed=seed + decisions + 1)
    route_start = time.perf_counter()
    route = [env.source]
    route_success = False
    route_failure = None
    with torch.no_grad():
        for _ in range(env.max_steps):
            action, _, _, _ = model.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("node"):
                route.append(info["node"])
            if info.get("success"):
                route_success = True
            if terminated or truncated:
                route_failure = info.get("failure")
                break
    route_seconds = time.perf_counter() - route_start
    arr = np.asarray(samples)
    return {"checkpoint_load_seconds": load_seconds, "decisions_timed": len(samples),
            "decision_seconds_median": float(np.median(arr)),
            "decision_seconds_mean": float(np.mean(arr)),
            "decision_seconds_std": float(np.std(arr)),
            "decision_seconds_min": float(np.min(arr)),
            "decision_seconds_max": float(np.max(arr)),
            "test_route_wall_seconds": route_seconds, "test_route": route,
            "test_route_success": route_success, "test_route_failure": route_failure,
            "test_route_revisits": len(route) - len(set(route))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reachability", action="store_true")
    parser.add_argument("--trials-per-hour", type=int, default=10)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", choices=("GNN", "LSTM"), default="LSTM")
    args = parser.parse_args()
    if args.reachability:
        print(json.dumps(probabilistic_reachability_check(
            trials_per_hour=args.trials_per_hour), indent=2))
    elif args.checkpoint:
        print(json.dumps(time_inference(args.checkpoint, model_name=args.model), indent=2))
    else:
        results = evaluate_all_baselines()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
