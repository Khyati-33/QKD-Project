"""Config-driven reproducible runner with protected-condition checks/resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from baselines import BASELINE_NAMES
from evaluation import evaluate_all_baselines, evaluate_policy
from physics import FIBER_HARDWARE_PROFILE_ID
from qkd_env import QKDRoutingEnv
from train import MODEL_TYPES, train_model


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTECTED = {
    "source_destination": {"source": "Delhi", "destination": "Chennai"},
    "qber_hard": 0.11,
    "fiber_outage_prob": 0.01,
    "seasons": ["normal", "summer", "winter", "monsoon"],
    "eval_seeds_per_season": 4,
    "trust_chain_reset_rule": "full_tn_clears_hop_qbers",
    "reward_hacking_protections": {"step_cost": -0.1, "revisit": -1.0,
        "switch": -0.25, "security_penalty": -10.0, "depletion_penalty": -5.0,
        "backtrack": -0.5},
    "evaluation_baselines": list(BASELINE_NAMES),
    "randomize_endpoints": False,
}


def resolve_config(raw: dict[str, Any]) -> dict[str, Any]:
    protected = raw.get("protected_conditions", {})
    resolved_protected = {}
    for key, expected in DEFAULT_PROTECTED.items():
        entry = protected.get(key, {"value": expected, "override": False})
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(f"protected_conditions.{key} must include value and override")
        value = entry["value"]
        override = bool(entry.get("override", False))
        if value != expected and not override:
            raise ValueError(
                f"protected condition {key!r} changed from {expected!r} to {value!r}; "
                "set override: true next to this setting to run a controlled experiment")
        resolved_protected[key] = {"value": value, "override": override}
    unknown = set(protected) - set(DEFAULT_PROTECTED)
    if unknown:
        raise ValueError(f"unknown protected conditions: {sorted(unknown)}")
    model_names = raw.get("models", ["LSTM"])
    if not model_names or any(m not in MODEL_TYPES for m in model_names):
        raise ValueError(f"models must be chosen from {tuple(MODEL_TYPES)}")
    env = dict(raw.get("environment", {}))
    training = dict(raw.get("training", {}))
    evaluation = dict(raw.get("evaluation", {}))
    physics = dict(raw.get("physics", {}))
    fiber_profile = physics.get("fiber_detector_profile", FIBER_HARDWARE_PROFILE_ID)
    if fiber_profile != FIBER_HARDWARE_PROFILE_ID:
        raise ValueError(f"unsupported fiber detector profile {fiber_profile!r}; "
                         f"only {FIBER_HARDWARE_PROFILE_ID!r} is implemented")
    fso_profile = physics.get("fso_channel_profile", "785nm_weather_turbulence_scenario")
    if fso_profile != "785nm_weather_turbulence_scenario":
        raise ValueError(f"unsupported FSO channel profile {fso_profile!r}")
    evaluation.setdefault("seasons", resolved_protected["seasons"]["value"])
    requested_seeds = int(evaluation.get("eval_seeds_per_season",
                                        resolved_protected["eval_seeds_per_season"]["value"]))
    protected_seeds = resolved_protected["eval_seeds_per_season"]["value"]
    if requested_seeds != protected_seeds and not resolved_protected["eval_seeds_per_season"]["override"]:
        raise ValueError("evaluation.eval_seeds_per_season changes protected condition; "
                         "set protected_conditions.eval_seeds_per_season.override: true")
    eval_seeds = requested_seeds
    env["source"] = resolved_protected["source_destination"]["value"]["source"]
    env["destination"] = resolved_protected["source_destination"]["value"]["destination"]
    env["randomize_endpoints"] = resolved_protected["randomize_endpoints"]["value"]
    env["fiber_outage_prob"] = resolved_protected["fiber_outage_prob"]["value"]
    env["qber_hard"] = resolved_protected["qber_hard"]["value"]
    env["reward_protection_overrides"] = resolved_protected["reward_hacking_protections"]["value"]
    for unsupported in ("seasons", "trust_chain_reset_rule"):
        if resolved_protected[unsupported]["value"] != DEFAULT_PROTECTED[unsupported]:
            raise ValueError(f"override for protected condition {unsupported!r} is not implemented")
    for alias, protected_key in (("source", "source_destination"),
                                 ("destination", "source_destination"),
                                 ("randomize_endpoints", "randomize_endpoints"),
                                 ("fiber_outage_prob", "fiber_outage_prob"),
                                 ("qber_hard", "qber_hard"),
                                 ("reward_protection_overrides", "reward_hacking_protections")):
        if alias in env:
            expected = resolved_protected[protected_key]["value"]
            if protected_key == "source_destination":
                expected = expected[alias]
            if env[alias] != expected and not resolved_protected[protected_key]["override"]:
                raise ValueError(f"environment.{alias} changes protected condition "
                                 f"{protected_key}; set that condition's override: true")
    return {
        "experiment_name": raw.get("experiment_name", "qkd_run"),
        "models": model_names,
        "seed": int(raw.get("seed", 1234)),
        "hidden_dim": int(raw.get("hidden_dim", 64)),
        "device": raw.get("device", "cpu"),
        "cpu_threads": int(raw.get("cpu_threads", min(8, os.cpu_count() or 1))),
        "model": {"dropedge_probability": float(raw.get("model", {}).get(
                            "dropedge_probability", 0.0))},
        "physics": {"fiber_detector_profile": fiber_profile,
                    "fso_channel_profile": fso_profile},
        "environment": {"season": env.get("season", "normal"),
            "use_case": env.get("use_case", "standard"),
            "node_feature_mode": env.get("node_feature_mode", "geographic"),
            "time_jitter_hours": float(env.get("time_jitter_hours", 0.0)),
            "max_steps": int(env.get("max_steps", 100)),
            "time_of_day_hours": float(env.get("time_of_day_hours", 12.0)) % 24.0,
            "dt_seconds": float(env.get("dt_seconds", 300.0)),
            "disabled_reward_terms": list(env.get("disabled_reward_terms", [])),
            "randomize_endpoints": env["randomize_endpoints"],
            "fiber_outage_prob": float(env["fiber_outage_prob"]),
            "qber_hard": float(env["qber_hard"]),
            "reward_protection_overrides": dict(env["reward_protection_overrides"]),
            "source": env["source"], "destination": env["destination"]},
        "training": {
            "bc_epochs": int(training.get("bc_epochs", 15)),
            "bc_episodes_per_epoch": int(training.get("bc_episodes_per_epoch", 8)),
            "bc_learning_rate": float(training.get("bc_learning_rate", 1e-3)),
            "bc_entropy_coef": float(training.get("bc_entropy_coef", 0.02)),
            "ppo_epochs": int(training.get("ppo_epochs", 100)),
            "steps_per_epoch": int(training.get("steps_per_epoch", 144)),
            "update_epochs": int(training.get("update_epochs", 4)),
            "batch_size": int(training.get("batch_size", 64)),
            "learning_rate": float(training.get("learning_rate", 3e-5)),
            "learning_rate_by_model": {str(k): float(v) for k, v in
                                       training.get("learning_rate_by_model", {}).items()},
            "clip_epsilon": float(training.get("clip_epsilon", 0.1)),
            "entropy_start": float(training.get("entropy_start", 0.003)),
            "critic_warmup_epochs": int(training.get("critic_warmup_epochs", 5)),
            "gamma": float(training.get("gamma", 0.99)),
            "gae_lambda": float(training.get("gae_lambda", 0.95)),
            "actor_learning_rate": float(training.get("actor_learning_rate",
                                                       training.get("learning_rate", 3e-5))),
            "critic_learning_rate": float(training.get("critic_learning_rate",
                                                        training.get("learning_rate", 3e-5))),
            "value_clip_epsilon": (None if training.get("value_clip_epsilon", 0.2) is None else
                                   float(training.get("value_clip_epsilon", 0.2))),
            "max_grad_norm": float(training.get("max_grad_norm", 0.5)),
            "reward_normalization": bool(training.get("reward_normalization", False)),
        },
        "evaluation": {"seasons": list(evaluation["seasons"]),
            "eval_seeds_per_season": eval_seeds,
            "interval": int(evaluation.get("interval", 5))},
        "protected_conditions": resolved_protected,
    }


def set_seeds(seed: int, cpu_threads: int | None = None) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)


def git_commit_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _snapshot_sources(run_dir: Path, *, stage: str = "pre_training") -> None:
    """Archive and hash the exact working sources used, including dirty files."""
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    candidates = list(ROOT.glob("*.py"))
    for folder_name in ("configs", "hardware_profiles", "tests"):
        folder = ROOT / folder_name
        if folder.exists():
            candidates.extend(path for path in folder.rglob("*") if path.is_file())
    candidates.extend(path for path in (ROOT / "requirements.txt",
        ROOT / "PHYSICS_EVIDENCE.md", ROOT / "EXPERIMENT_PROTOCOL.md",
        ROOT / "pytest.ini") if path.exists())
    files = {}
    for source in sorted(set(candidates)):
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[str(relative).replace("\\", "/")] = hashlib.sha256(source.read_bytes()).hexdigest()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    manifest = {"snapshot_stage": stage, "git_commit": git_commit_hash(),
                "working_tree_status": status.splitlines(),
                "source_files_sha256": files}
    (run_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")


def _svg_line_plot(title: str, values: list[float], path: Path, ylabel: str) -> None:
    width, height, margin = 800, 420, 60
    usable_w, usable_h = width - 2 * margin, height - 2 * margin
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        finite = [0.0]
    lo, hi = min(finite), max(finite)
    if hi == lo:
        hi = lo + 1.0
    points = []
    for i, value in enumerate(finite):
        x = margin + usable_w * (i / max(len(finite) - 1, 1))
        y = margin + usable_h * (hi - value) / (hi - lo)
        points.append(f"{x:.1f},{y:.1f}")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>'
           f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" '
           f'text-anchor="middle" font-size="13">{ylabel}</text>'
           f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
           f'y2="{height-margin}" stroke="black"/>'
           f'<polyline fill="none" stroke="#145da0" stroke-width="2" points="{" ".join(points)}"/>'
           f'<text x="{margin}" y="{height-18}" font-size="12">1</text>'
           f'<text x="{width-margin}" y="{height-18}" text-anchor="end" font-size="12">'
           f'{len(finite)}</text></svg>')
    path.write_text(svg, encoding="utf8")


def _svg_hop_histogram(title: str, values: list[int], path: Path) -> None:
    width, height, margin = 800, 420, 60
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        counts = {0: 0}
    max_count = max(max(counts.values()), 1)
    bw = (width - 2 * margin) / max(len(counts), 1)
    bars = []
    for i, (hop, count) in enumerate(sorted(counts.items())):
        bar_h = (height - 2 * margin) * count / max_count
        x, y = margin + i * bw, height - margin - bar_h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.75:.1f}" '
                    f'height="{bar_h:.1f}" fill="#e07a2d"/><text x="{x+bw*.35:.1f}" '
                    f'y="{height-margin+18}" text-anchor="middle" font-size="12">{hop}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>'
           f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">Hops</text>'
           + "".join(bars) + "</svg>")
    path.write_text(svg, encoding="utf8")


def _load_raw_config(config_path: Path | None, resume: str | None):
    if resume and config_path is None:
        resume_path = Path(resume)
        run_dir = resume_path if resume_path.is_dir() else resume_path.parents[2]
        config_path = run_dir / "configs" / "resolved.yaml"
    if config_path is None:
        raise ValueError("--config is required unless --resume points to an existing run")
    raw = yaml.safe_load(config_path.read_text(encoding="utf8"))
    return raw, config_path


def run_experiment(config_path: Path | None, *, resume: str | None = None,
                   stop_after_ppo_epochs: int | None = None) -> Path:
    raw, actual_config_path = _load_raw_config(config_path, resume)
    cfg = resolve_config(raw)
    set_seeds(cfg["seed"], cfg["cpu_threads"])
    if resume:
        resume_path = Path(resume)
        run_dir = resume_path if resume_path.is_dir() else resume_path.parents[2]
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        commit = (run_dir / "git_commit.txt").read_text().strip() if (run_dir / "git_commit.txt").exists() else git_commit_hash()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = ROOT / "experiments" / "runs" / f"{cfg['experiment_name']}_{run_id}"
        commit = git_commit_hash()
    for folder in ("checkpoints", "plots", "logs", "routes", "configs"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf8")
    (run_dir / "configs" / "resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    source_config_path = run_dir / "configs" / "source.yaml"
    if not resume or not source_config_path.exists():
        source_config_path.write_text(actual_config_path.read_text(encoding="utf8"), encoding="utf8")
    if not resume and not (run_dir / "source_manifest.json").exists():
        _snapshot_sources(run_dir)
    (run_dir / "logs" / "runner.log").open("a", encoding="utf8").write(
        f"run start/resume: {datetime.now(timezone.utc).isoformat()}\n")

    all_metrics = {}
    eval_records = json.loads((run_dir / "evaluation.json").read_text(encoding="utf8")) \
        if resume and (run_dir / "evaluation.json").exists() else []
    route_path = run_dir / "routes" / "representative_routes.json"
    route_records = json.loads(route_path.read_text(encoding="utf8")) \
        if resume and route_path.exists() else []
    hop_counts = []
    for record in eval_records:
        hop_counts.extend(record.get("policy", {}).get("successful_hop_counts", []))
        for baseline in record.get("baselines", {}).values():
            hop_counts.extend(baseline.get("successful_hop_counts", []))
    training_cfg = dict(cfg["training"])
    training_cfg.update(seed=cfg["seed"], hidden_dim=cfg["hidden_dim"], device=cfg["device"])
    for model_index, model_name in enumerate(cfg["models"]):
        model_training_cfg = dict(training_cfg)
        model_specific_lr = training_cfg["learning_rate_by_model"].get(model_name)
        if model_specific_lr is not None:
            model_training_cfg["learning_rate"] = model_specific_lr
            model_training_cfg["actor_learning_rate"] = model_specific_lr
        model_eval_records = []
        interval = cfg["evaluation"]["interval"]
        epochs = model_training_cfg["ppo_epochs"]
        seasons = tuple(cfg["evaluation"]["seasons"])
        eval_seeds = cfg["evaluation"]["eval_seeds_per_season"]

        def callback(model, epoch):
            if epoch % interval and epoch != epochs:
                return {}
            env_kwargs = {"source": cfg["environment"]["source"],
                          "destination": cfg["environment"]["destination"],
                          "disabled_reward_terms": set(cfg["environment"]["disabled_reward_terms"]),
                          "randomize_endpoints": cfg["environment"]["randomize_endpoints"],
                          "fiber_outage_prob": cfg["environment"]["fiber_outage_prob"],
                          "qber_hard": cfg["environment"]["qber_hard"],
                          "reward_protection_overrides": cfg["environment"]["reward_protection_overrides"],
                          "max_steps": cfg["environment"]["max_steps"],
                          "use_case": cfg["environment"]["use_case"],
                          "node_feature_mode": cfg["environment"]["node_feature_mode"],
                          "time_jitter_hours": cfg["environment"]["time_jitter_hours"],
                          "time_of_day_hours": cfg["environment"]["time_of_day_hours"],
                          "dt_seconds": cfg["environment"]["dt_seconds"]}
            policy_result = evaluate_policy(model, seasons=seasons,
                eval_seeds_per_season=eval_seeds, env_kwargs=env_kwargs,
                seed_base=70_000 + model_index * 500_000 + epoch * 1_000)
            baseline_result = evaluate_all_baselines(seasons=seasons,
                eval_seeds_per_season=eval_seeds, env_kwargs=env_kwargs,
                seed_base=80_000 + model_index * 500_000 + epoch * 1_000,
                baseline_names=tuple(cfg["protected_conditions"]["evaluation_baselines"]["value"]))
            record = {"model": model_name, "ppo_epoch": epoch,
                      "policy": policy_result, "baselines": baseline_result}
            model_eval_records.append(record); eval_records.append(record)
            route_records.append({"model": model_name, "ppo_epoch": epoch,
                                  "policy": policy_result["sample_path"],
                                  "policy_revisits": policy_result["sample_path_revisits"],
                                  "baselines": {k: v["sample_path"] for k, v in baseline_result.items()}})
            hop_counts.extend(policy_result["successful_hop_counts"])
            for result in baseline_result.values():
                hop_counts.extend(result["successful_hop_counts"])
            (run_dir / "evaluation.json").write_text(json.dumps(eval_records, indent=2), encoding="utf8")
            (run_dir / "routes" / "representative_routes.json").write_text(
                json.dumps(route_records, indent=2), encoding="utf8")
            return {"overall_reward": policy_result["overall_reward"],
                    "success_rate": policy_result["success_rate"],
                    "avg_hops_success": policy_result["avg_hops_success"],
                    "zero_success_epochs": policy_result["zero_success_epochs"]}

        env = QKDRoutingEnv(
            source=cfg["environment"]["source"], destination=cfg["environment"]["destination"],
            season=cfg["environment"]["season"],
            use_case=cfg["environment"]["use_case"],
            node_feature_mode=cfg["environment"]["node_feature_mode"],
            time_jitter_hours=cfg["environment"]["time_jitter_hours"],
            disabled_reward_terms=set(cfg["environment"]["disabled_reward_terms"]),
            randomize_endpoints=cfg["environment"]["randomize_endpoints"],
            fiber_outage_prob=cfg["environment"]["fiber_outage_prob"],
            qber_hard=cfg["environment"]["qber_hard"],
            reward_protection_overrides=cfg["environment"]["reward_protection_overrides"],
            max_steps=cfg["environment"]["max_steps"],
            time_of_day_hours=cfg["environment"]["time_of_day_hours"],
            dt_seconds=cfg["environment"]["dt_seconds"])
        model_resume = None
        if resume:
            rp = Path(resume)
            if rp.is_file():
                model_resume = rp if model_name.lower() in str(rp).lower() else None
            else:
                candidate = run_dir / "checkpoints" / model_name / "latest.pt"
                if candidate.exists():
                    model_resume = candidate
        result = train_model(model_name, env, model_training_cfg, run_dir,
                             dropedge_probability=cfg["model"]["dropedge_probability"],
                             resume_from=model_resume, epoch_callback=callback,
                             stop_after_ppo_epochs=stop_after_ppo_epochs)
        all_metrics[model_name] = result["history"]
    (run_dir / "training_metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf8")
    (run_dir / "evaluation.json").write_text(json.dumps(eval_records, indent=2), encoding="utf8")
    (run_dir / "routes" / "representative_routes.json").write_text(json.dumps(route_records, indent=2), encoding="utf8")
    reward_values = []
    for model, history in all_metrics.items():
        for row in history:
            if row.get("phase") == "ppo":
                reward_values.append(float(row.get("overall_reward", row.get("rollout_reward", 0.0))))
    _svg_line_plot("Training and evaluation reward", reward_values,
                   run_dir / "plots" / "reward_curves.svg", "Reward")
    _svg_hop_histogram("Successful route hop-count distribution", hop_counts,
                       run_dir / "plots" / "hop_count_distributions.svg")
    complete = True
    for model_name, history in all_metrics.items():
        ppo_rows = [r for r in history if r.get("phase") == "ppo"]
        if len(ppo_rows) < training_cfg["ppo_epochs"]:
            complete = False
    summary = {"experiment_name": cfg["experiment_name"], "git_commit": commit,
               "run_dir": str(run_dir), "status": "complete" if complete else "partial",
               "models": cfg["models"], "reward_evaluations": len(eval_records),
               "representative_route_count": len(route_records)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--resume", help="run directory or checkpoint path")
    parser.add_argument("--stop-after-ppo-epochs", type=int,
                        help="testing aid: stop after saving this many new PPO epochs")
    args = parser.parse_args()
    run_experiment(Path(args.config) if args.config else None,
                   resume=args.resume,
                   stop_after_ppo_epochs=args.stop_after_ppo_epochs)


if __name__ == "__main__":
    main()
