from pathlib import Path
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
paper = Path(__file__).resolve().parent
sup = paper / "supplementary"
files = {
    "osm_road_corridors.json": "data/osm_road_corridors.json",
    "physics_profile_report_idq.json": "experiments/physics_profile_report_idq.json",
    "comparison_200ep.json": "experiments/defence_monsoon_night_200ep_mumbai_kolkata/comparison.json",
    "comparison_100ep.json": "experiments/defence_monsoon_night_100ep_mumbai_kolkata/comparison.json",
    "gnn_training_metrics_200ep.json": "experiments/defence_monsoon_night_200ep_mumbai_kolkata/gnn_training_metrics.json",
    "config_200ep.yaml": "experiments/defence_monsoon_night_200ep_mumbai_kolkata/config_snapshot.yaml",
    "config_100ep.yaml": "experiments/defence_monsoon_night_100ep_mumbai_kolkata/config_snapshot.yaml",
    "adverse_conditions_comparison.json": "experiments/adverse_conditions_comparison.json",
    "adverse_statistics.json": "experiments/adverse_statistics.json",
    "inference_latency_comparison.json": "experiments/inference_latency_comparison.json",
    "local_subgraph_benchmark.json": "experiments/local_subgraph_benchmark.json",
    "optimized_inference_benchmark.json": "experiments/optimized_inference_benchmark.json",
    "fso_model_validation.json": "experiments/fso_model_validation.json",
    "paper_metrics.json": "experiments/paper_metrics.json",
    "idq_id281_profile.yaml": "hardware_profiles/idq_id281_1550_uk_ireland.yaml",
    "physics_evidence.md": "PHYSICS_EVIDENCE.md",
    "experiment_protocol.md": "EXPERIMENT_PROTOCOL.md",
    "validation_translation_plan.md": "VALIDATION_AND_TRANSLATION_PLAN.md",
    "corrected_route_comparison_20261002.json": "paper/supplementary/corrected_route_comparison_20261002.json",
    "episode_metrics.csv": "paper/supplementary/episode_metrics.csv",
    "link_quality_sensitivity_20261002.json": "paper/supplementary/link_quality_sensitivity_20261002.json",
    "current_checkpoint_pair_metrics_20261002.json": "paper/supplementary/current_checkpoint_pair_metrics_20261002.json",
    "route_comparison_gnn_config_20260928.yaml": "paper/supplementary/route_comparison_gnn_config_20260928.yaml",
    "checkpoint_gnn_latest_50ep_idq_20260928.pt": "paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt",
    "topology_source.py": "topology.py",
    "physics_source.py": "physics.py",
    "environment_source.py": "qkd_env.py",
    "gnn_model_source.py": "models.py",
    "attention_source.py": "qkd_attention.py",
    "ppo_source.py": "ppo.py",
    "evaluation_source.py": "evaluation.py",
    "baseline_source.py": "baselines.py",
    "training_source.py": "train.py",
    "runner_source.py": "run_experiment.py",
    "inference_source.py": "inference_engine.py",
    "training_seed_stability_source.py": "scripts/run_training_seed_stability.py",
    "training_seed_progress_source.py": "scripts/refresh_training_seed_progress.py",
    "training_seed_stability_progress_20261002.json": "paper/supplementary/training_seed_stability_progress_20261002.json",
    "link_quality_choice_source.py": "link_quality_choice_test.py",
    "small_graph_oracle_source.py": "scripts/run_small_graph_oracle.py",
    "pair_season_matrix_20261002.json": "paper/supplementary/pair_season_matrix_20261002.json",
    "pair_season_matrix_20261002.csv": "paper/supplementary/pair_season_matrix_20261002.csv",
    "pair_season_matrix_source.py": "scripts/run_pair_season_matrix.py",
    "pair_season_analysis_source.py": "scripts/analyze_pair_season_matrix.py",
    "archive_training_seed_source.py": "scripts/archive_training_seed_evidence.py",
    "inference_profile_source.py": "scripts/benchmark_inference_profile.py",
    "matched_ablation_source.py": "scripts/run_matched_budget_ablations.py",
    "pair_season_matrix_20261002.json": "paper/supplementary/pair_season_matrix_20261002.json",
    "pair_season_matrix_20261002.csv": "paper/supplementary/pair_season_matrix_20261002.csv",
    "inference_profile_20261002.json": "paper/supplementary/inference_profile_20261002.json",
    "seed_stability_50ep.yaml": "configs/seed_stability_50ep.yaml",
    "configuration_test.py": "tests/test_config.py",
}
for dest, src in files.items():
    source, target = root / src, sup / dest
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
shutil.copy2(root / "figures/india_road_corridor_map.html", sup / "interactive_corridor_map.html")
if not (sup / "README.md").exists():
    (sup / "README.md").write_text(
        "# Supplementary files\n\n"
        "These files preserve inputs and outputs for the research experiments. "
        "The corrected route comparison and physics validations are simulator-derived; "
        "they do not establish field performance or composable QKD security. "
        "Road routes are OpenStreetMap corridor proxies, not surveyed fiber assets. "
        "See this README and the evidence ledger for scope and limitations. "
        "OpenStreetMap attribution and licensing terms apply.\n", encoding="utf-8")
out = root / "QKD_IEEE_Overleaf_Project.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in paper.rglob("*"):
        if (path.is_file()
                and path.name not in {"package_overleaf.py", "build_artifacts.py"}
                and path.suffix.lower() not in {".aux", ".log", ".out", ".blg", ".fls"}):
            archive.write(path, path.relative_to(paper))
print(f"Created {out} ({out.stat().st_size} bytes)")
