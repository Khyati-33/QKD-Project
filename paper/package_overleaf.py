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
    "matched_budget_ablations_progress_20261002.json": "paper/supplementary/matched_budget_ablations_progress_20261002.json",
    "link_quality_choice_source.py": "link_quality_choice_test.py",
    "small_graph_oracle_source.py": "scripts/run_small_graph_oracle.py",
    "pair_season_matrix_20261002.json": "paper/supplementary/pair_season_matrix_20261002.json",
    "pair_season_matrix_20261002.csv": "paper/supplementary/pair_season_matrix_20261002.csv",
    "pair_season_matrix_source.py": "scripts/run_pair_season_matrix.py",
    "reward_component_audit_source.py": "scripts/run_reward_component_audit.py",
    "pair_season_analysis_source.py": "scripts/analyze_pair_season_matrix.py",
    "archive_training_seed_source.py": "scripts/archive_training_seed_evidence.py",
    "inference_profile_source.py": "scripts/benchmark_inference_profile.py",
    "matched_ablation_source.py": "scripts/run_matched_budget_ablations.py",
    "refresh_ablation_progress_source.py": "scripts/refresh_ablation_progress.py",
    "pair_season_matrix_20261002.json": "paper/supplementary/pair_season_matrix_20261002.json",
    "pair_season_matrix_20261002.csv": "paper/supplementary/pair_season_matrix_20261002.csv",
    "inference_profile_20261002.json": "paper/supplementary/inference_profile_20261002.json",
    "seed_stability_50ep.yaml": "configs/seed_stability_50ep.yaml",
    "configuration_test.py": "tests/test_config.py",
    "final_gnn_lstm_holdout_20261007.json": "experiments/final_gnn_lstm_holdout_20261007.json",
    "final_gnn_lstm_reward_audit_20261007.json": "experiments/final_gnn_lstm_reward_audit_20261007/audit_summary.json",
    "final_gnn_lstm_reward_episode_ledger.csv": "experiments/final_gnn_lstm_reward_audit_20261007/episode_reward_components.csv",
    "lstm_seed_runs_20261007.json": "experiments/final_holdout_lstm_20261007/seed_runs.json",
    "lstm_seed_20261002.yaml": "experiments/final_holdout_lstm_20261007/configs/seed_20261002.yaml",
    "lstm_seed_20261003.yaml": "experiments/final_holdout_lstm_20261007/configs/seed_20261003.yaml",
    "lstm_seed_20261004.yaml": "experiments/final_holdout_lstm_20261007/configs/seed_20261004.yaml",
    "lstm_seed_20261005.yaml": "experiments/final_holdout_lstm_20261007/configs/seed_20261005.yaml",
    "lstm_seed_20261006.yaml": "experiments/final_holdout_lstm_20261007/configs/seed_20261006.yaml",
    "matched_gnn_lstm_protocol_20261007.md": "paper/supplementary/final_gnn_lstm_protocol_20261007.md",
    "matched_campaign_source.py": "scripts/run_final_model_campaign.py",
    "lstm_heldout_evaluation_source.py": "scripts/run_lstm_heldout_evaluation.py",
    "gnn_lstm_comparison_plot_source.py": "scripts/plot_gnn_lstm_comparison.py",
    "gnn_lstm_reward_audit_source.py": "scripts/run_gnn_lstm_reward_audit.py",
    "figure_generation_source.py": "paper/generate_figures.py",
}
active_supplementary_files = {
    "final_gnn_lstm_holdout_20261007.json",
    "final_gnn_lstm_reward_audit_20261007.json",
    "final_gnn_lstm_reward_episode_ledger.csv",
    "lstm_seed_runs_20261007.json",
    "lstm_seed_20261002.yaml",
    "lstm_seed_20261003.yaml",
    "lstm_seed_20261004.yaml",
    "lstm_seed_20261005.yaml",
    "lstm_seed_20261006.yaml",
    "matched_gnn_lstm_protocol_20261007.md",
    "gnn_model_source.py",
    "topology_source.py",
    "physics_source.py",
    "environment_source.py",
    "ppo_source.py",
    "evaluation_source.py",
    "training_source.py",
    "runner_source.py",
    "inference_source.py",
    "matched_campaign_source.py",
    "lstm_heldout_evaluation_source.py",
    "gnn_lstm_comparison_plot_source.py",
    "gnn_lstm_reward_audit_source.py",
    "figure_generation_source.py",
}
for dest, src in files.items():
    if dest not in active_supplementary_files:
        continue
    source, target = root / src, sup / dest
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
shutil.copy2(root / "figures/india_road_corridor_map.html", sup / "interactive_corridor_map.html")
out = root / "QKD_IEEE_Overleaf_Project.zip"
publication_files = {
    "README.md", "main.tex", "references.bib", "main.pdf", "QKD_IEEE_Paper.pdf",
    "supplementary/README.md",
    "supplementary/final_gnn_lstm_protocol_20261007.md",
    "supplementary/archive_figures.pdf",
    "supplementary/archive_figures.tex",
    "supplementary/final_gnn_lstm_holdout_20261007.json",
    "supplementary/final_gnn_lstm_reward_audit_20261007.json",
    "supplementary/final_gnn_lstm_reward_episode_ledger.csv",
    "supplementary/lstm_seed_runs_20261007.json",
    "supplementary/lstm_seed_20261002.yaml",
    "supplementary/lstm_seed_20261003.yaml",
    "supplementary/lstm_seed_20261004.yaml",
    "supplementary/lstm_seed_20261005.yaml",
    "supplementary/lstm_seed_20261006.yaml",
    "supplementary/gnn_model_source.py",
    "supplementary/topology_source.py",
    "supplementary/physics_source.py",
    "supplementary/environment_source.py",
    "supplementary/ppo_source.py",
    "supplementary/evaluation_source.py",
    "supplementary/training_source.py",
    "supplementary/runner_source.py",
    "supplementary/inference_source.py",
    "supplementary/matched_campaign_source.py",
    "supplementary/lstm_heldout_evaluation_source.py",
    "supplementary/gnn_lstm_comparison_plot_source.py",
    "supplementary/gnn_lstm_reward_audit_source.py",
    "supplementary/figure_generation_source.py",
    "figures/physical_state_pipeline.pdf",
    "figures/corridor_map.png",
    "figures/matched_architecture.pdf",
}
publication_files.update(
    f"figures/final_evaluation/{name}"
    for name in (
        "01_gnn_lstm_primary.pdf", "01_gnn_lstm_primary.png",
        "02_gnn_lstm_by_condition.pdf", "02_gnn_lstm_by_condition.png",
        "03_gnn_lstm_seed_variability.pdf", "03_gnn_lstm_seed_variability.png",
        "04_gnn_lstm_latency.pdf", "04_gnn_lstm_latency.png",
        "05_gnn_lstm_reward_components.pdf", "05_gnn_lstm_reward_components.png",
    )
)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in paper.rglob("*"):
        relative = path.relative_to(paper).as_posix()
        if path.is_file() and relative in publication_files:
            archive.write(path, path.relative_to(paper))
print(f"Created {out} ({out.stat().st_size} bytes)")
