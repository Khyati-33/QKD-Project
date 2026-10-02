# Supplementary files and evidence status

This directory contains both archived manuscript evidence and the reproducibility package for the 2026-10-02 roadmap update. Road geometry is sourced from OpenStreetMap and is a routing corridor proxy, not a surveyed fiber asset. FSO visibility, weather, and optical parameters are scenario assumptions. No file here constitutes field validation or a composable QKD security proof.

## New route-level comparison

- `corrected_route_comparison_20261002.json` summarizes the paired-seed comparison for one archived GNN checkpoint and four route baselines.
- `episode_metrics.csv` preserves the per-episode routes and metrics. All methods received the same four-season, 16-seed environment set. No invalid action proposals or environment fallbacks occurred. `no_valid_action_states` records terminal conditions where no feasible move remained.
- `route_comparison_gnn_config_20260928.yaml` is the resolved configuration used for the archived 50-epoch GNN checkpoint.
- `checkpoint_gnn_latest_50ep_idq_20260928.pt` preserves that exact checkpoint; its SHA-256 is in the comparison report.

Results: GNN-PPO and BFS-hop each completed 16/16 episodes with 28.25 mean hops. Distance-weighted Dijkstra completed 4/16 and averaged 233 hops on its successful routes. Random and greedy Max-SKR completed 0/16. The 95% Wilson intervals are broad at this sample size. These results do not show a GNN advantage over BFS and do not measure demand blocking, QKP load balance, or network service capacity.

Reproduce with `python scripts/run_corrected_route_comparison.py --seeds-per-season 4`. The source and checkpoint hashes, git commit, shared-seed declaration, and worktree state are recorded in the JSON summary.

## Physics and policy diagnostics

- `physics_analytical_validation.json`: independent transcription of the configured fiber QBER/rate equations over 201 distances, configured FSO availability anchor checks, and a 20,000-step temporal-correlation sample. This is internal consistency only.
- `fso_model_validation.json`: 2,000 availability samples at each of 16 season/distance conditions, with errors and approximate binomial uncertainty. Targets are model configuration values, not observed weather.
- `link_quality_sensitivity_20261002.json`: current-checkpoint twin-link comparison plus controlled proxy-rate ratio, feasible QBER margin, and outage-mask sweeps. The inputs are synthetic policy counterfactuals.
- `current_checkpoint_pair_metrics_20261002.json`: one route sample for each of 42 ordered city pairs under the monsoon-night scenario, plus local-inference latency. This is not a held-out endpoint test and has no matched baseline.
- `test_results_20261002.json`: full repository regression suite result after these implementation changes (40 passed, 0 failed).

Reproduce physics checks with `python scripts/validate_physics_analytical.py` and `python scripts/validate_fso_model.py`. Reproduce controlled link sensitivity with `python link_quality_choice_test.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --samples-per-condition 32 --output <path>`. Reproduce the ordered-pair run with `python scripts/paper_metrics.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --output <path>`.

In the controlled rate-ratio sweep, the checkpoint's FSO action probability rose from about 0.311 at a 0.25x FSO/fiber proxy-rate ratio to 0.638 at 4x. At QBER 0.111 the FSO candidate was masked and assigned zero probability. Under feasible QBER values, probability shifted only slightly (about 0.473 to 0.472), so the primary observed sensitivity was to the proxy rate and hard feasibility mask.

## Earlier archived evidence

`preliminary_comparison.csv`, `comparison_100ep.json`, `comparison_200ep.json`, and the older training metrics describe earlier runs and must not be merged with the paired-seed 2026-10-02 comparison. The manuscript may still cite an archived run only when it labels its scope and provenance. See `experiment_protocol.md`, `physics_evidence.md`, and `validation_translation_plan.md` for additional caveats.

OpenStreetMap attribution and licensing terms apply to the included corridor geometry.
