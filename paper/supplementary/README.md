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
- `link_quality_sensitivity_20261002.json`: current-checkpoint twin-link comparison plus controlled proxy-rate ratio, QBER threshold, hard outage-mask, and 0.01x-100x turbulence-Cn2 sweeps. In the turbulence sweep, the policy selected FSO in every sample even when the 100x Cn2 input reduced median proxy from 0.202 to 0.183; action probability shifted only 0.749 to 0.740. The FSO choice remained selected in 32/32 trials. This is a policy-input diagnostic with fixed route context, not calibrated weather or route-level performance. Pointing/background multipliers and burst outage persistence are not available as independent simulator controls.
- `current_checkpoint_pair_metrics_20261002.json`: one route sample for each of 42 ordered city pairs under the monsoon-night scenario, plus local-inference latency. This is not a held-out endpoint test and has no matched baseline.
- `test_results_20261002.json`: full repository regression suite result after these implementation changes (41 passed, 0 failed).
- `training_seed_stability_20261002.json` will contain the aggregate five-seed report when the active full-budget campaign finishes. The fixed protocol and per-seed configs are defined by `configs/seed_stability_50ep.yaml` and `scripts/run_training_seed_stability.py`; until the aggregate report exists, P0-6 remains in progress.
- `training_seed_stability_progress_20261002.json` inventories completed runs and the newest run without a final summary. Refresh it with `python scripts/refresh_training_seed_progress.py`; it is progress evidence only and must not be used as an aggregate stability result.
- Once the five-seed campaign completes, archive and validate its reports, resolved configs, and checkpoints with `python scripts/archive_training_seed_evidence.py --campaign-dir experiments/runs/<completed-campaign>`; the archiver refuses partial or non-50-epoch campaigns.
- `small_graph_oracle_20261002.json`: exhaustive simple-path reward oracle on five generated seven-node graphs, four channel seeds each, with static episode link states and 5% per-edge fiber outages. All 20 cases had a feasible oracle; the seed-20261002 policy succeeded in 18/20 and exactly matched oracle reward in 11/20 overall (11/18 successful policy episodes). Mean reward regret was 0.663 among successful policy episodes and 2.209 including failures. This is a toy static-channel result, not a full-network optimality claim.

Reproduce physics checks with `python scripts/validate_physics_analytical.py` and `python scripts/validate_fso_model.py`. Reproduce controlled link sensitivity with `python link_quality_choice_test.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --samples-per-condition 32 --output <path>`. Reproduce the ordered-pair run with `python scripts/paper_metrics.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --output <path>`.

Reproduce the exact toy-graph oracle with `python scripts/run_small_graph_oracle.py --checkpoint <completed-GNN-checkpoint> --output <path>`. The enumerated oracle maximizes the environment's cumulative reward over all feasible simple paths under a frozen per-episode link state; the policy is evaluated on that same state.

Reproduce the paired cross-pair/season/time matrix with `python scripts/run_pair_season_matrix.py --checkpoint <completed-GNN-checkpoint> --seeds-per-condition 3 --output <path>`. This is a transfer diagnostic; the BC stage randomized endpoints, so it is not a strict end-to-end endpoint holdout.

In the controlled rate-ratio sweep, the checkpoint's FSO action probability rose from about 0.311 at a 0.25x FSO/fiber proxy-rate ratio to 0.638 at 4x. At QBER 0.111 the FSO candidate was masked and assigned zero probability. Under feasible QBER values, probability shifted only slightly (about 0.473 to 0.472), so the primary observed sensitivity was to the proxy rate and hard feasibility mask.

The controlled turbulence sweep increases Cn2 while holding the paired-link route context and clear-state condition fixed. Even after a 100x multiplier, the checkpoint still selected FSO in every sampled twin-link decision; treat this as evidence of weak preference change under this particular feature scaling, not a universal statement about FSO routing.

## Earlier archived evidence

`preliminary_comparison.csv`, `comparison_100ep.json`, `comparison_200ep.json`, and the older training metrics describe earlier runs and must not be merged with the paired-seed 2026-10-02 comparison. The manuscript may still cite an archived run only when it labels its scope and provenance. See `experiment_protocol.md`, `physics_evidence.md`, and `validation_translation_plan.md` for additional caveats.

OpenStreetMap attribution and licensing terms apply to the included corridor geometry.
