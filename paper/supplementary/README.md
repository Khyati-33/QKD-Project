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
- `pair_season_matrix_20261002.json` and `.csv`: completed paired transfer diagnostic across six ordered endpoint pairs, six season/time conditions, and three shared environment seeds (108 episodes per method). GNN-PPO succeeded in 99/108 (91.7%, Wilson 95% CI 84.9%-95.6%); BFS-hop succeeded in 108/108 (100%, 96.6%-100%). Successful routes averaged 31.94 hops for GNN-PPO and 15.40 for BFS-hop. The tested GNN therefore did not outperform BFS on success or hop count. These simulator-derived pairs were excluded from PPO's fixed-endpoint rollouts, but the BC stage randomized endpoints, so this is not a strict end-to-end endpoint holdout.
- Paired comparison in `pair_season_matrix_20261002.json`: BFS-only success occurred in 9 episodes and GNN-only success in none (exact two-sided McNemar p=0.0039). Among the 99 episodes where both succeeded, GNN-PPO used 16.00 more hops on average (median difference 6); a 20,000-replicate endpoint-pair cluster bootstrap 95% interval for the mean difference was 6.48 to 25.97 hops. The cluster bootstrap resamples the six ordered endpoint pairs, so its interval is appropriately interpreted as coarse with only six clusters.

Per-condition results (18 episodes per policy):

| Season / hour | GNN-PPO success | GNN successful mean hops | BFS-hop success | BFS successful mean hops |
|---|---:|---:|---:|---:|
| Normal / 02:00 | 14/18 (77.8%) | 48.79 | 18/18 (100%) | 15.50 |
| Normal / 22:00 | 15/18 (83.3%) | 52.53 | 18/18 (100%) | 15.44 |
| Summer / 22:00 | 16/18 (88.9%) | 40.75 | 18/18 (100%) | 15.28 |
| Winter / 22:00 | 18/18 (100%) | 21.83 | 18/18 (100%) | 15.39 |
| Monsoon / 02:00 | 18/18 (100%) | 17.78 | 18/18 (100%) | 15.28 |
| Monsoon / 22:00 | 18/18 (100%) | 18.11 | 18/18 (100%) | 15.50 |
- `inference_profile_20261002.json`: 840 deterministic single-decision calls per CPU thread setting over 42 ordered pairs, with encoder memoization disabled. At 8 threads, p50/p95/p99 were 12.76/16.89/22.58 ms; process RSS after the run was 348.3 MiB and model parameter storage was 0.53 MiB. The measurement shared the host with the 8-thread seed-stability training job, so treat latency as a concurrent-load result. It is one host/topology and does not measure graph-size scaling.
- `matched_budget_ablations_20261002.json` will be written by the five-condition matched-budget ablation campaign when its runs finish. It compares baseline, geographic node features, no SKR reward term, no DropEdge, and 32-wide GNN using the same 50 PPO epochs, evaluation protocol, and training seed. Results are single-seed sensitivity evidence, not between-seed estimates.
- `matched_budget_ablations_progress_20261002.json` is a checkpoint-derived progress record, not a result report. Refresh it with `python scripts/refresh_ablation_progress.py` while the training campaign runs.

Reproduce physics checks with `python scripts/validate_physics_analytical.py` and `python scripts/validate_fso_model.py`. Reproduce controlled link sensitivity with `python link_quality_choice_test.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --samples-per-condition 32 --output <path>`. Reproduce the ordered-pair run with `python scripts/paper_metrics.py --checkpoint paper/supplementary/checkpoint_gnn_latest_50ep_idq_20260928.pt --output <path>`.

Reproduce the exact toy-graph oracle with `python scripts/run_small_graph_oracle.py --checkpoint <completed-GNN-checkpoint> --output <path>`. The enumerated oracle maximizes the environment's cumulative reward over all feasible simple paths under a frozen per-episode link state; the policy is evaluated on that same state.

Reproduce the paired cross-pair/season/time matrix with `python scripts/run_pair_season_matrix.py --checkpoint <completed-GNN-checkpoint> --seeds-per-condition 3 --output <path>`. This is a transfer diagnostic; the BC stage randomized endpoints, so it is not a strict end-to-end endpoint holdout.

Recompute the paired success and hop analysis from a completed matrix with `python scripts/analyze_pair_season_matrix.py <matrix-report.json>`. The exact McNemar test uses shared episodes; the hop interval uses an endpoint-pair cluster bootstrap.

Reproduce the CPU inference profile with `python scripts/benchmark_inference_profile.py --checkpoint <completed-GNN-checkpoint> --repeats-per-pair 20`. The script reports per-decision p50/p95/p99 latency and process/model memory for the selected CPU thread counts. Record concurrent CPU jobs when interpreting the result.

Run the matched-budget architecture, feature, and reward sensitivity study with `python scripts/run_matched_budget_ablations.py`. It sequentially trains five configurations at the 50-epoch budget and writes progress after each variant.

Refresh the active matched-ablation checkpoint state with `python scripts/refresh_ablation_progress.py`.

In the controlled rate-ratio sweep, the checkpoint's FSO action probability rose from about 0.311 at a 0.25x FSO/fiber proxy-rate ratio to 0.638 at 4x. At QBER 0.111 the FSO candidate was masked and assigned zero probability. Under feasible QBER values, probability shifted only slightly (about 0.473 to 0.472), so the primary observed sensitivity was to the proxy rate and hard feasibility mask.

The controlled turbulence sweep increases Cn2 while holding the paired-link route context and clear-state condition fixed. Even after a 100x multiplier, the checkpoint still selected FSO in every sampled twin-link decision; treat this as evidence of weak preference change under this particular feature scaling, not a universal statement about FSO routing.

## Earlier archived evidence

`preliminary_comparison.csv`, `comparison_100ep.json`, `comparison_200ep.json`, and the older training metrics describe earlier runs and must not be merged with the paired-seed 2026-10-02 comparison. The manuscript may still cite an archived run only when it labels its scope and provenance. See `experiment_protocol.md`, `physics_evidence.md`, and `validation_translation_plan.md` for additional caveats.

OpenStreetMap attribution and licensing terms apply to the included corridor geometry.

## Current-checkpoint reward-component audit (2026-10-03)

`reward_component_audit_20261003/` contains a second, explicitly requested audit using the current GNN checkpoint (SHA-256 `154c8a81e4335b2bd609999155d785b5f1eaeedf8f5cd90e62023d795a368b27`), the same six ordered city pairs and six season/time labels, and seeds 0, 1, and 2 reused in every pair-condition cell. It uses the configured 400-step limit, no time jitter, and the current defense reward-protection overrides. Summer and winter use 22:00 because those labels do not specify an hour. This differs from the archived `pair_season_matrix_20261002` transfer result (different checkpoint, seed schedule, and 144-step cap), so do not combine the rates as a direct checkpoint comparison.

Each of five policies has 108 scenario rows (540 episodes total). GNN-PPO completed 102/108 (94.4%) with 15.78 mean hops on successful episodes; BFS-hop completed 108/108 with 15.83 mean successful hops; Dijkstra-km completed 48/108 and averaged 114.25 successful hops; Random and Max-SKR completed none. GNN-PPO tied BFS on 92/102 jointly successful rows and used one additional hop on the other ten. Its six failures all occurred for Kolkata-to-Delhi with seed 1 across the six conditions, and all incurred the key-pool-depletion penalty. Because the same three seed values are repeated across conditions and endpoint pairs, the rows are descriptive repeated-seed observations; do not interpret episode-level Wilson intervals or an unclustered paired test as independent-trial inference.

The episode-level `episode_reward_components.csv` records every reward term, `policy_summary.csv` contains per-policy means, and `audit_summary.json` records the protocol and scope. The `plots/` directory has four final figures, each in PNG and PDF format: utility components, protection/failure costs, GNN-minus-BFS component deltas, and completion/hop outcomes. Reward equations were not changed. Weighted-SP is not included because there is no such implementation in this repository. This remains a simulator-only single-route diagnostic and does not measure demand blocking, network-wide key-pool load balancing, service throughput, or field performance.
