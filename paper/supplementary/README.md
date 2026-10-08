# Supplementary files

This supplement supports the manuscript's matched GNN-PPO versus LSTM-PPO study. It provides the frozen evaluation protocol, five-seed training manifest, held-out episode-level report, selected-seed reward audit, and the scripts used to generate the comparison figures. The LSTM processes node-feature rows in a fixed topology order, uses no graph-edge input, and resets hidden state for each policy call.

Across the five held-out checkpoints, GNN-PPO completed 249/540 routes (46.1%; nested 95% interval 31.3%-58.5%) and LSTM-PPO completed 403/540 (74.6%; 58.0%-87.6%). The paired LSTM-minus-GNN difference was 28.5 percentage points (20.6%-37.2%). The selected-seed reward audit evaluates 108 shared scenarios: GNN-PPO completion was 46.3% with 92.56 mean successful-route hops; LSTM-PPO completion was 84.3% with 62.22 hops. Reward-component totals are descriptive because trajectory lengths differ.

All results are simulator outcomes for the stated topology and scenario matrix. The corridor map is an OpenStreetMap road proxy rather than verified fiber infrastructure; FSO and key-pool settings are not calibrated field measurements. Normalized rate is not delivered secret-key throughput, and the optional finite-key engineering model is not a composable security proof.

The report's `source_sha256` entries identify code associated with the evaluation. The reproducibility bundle includes the resolved seed manifest and five per-seed configurations, frozen protocol, episode ledger, evaluation/plotting scripts, figure-generation source, model source, and selected figure files. Earlier routing-baseline experiments remain in the local research archive and are not part of this manuscript comparison or publication bundle.
