# Overleaf manuscript package

Upload `QKD_IEEE_Overleaf_Project.zip` to Overleaf, or upload this directory's contents. The main document is `main.tex` and uses the IEEEtran journal class in two-column layout. The abstract summarizes the current limited evidence. Replace the draft author block and complete funding and acknowledgment details before submission.

The bibliography contains 30 entries. Figures are generated from the project topology and archived model report. The `supplementary` directory carries source geometry, configurations, archived outputs, the evidence ledger, experiment protocol, and relevant code snapshots. `build_artifacts.py` generates figures from project-level data and model outputs.

The manuscript uses original prose and figures. Its organization separates problem formulation, simulator and experimental setup, routing policy, paired route and small-graph oracle results, and limitations.

The manuscript now incorporates the completed six-pair, six-condition transfer matrix and exact small-graph reward oracle. It reports where the agent outperforms the distance-weighted and greedy link-cost baselines, while making clear that BFS had higher success and shorter routes. The paired analysis uses a single trained checkpoint and randomized-endpoint behavior cloning, so it is not a strict endpoint holdout or a training-seed estimate. All channel quantities remain simulator outputs; there is no field validation or formal security proof. Check citations, author details, OSM attribution, and compilation in Overleaf before submission.

The current draft also reports a separate current-checkpoint reward-component audit across the same endpoint-condition grid using seeds 0, 1, and 2 and the configured 400-step horizon. This audit is labeled separately from the archived transfer matrix because checkpoint, seed schedule, and horizon differ. The four reward/outcome plots and episode-level component data are packaged under `supplementary/reward_component_audit_20261003/`; repeated seed values make the rows a descriptive diagnostic rather than independent trials.
