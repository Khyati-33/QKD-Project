# Overleaf manuscript package

Upload `QKD_IEEE_Overleaf_Project.zip` to Overleaf, or upload this directory's contents. The main document is `main.tex` and uses the IEEEtran journal class in two-column layout. The abstract and final simulator results are now included. Replace the draft author block and complete funding and acknowledgment details before submission.

The bibliography contains 30 entries. Figures are generated from the project topology and archived model report. The `supplementary` directory carries source geometry, configurations, archived outputs, the evidence ledger, experiment protocol, and relevant code snapshots. `build_artifacts.py` generates figures from project-level data and model outputs.

The manuscript uses original prose and figures. Its organization separates problem formulation, simulator and experimental setup, routing policy, paired route and small-graph oracle results, and limitations.

The manuscript includes the frozen matched experiment protocol, the five-seed strict held-out comparison, a selected-checkpoint reward-component audit, a small-graph reward oracle, and controlled policy-input counterfactuals. The strict evaluation report is `experiments/final_holdout_20261007.json`; the reward audit is `experiments/final_reward_audit_20261007/audit_summary.json`. The current results do not show GNN-PPO outperforming matched MLP-PPO or constrained shortest-path baselines. All channel quantities remain simulator outputs; there is no field validation or formal security proof. Check author details, OpenStreetMap attribution, and the final compiled layout before submission.

The manuscript reports two distinct reward audits: the archived 2026-10-03 current-checkpoint diagnostic (repeated seeds 0, 1, and 2) and the final selected-seed comparison against MLP-PPO and BFS-hop. They use different checkpoints and settings and are not pooled. The latter's episode-level data and plot are under `experiments/final_reward_audit_20261007/` and `figures/final_evaluation/`.

The manuscript includes all files from `figures/`: implementation-backed diagrams and final evaluation plots appear in the main text, and the remaining archive assets appear in the supplementary figure appendix. Captions identify conceptual diagrams, model-generated outputs, exploratory comparisons, and the explicitly illustrative training-curve image so they are not confused with final held-out evidence. Five final analysis figures (four held-out/stress plots and one reward-component plot) appear in the main results section.
