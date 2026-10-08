# Manuscript package

`main.tex` is the IEEEtran manuscript. Its only empirical architecture comparison is matched GNN-PPO versus LSTM-PPO. The four primary held-out graphs are generated from `experiments/final_gnn_lstm_holdout_20261007.json`; the reward-component audit is a separate fifth diagnostic figure. Figures are placed beside the manuscript sections that interpret them, with barriers preventing float drift across sections.

The publication bundle contains the matched protocol, final comparison outputs, source needed to reproduce the study, manuscript figures, and the supplementary figure archive. Earlier GNN-versus-route-baseline and GNN-versus-MLP evidence is retained in the local research archive only and excluded from both the manuscript and the Overleaf bundle.

The results are simulator-only. Corridor geometry is an OpenStreetMap proxy; channel and key-pool settings are not calibrated field measurements. Normalized rate is not delivered secret-key throughput. Replace the draft author block and complete funding and acknowledgment details before submission.
