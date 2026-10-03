# Paired reward-component audit

Evaluated 108 matched endpoint/condition/seed scenarios per policy (5 policies; 540 episodes total). Every cumulative reward term is stored in `episode_reward_components.csv`; `policy_summary.csv` contains per-policy means. Four figures are saved in `plots/` as both PNG and PDF.

The reward decomposition uses the environment's existing `info['reward_terms']`; reward equations were not changed. Conditions without a specified hour use the checkpoint's 22:00 operating point. This is a simulator-only route audit.
