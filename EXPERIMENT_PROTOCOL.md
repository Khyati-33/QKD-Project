# Experiment protocol

## Protected scientific conditions

The runner checks `protected_conditions` in every YAML file. Each protected
setting is represented by a `value` and a neighboring `override` flag. Changing
one requires setting that setting's own `override: true`; changing a value
without that flag raises an error before training starts.

- Default endpoints: Delhi to Chennai.
- `QBER_HARD = 0.11`.
- Fiber outage probability: `0.01`.
- Seasons: normal, summer, winter, and monsoon.
- Evaluation seeds per season: 4.
- Trust-chain behavior: accumulated hop QBERs reset when passing through a
  `full_tn` node.
- Reward-hacking protections: fixed step cost, revisit and link-switch
  penalties, km-distance backtracking penalty, security penalty, and
  key-pool-depletion penalty remain active.
- Evaluation baselines: Random, Dijkstra-km, BFS-hop, and Max-SKR.
- `randomize_endpoints` defaults to false and stays off for the baseline family.

## Freely adjustable experimental variables

PPO rollout length and minibatch size, model-specific learning rates (including
a GNN-specific learning rate), entropy
coefficient and schedule, critic warmup, and `disabled_reward_terms` from the
supported set are experimental variables. Architecture changes must be run as
controlled experiments and recorded in the resolved run configuration.
The defence profile follows the notebook's defence reward priorities. Changes
to fixed protection magnitudes use an explicit protected-condition override in
`configs/defence_monsoon_night.yaml`.

## Training implementation notes

- The GNN has three message-passing layers (the configured maximum), one shared
  actor/critic trunk, and one disjoint-graph message-passing call per PPO
  minibatch. Dynamic QBER/SKR-invalid links contribute zero messages in both
  directions; the action head retains its hard `-inf` mask.
- DropEdge is optional and symmetric. The defence config uses 0.05; this
  affects representation regularization only and never weakens the physics mask.
- Relative node features replace geographic coordinates with destination-
  relative Dijkstra distance and normalized degree, plus node type and city
  flag, removing absolute coordinates and IDs from the model input.
- Reward normalization, clipped value loss, actor/critic learning-rate groups,
  gradient clipping, and GAE lambda are explicit configuration fields. The
  running normalizer state is saved with checkpoints for exact resume.
- The environment samples independent fiber outages and time-varying FSO
  atmospheric conditions at every simulated step. FSO outage draws are shared
  across the adjacent segments in one detour corridor, preserving each segment's
  marginal viability while modeling correlated weather. The adverse-condition config adds a seeded ±1 hour
  start-time jitter around 22:00 in monsoon; evaluation spans all four seasons
  with four seeds per season.
- CPU intra-op threads are configurable. PPO samples are generated online, so
  offline `DataLoader` workers and pinned memory would add overhead. Fixed-size
  padded observations and disjoint graph batching address relevant batching
  costs instead.
- Quantization-aware training, shared-trunk gradient detachment, and Polyak
  target networks are not enabled: they are deployment or off-policy methods
  that would change PPO's objective. Profile checkpoints before adding
  compilation or quantization as a separate deployment experiment.

## Required checks for every run

Headline reward alone is never sufficient to judge a run. Check route efficiency
against the computed default 23-hop route, loops and revisits in representative
paths, zero-success epochs, seed variation, and QBER/SKR validity every time.
Report reward medians with success rate, successful route lengths, and sample
paths. Do not treat `success_rate = 1.0` alone as evidence of a good policy.

Run directories must contain the resolved configuration, git commit, raw
per-epoch metrics, evaluation summaries, checkpoints by reward and success rate
plus latest, representative sample routes, and reward/hop-count plots.

## Smoke-test scope

Use `configs/smoke_test.yaml` for short end-to-end runs and resume checks. Do not
start the 25- or 100-epoch research campaigns as part of foundation validation.
Keep randomized endpoints off and draw no model-comparison conclusions from a
smoke run.
