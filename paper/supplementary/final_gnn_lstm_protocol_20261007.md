# Matched GNN-PPO versus LSTM-PPO evaluation protocol

## Question and scope

This study compares the graph message-passing encoder with the repository's LSTM node-sequence encoder under the same routing environment, actor-critic head, PPO objective, candidate-edge features, and physical-feasibility mask. It is a simulator-level architecture comparison. It does not test field performance, delivered secret-key throughput, multi-request service, or formal QKD security.

The LSTM consumes node-feature rows in the topology's stable node order and does not receive graph edges. Its recurrent state is internal to one policy forward pass; it is not persistent route-history memory across decisions. This distinction is part of the comparator definition and limits interpretation of any difference.

## Training and checkpoint selection

- Training seeds: `20261002`, `20261003`, `20261004`, `20261005`, and `20261006` for each architecture.
- Both architectures use the same versioned configuration, 15 behavior-cloning epochs, 50 PPO epochs, and the same optimization and reward settings.
- Test endpoint pairs and their reversals are excluded from behavior-cloning demonstrations. PPO uses the fixed Delhi-to-Chennai training pair.
- Checkpoint selection uses success rate on the fixed monsoon validation suite only. The final held-out scenarios are not used for selection or tuning.
- GNN-PPO checkpoints are the five frozen checkpoints in the completed strict evaluation report. LSTM-PPO checkpoints are trained independently with the matched configuration.

## Held-out evaluation

The test matrix contains six ordered city pairs, six season/time conditions (normal, summer, and winter at 02:00 and 22:00), and three shared environment replicates: 108 scenarios per checkpoint. Both policies use identical scenario identifiers and environment seeds, with seed formula `700000 + 100*pair_index + 10*condition_index + replicate`. Five checkpoints per architecture yield 540 episodes per architecture.

Route completion is the primary endpoint. Successful-route hop count, episode reward, invalid actions, fallback actions, and no-valid-action states are secondary diagnostics. Report per-checkpoint outcomes, the mean across five checkpoints, and a nested bootstrap interval that resamples training checkpoints outside and scenario clusters inside. The primary paired effect is LSTM-PPO completion minus GNN-PPO completion on identical scenario seeds, with the same nested resampling structure.

## Inference measurement

Measure 1,000 deterministic actor calls per checkpoint after 100 warm-up calls on the same held-out observation. Use CPU, four Torch threads, and disabled encoder caching. Report mean and p50/p95/p99 latency across measured calls, then summarize checkpoint-to-checkpoint variation. These are single-host policy-call measurements; environment simulation and deployment overhead are excluded.

## Exact rollout memoization note (2026-10-07)

During held-out evaluation, repeated LSTM encoder computation was memoized by the complete node-feature matrix and encoder parameter versions. For a fixed topology and destination, that matrix is constant throughout an episode; the cache is reused only on an exact input match, while candidate scoring, critic evaluation, and action masking are recomputed from each current observation. This changes computation time, not the deterministic policy output. A numerical equivalence check confirmed matching logits and values on an unchanged observation and after a route-context change with the same node-feature matrix. Episodes journaled before this optimization are retained; the later cached episodes use the same checkpoint, scenario IDs, environment seeds, and forward outputs. Encoder caching is explicitly disabled for the reported latency measurements above.

## Reporting constraints

Do not describe an architecture as superior from cumulative shaped reward alone. Interpret route completion alongside successful-route length, seed variability, invalid-action/fallback counts, and inference cost. Any comparison is conditional on this topology, feature ordering, training budget, checkpoint-selection procedure, and scenario distribution.
