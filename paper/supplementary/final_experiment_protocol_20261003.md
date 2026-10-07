# Frozen completion protocol (2026-10-03)

This protocol is fixed before the completion experiments are inspected. Any deviation must be logged with the run and reported; test results must not be used for model, reward, or baseline-weight selection.

## Data partitions

- **Training endpoints:** PPO uses Delhi-to-Chennai only. Behavior-cloning source/destination pairs are sampled from ordered city pairs after excluding the six test pairs below. Reject pairs that cross the held-out-pair set in either direction.
- **Training environment:** monsoon, 22:00 nominal time with the configured training jitter, fiber outage probability 0.01, and QBER threshold 0.11.
- **Validation:** Delhi-to-Chennai, monsoon, fixed 22:00, environment seeds 11001-11004. Evaluate deterministically at PPO epochs 10, 20, 30, 40, and 50. Validation can select the checkpoint and tune only the predeclared weighted-SP grid.
- **Held-out endpoint pairs:** Delhi-to-Mumbai, Mumbai-to-Chennai, Kolkata-to-Delhi, Bangalore-to-Hyderabad, Jaipur-to-Kolkata, and Hyderabad-to-Mumbai. These ordered pairs and their reversals are excluded from PPO rollouts and BC demonstrations.
- **Held-out environments:** normal, summer, and winter, at 02:00 and 22:00. None is used for training or validation.
- **Test scenarios:** the Cartesian product of the six ordered pairs, six held-out season/time conditions, and environment replicate indices 0, 1, and 2 (108 paired scenarios). Assign a unique integer seed to each scenario using `700000 + 100*pair_index + 10*condition_index + replicate_index`; every policy receives the same scenario ID and seed.

## Learned models and training

- Train five independent GNN-PPO policies with training seeds 20261002-20261006.
- Train five MLP-PPO policies with the same five seeds, data partitions, reward, feasibility mask, behavior-cloning budget, PPO budget, optimizer settings, and validation procedure. The MLP encoder is pointwise over node features and receives the same current, destination, neighbor, candidate-edge, and mask inputs; it omits graph message passing.
- Use 15 behavior-cloning epochs (8 episodes per epoch), then 50 PPO epochs, 144 rollout steps per epoch, batch size 64, 4 update epochs, learning rate 3e-5, clip epsilon 0.1, gamma 0.99, GAE lambda 0.9, hidden width 64, and DropEdge 0.05 for GNN-PPO. Record code revision, resolved config, checkpoint hash, device, CPU/thread settings, and wall time for each run.
- Record training episode return and entropy. Use only fixed validation episodes for convergence curves and checkpoint selection.

## Baselines

Evaluate Random, BFS-hop, Dijkstra-km, and Max-SKR on the same 108 test scenarios. Add state-aware weighted shortest path with edge cost

`c_e = w_d*d_norm + w_q*(Q/Q_max) + w_r*(-log(max(r_proxy, 1e-6))) + w_k*(1-pool_norm)`.

Set `w_d=1`; select nonnegative `w_q`, `w_r`, and `w_k` from `{0, 0.5, 1, 2}` on validation only, minimizing validation failure rate and breaking ties by lower successful-route hop count, then lower mean cost. Recompute edge costs from current observable state at each next-hop decision. If this baseline cannot be implemented against the existing evaluator without changing semantics, report it as not implemented and do not invent results.

## Stress, audit, and uncertainty

- Predeclared fiber-outage stress probabilities: 0, 0.01, 0.05, and 0.10. Keep all other parameters and the scenario IDs paired. Report every method, including failure.
- Repeat the reward-component audit on selected final GNN-PPO and MLP-PPO checkpoints using the fixed test scenarios; report per-scenario term differences against BFS and distinguish descriptive reward from route-quality metrics.
- Preserve the existing exact simple-path reward oracle and controlled one-input-at-a-time policy-sensitivity experiments as separate analyses with their own checkpoint and scope labels.
- Report environment-scenario results paired within scenario. Separately report model means and variability across the five training seeds. Do not treat routes from one trained model as independent training replications. For trained-policy intervals, resample training seeds as the outer unit and scenario clusters as the inner unit; include the resampling method and replicates in the result artifact.
- Report inference latency on a stated host with warm-up, deterministic actions, cache disabled, thread count, sample size, mean, p50, p95, and p99. Do not generalize latency to other graph sizes unless they are measured.

## Frozen claims policy

No final abstract or single-checkpoint superiority claim is written until all five seeds of both architectures and the complete paired test suite finish. Any incomplete baseline or experiment is named as such. The final conclusion may state that graph-structured routing helps only if the matched MLP comparison and held-out physical-state tests support that conclusion. QBER screening is not described as a security proof; normalized rate is not described as bits per second; road corridors are not described as deployed fiber; and synthetic weather is not described as measured weather.
