# Research model and paper claims

## What the environment is

The routing environment is a **POMDP approximation**, not a fully observed
MDP. At decision time, the agent observes the current node, destination,
candidate-link QBER/SKR, candidate validity mask, trust-chain state, visited
state, and key-pool state. It does not observe future turbulence, future fiber
outages, the latent FSO `Cn2` state of every corridor, or the future actions of
other network users. The observation is therefore an emission from a hidden
channel state.

The environment becomes an MDP only after augmenting the state with every
latent channel variable and queue/resource variable. That state is not
available to the policy. The current feed-forward GNN is consequently a
reactive policy over the current observation, not a full belief-state policy.

## Why PPO is used

PPO is suitable as a first constrained-policy method because the action is a
masked discrete next-hop choice, transitions are stochastic, rewards combine
progress, QBER/SKR validity, trust-chain security, pool depletion, latency,
energy, congestion, switching, and revisit penalties, and route termination
is variable. PPO also supports on-policy training under changing channel
realizations without requiring a differentiable channel model.

PPO does not make the policy POMDP-optimal. A research-grade POMDP extension
should use a recurrent GNN or an explicit belief state containing recent QBER,
SKR, outage, and time-series summaries. The current `LSTMActorCritic` is an
architecture option, but training must carry hidden state across episode steps
before it can be described as recurrent POMDP control.

## State, action, and constraints

- **Action:** select one neighbor slot; invalid physical links are masked.
- **Hard constraints:** QBER below the configured threshold, positive SKR,
  trust-chain parity error below threshold, non-depleted key pool, maximum
  route steps, and configured fiber/FSO distance limits.
- **Soft objectives:** progress, SKR, key pool, latency, energy, congestion,
  link switching, revisits, and route security.
- **Physical uncertainty:** FSO turbulence is now sampled as a temporally
  correlated latent log-`Cn2` process. The current correlation value is an
  engineering assumption until fitted to measured time series.

The paper must state whether invalid actions are impossible actions (masking),
failed transmissions with a penalty, or unavailable links. Mixing these
interpretations changes the policy problem.

## Noise model scope

The fiber model currently includes attenuation, visibility, detector efficiency,
dark-count noise, crosstalk, and stochastic outages. The FSO model includes
diffraction, atmospheric transmission, turbulence-dependent transmittance,
pointing loss, background counts, seasonal/diurnal factors, and correlated
latent turbulence. The 785 nm FSO device profile is explicitly labelled an
assumption, not a field characterization.

This is still not a complete QKD security proof. Source photon statistics,
decoy-state estimation, finite-key bounds, error-correction leakage, detector
dead time, afterpulsing, timing jitter, and composable security must be added
before making device-secure-key claims.

## Required validation before a journal claim

1. Fit fiber and FSO parameters to measured or traceable device/channel data.
2. Reproduce published attenuation, turbulence, QBER, and key-rate curves.
3. Add finite-key decoy-state BB84 bounds with an explicit security parameter.
4. Report uncertainty intervals and sensitivity for every physical parameter.
5. Compare PPO/GNN, recurrent PPO, BFS, Dijkstra, Max-SKR, robust shortest
   path, and a chance-constrained optimizer over identical channel traces.
6. Use held-out weather traces, multiple seeds, confidence intervals, and
   statistical tests. Report success, secure-key yield, outage probability,
   QBER violations, route length, latency, energy, entropy, and inference cost.
