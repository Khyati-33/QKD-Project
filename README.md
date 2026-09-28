# QKD Routing Simulator

A research simulator for routing through a modeled quantum key distribution (QKD) network. It combines a graph-based network environment, fiber and free-space optical (FSO) link models, baseline routing methods, and a PPO agent with graph neural network (GNN) policy support.

## Project status

This repository is an assumption-based simulator, not a field-validated model of an Indian QKD network. The current graph follows stored OpenStreetMap driving routes as proposed fiber-corridor proxies. It does not show verified operator fiber routes, and candidate FSO spans are not checked for line of sight. Weather distributions, optical parameters, QBER, and secret-key-rate (SKR) calculations also have documented limits. See [data/README.md](data/README.md), [PHYSICS_EVIDENCE.md](PHYSICS_EVIDENCE.md), and [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for sources, assumptions, and experiment controls.

The saved 50-epoch run evaluated the archived, abstract topology. Its performance metrics do not apply to the current road-corridor graph; retrain and evaluate before interpreting policy performance on this graph.

## Contents

- `topology.py`, `qkd_env.py`, `physics.py`: network graph, environment, and link models.
- `models.py`, `qkd_attention.py`, `ppo.py`: policy architectures and PPO training.
- `baselines.py`, `evaluation.py`: reference routing policies and evaluation.
- `configs/`: smoke, baseline, and adverse-condition experiment configurations.
- `hardware_profiles/`: device profile inputs used by the simulator.
- `data/`: stored road-route geometry and source/attribution notes.
- `figures/india_road_corridor_map.html`: interactive view of the current topology.
- `experiments/`: selected reports and metrics. Large run artifacts and checkpoints are ignored by Git.
- `tests/`: unit and resume-smoke checks.

## Setup

Use Python 3.10 or newer in a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch Geometric may need an installation command matching your PyTorch version and platform; consult its official installation instructions if the requirements install fails.

## Run

Run the short end-to-end smoke configuration:

```powershell
python run_experiment.py --config configs/smoke_test.yaml
```

Train directly with a configuration:

```powershell
python train.py --config configs/defence_monsoon_night.yaml
```

Generate the device-informed physics report and topology audit:

```powershell
python physics_profile_report.py
python audit_physics_topology.py
```

Regenerate the current interactive road-corridor map:

```powershell
python make_folium_map.py
```

Run checks with pytest:

```powershell
python -m pytest
```

Experiments should record the resolved configuration, code revision, metrics, checkpoints, and evaluation summaries. Follow the protocol before changing protected conditions or drawing conclusions from a run.

## Key documented limitations

- Road routes are OpenStreetMap fastest-driving paths used as corridor proxies, not verified fiber routes or NH-only alignments.
- FSO relay visibility is assumed; terrain and obstacle clearance are not modeled.
- Weather and FSO availability distributions are scenario assumptions, not fitted to time-resolved Indian corridor observations.
- QBER and SKR are simplified simulation proxies, not measured results or a full finite-key security calculation.
- The chosen fiber-device profile informs fiber parameters; the FSO detector and optical settings remain separate assumptions.
- The environment is a partially observed channel-control problem: FSO turbulence now has a correlated latent log-Cn2 state, while future channel states remain hidden from the policy.
- An opt-in vacuum-plus-weak-decoy finite-key BB84 engineering estimator is available through `key_rate_model: finite_key_decoy_bb84`; the default asymptotic proxy remains available for controlled comparisons.
- See [RESEARCH_MODEL.md](RESEARCH_MODEL.md) for the MDP/POMDP interpretation, PPO rationale, constraints, noise-model scope, and validation requirements.

See [PHYSICS_EVIDENCE.md](PHYSICS_EVIDENCE.md) for details, references, and the work needed before making empirical performance claims.
