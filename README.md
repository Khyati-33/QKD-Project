# QKD Routing Simulator

A research simulator for routing through a modeled quantum key distribution (QKD) network. It combines a graph-based network environment, fiber and free-space optical (FSO) link models, baseline routing methods, and a PPO agent with graph neural network (GNN) policy support.

## Project status

This repository is an assumption-based simulator, not a field-validated model of an Indian QKD network. Its topology, weather distributions, optical parameters, QBER, and secret-key-rate (SKR) calculations have documented limitations. See [PHYSICS_EVIDENCE.md](PHYSICS_EVIDENCE.md) for the physics and topology audit, and [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for experiment controls and reporting requirements.

One diagnostic found that the trained policy did not prefer the FSO candidate in a controlled comparison, even when its modeled SKR proxy was higher. Route completion alone should not be interpreted as evidence of weather-aware link selection.

## Contents

- `topology.py`, `qkd_env.py`, `physics.py`: network graph, environment, and link models.
- `models.py`, `qkd_attention.py`, `ppo.py`: policy architectures and PPO training.
- `baselines.py`, `evaluation.py`: reference routing policies and evaluation.
- `configs/`: smoke, baseline, and adverse-condition experiment configurations.
- `hardware_profiles/`: device profile inputs used by the simulator.
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

Run checks with pytest:

```powershell
python -m pytest
```

Experiments should record the resolved configuration, code revision, metrics, checkpoints, and evaluation summaries. Follow the protocol before changing protected conditions or drawing conclusions from a run.

## Key documented limitations

- The modeled India route graph is synthetic; some assigned corridor distances are shorter than the endpoint geodesic distance.
- Weather and FSO availability distributions are scenario assumptions, not fitted to time-resolved Indian corridor observations.
- QBER and SKR are simplified simulation proxies, not measured results or a full finite-key security calculation.
- The chosen fiber-device profile informs fiber parameters; the FSO detector and optical settings remain separate assumptions.

See [PHYSICS_EVIDENCE.md](PHYSICS_EVIDENCE.md) for details, references, and the work needed before making empirical performance claims.
