# Physics Evidence Ledger and Distance Audit

Audit date: 2026-09-27

## Status key

- **Measured support**: a published experiment reports measurements relevant to the quantity.
- **Published model support**: a published analytical or simulation method informs the quantity, but it is not a direct measurement of this project's link.
- **Inherited target**: carried over from the project handoff or its notebook description; no traceable publication or raw data is attached here.
- **Scenario assumption**: chosen to make a runnable simulator; it is not an empirical claim.
- **Derived**: computed from the topology or another listed value.

**Current conclusion:** this repository has no end-to-end QBER/SKR/outage parameter set calibrated to a single matching QKD hardware system and Indian corridor weather dataset. The publications below are useful anchors, but none validates the present 80 km fiber hops or the full 80 km FSO detours as deployed links.

## Current code values

## Selected hardware profile

The selected receiver profile is the **ID Quantique ID281 SNSPD characterized at 1550 nm** in the published UK–Ireland 224 km undersea-fiber feasibility experiment. The paper reports two detector channels; the characterization cited here is 93.0% system detection efficiency and dark counts below 70 counts/s per channel. The profile records timing jitter, recovery time, temperature, and several other device characteristics as unknown for that tested channel rather than borrowing unrelated values from a different unit. ID Quantique's family-level figures (<40 ps FWHM and typically <30 ns recovery for standard SNSPDs) are recorded as manufacturer capabilities only, not as measurements of the selected channel.

The machine-readable profile is [hardware_profiles/idq_id281_1550_uk_ireland.yaml](hardware_profiles/idq_id281_1550_uk_ireland.yaml). `physics.py` now uses its 0.93 efficiency and 70 cps conservative dark-count bound for the 1550 nm fiber leg. The values are not applied to the simulator's separate 785 nm FSO leg. This remains a detector-informed model, not a complete QKD system calibration: source statistics, protocol, receiver optics, timing window, channel loss, error correction, and finite-key security parameters are still needed before QBER/SKR can represent measured or security-proof rates.

Sources: [published undersea-link experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10743312/), [ID Quantique ID281 product specifications](https://www.idquantique.com/quantum-detection-systems/products/id281-snspd-system/).

### Fiber and protocol model (`physics.py`)

| Code quantity | Current value / rule | Status | Evidence and audit note |
|---|---:|---|---|
| `QBER_HARD` | 0.11 | Inherited target | A hard cutoff is protocol/security-model dependent. The code does not identify a complete QKD protocol, finite-key treatment, or security proof that validates 11% for this system. |
| `FIBER_OUTAGE_PROB` | 0.01 per sampled link | Scenario assumption | Independent Bernoulli outage; no carrier, maintenance, or field-failure dataset supports 1% or independence between links. |
| Fiber attenuation | 0.2 dB/km at modeled 1550 nm | Inherited target | No measured fiber type or route-specific attenuation is specified. The attenuation must be tied to a fiber class and route measurement. |
| Excess loss | 0.46 dB | Inherited target | No connector/splice budget or source measurement recorded. |
| Fiber detector efficiency | 0.93 at 1550 nm | Measured device characterization | ID281 channel value reported in the UK–Ireland undersea-link experiment; see selected profile above. |
| Fiber dark count | 70 Hz used as a conservative upper bound at 1550 nm | Measured upper bound | The experiment reports <70 counts/s per channel. This is not an exact measured value; the model uses the bound. |
| Visibility | 0.98 | Inherited target | Treated as a fixed optical error parameter; no calibration source recorded. |
| Pulse rate | 100 MHz | Inherited target | Source/device configuration is not recorded. |
| Fiber detector ambient-temperature coefficient | Removed; SNSPD dark counts are held fixed at the characterized bound | Device-physics correction | A cryogenically stabilized SNSPD's detector temperature should not track ambient diurnal temperature. `temperature_at_time()` remains as a legacy utility but no longer changes fiber QBER. |
| Crosstalk | −30 dB | Inherited target | No component or measured crosstalk source recorded. |
| Fiber QBER | Visibility + dark-count/transmission + crosstalk formula | Simplified device-informed proxy, unvalidated | Uses measured ID281 SDE and conservative dark-count bound, but visibility, 100 MHz rate, attenuation, excess loss, crosstalk, and source model are not calibrated together. It is not an experimental QBER. |

### FSO/weather and SKR model (`physics.py`)

| Code quantity | Current value / rule | Status | Evidence and audit note |
|---|---:|---|---|
| FSO availability anchors | Normal: 0.93/0.82/0.54/0.01; summer: 0.90/0.76/0.48/0.005; winter: 0.92/0.80/0.52/0.008; monsoon: 0.88/0.70/0.43/0.001 at 10/15/30/78 km | Inherited target / scenario assumption | These are handoff target values, not a fitted India dataset or a measured QKD link. Log interpolation between them is also a modeling choice. |
| FSO outage sampling | One shared draw per eight-segment detour | Scenario assumption | Models a detour as fully available or unavailable together. No measured corridor outage-correlation data supports this structure. |
| Turbulence/weather | `Cn2` base 1e-14, diurnal envelope, season multipliers; monsoon multiplier 5 | Scenario assumption | No local `Cn2`, visibility, rain, aerosol, or turbulence time series was used to fit these distributions. |
| Effective turbulence scale | `Cn2 * 1e-3` | Explicit numerical calibration, unsupported | The code comment says this was introduced to avoid near-zero transmittance and retain target viability. It is not calibrated from a cited experiment. |
| FSO optics | 785 nm, 80 mm beam waist, 200 mm receiver radius (400 mm diameter), 60% detector efficiency, 1 µrad pointing jitter | Inherited target / scenario assumption | No matching device profile or measured end-to-end optical budget is documented. |
| FSO QBER | Fixed 1% baseline plus modeled background/transmittance penalty | Scenario assumption | No detector count traces or protocol-specific QBER fit are present. |
| SKR | `T * (1 - 2 h2(QBER))` proxy | Simplified theoretical proxy | This is not a full finite-key implementation and has no calibrated source intensity, error-correction efficiency, detector counts, protocol, or security parameter. It should not be called a measured SKR. |

Derived outputs are saved in `experiments/physics_profile_report_idq.json` and copied into the run artifact. At 80 km, the model computes 16.46 dB fiber loss including excess loss, 0.0210 detection probability per incident photon, QBER proxy 0.010533, and normalized SKR proxy 0.01747. The 2.10 Mcps figure at 100 MHz assumes one photon per pulse and is only an idealized detection bound; the weak-pulse source statistics needed for a QKD rate are omitted. At 78 km, the corresponding QBER proxy is 0.010530 and normalized SKR proxy 0.01916.

For 10 km monsoon FSO, the 1,000-sample model report gives configured availability 88% and empirical availability 88.5%. Median total detection probability (the code field called transmittance includes detector efficiency) and normalized SKR proxy are 0.2426 and 0.2035 at 22:00; at 14:00 they are 0.2267 and 0.1902. Median Rytov variance is 0.00162 at 22:00 and 0.1389 at 14:00. These are model-derived conditional statistics, not Indian weather observations or field measurements. The chosen `Cn2 * 1e-3` path calibration remains unsupported by measurement.

## 50-epoch agent response check

The GNN was trained for 15 BC epochs and 50 PPO epochs in monsoon starting at 22:00 with ±1 hour jitter. Across the final checkpoint's four-season, four-seed evaluation, it achieved 100% success, 23.375 mean hops (23-hop reference), no revisits in the logged representative route, and 100% validity among 374 traversed QBER/SKR edges. Its representative route used 23 fiber edges and zero FSO edges.

The controlled edge-choice diagnostic held progress and candidate-neighbor embedding fixed and compared an 80 km ID281 fiber sample with a clear 10 km weather-conditioned FSO sample. FSO had the higher normalized SKR proxy in all 384 clear-path comparisons, but the best-by-reward policy selected FSO in 0/384 decisions; its average FSO probability stayed around 0.478–0.483. The same no-FSO result held for the other saved checkpoints: monsoon-night FSO probabilities were 0.487 for best-by-success-rate and 0.472 for latest, with 0 selections in each set of 32 trials. Within each condition, the best-by-reward FSO probability was inversely correlated with the FSO SKR proxy in this controlled twin-edge test. This shows that the model received physics-derived features but did **not** learn a useful preference for the physically higher-proxy-SKR FSO candidate. Route success therefore does not establish weather-aware link choice. Raw samples and summaries are in `experiments/runs/defence_monsoon_night_50ep_idq_20260927T165951Z_9ce167b1/link_quality_choice_test*.json` and `evaluation.json`.

### Published anchors and their limits

| Source | Relevant reported evidence | What it supports here | What it does not establish |
|---|---|---|---|
| [Ecker et al., *Towards metropolitan free-space quantum networks* (2023)](https://www.nature.com/articles/s41534-023-00754-0) | Ground-to-ground 1.7 km QKD experiment in Jena; 810 nm quantum channel, active beacon/beam stabilization, 200 mm telescope aperture; night QBER below 2% and mean SKR 5.4 kbps. The authors estimate 3.3 kbps for 10 km from the measured night data, assuming atmospheric scattering/absorption negligible. | A measured short metropolitan QKD link and a clearly labeled 10 km extrapolation. Useful for defining one possible reference profile. | Direct measured performance of our 10 km relays, Indian monsoon availability, our 785 nm / 80 mm waist / 400 mm aperture optics, or our generic SKR equation. |
| [Kshatriya et al., *Estimation of FSO link availability using climatic data* (2016)](https://link.springer.com/article/10.1007/s12596-016-0327-4) | Feasibility study for four Indian cities using 365 days of 2013 visibility data sourced from Wundermap. | Evidence that Indian city-specific climate inputs affect availability estimates; a lead for reproducing a published Indian availability analysis. | Direct QKD link measurements, current weather, all seven cities, or the present link/device setup. |
| [Basahel et al., tropical FSO availability study (2018)](https://doi.org/10.1016/j.ijleo.2017.11.203) | Availability analysis using three years of measured Malaysian visibility data; analyzes a 5 km terrestrial FSO link and specific 1550 nm / transmitter-power assumptions. | Shows how long-term visibility data can be used to derive site-specific attenuation and availability. | India-specific monsoon probabilities, 10 km QKD relay performance, or our optical setup. |
| [ITU-R P.1814-1 (2025)](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1814-1-202509-I%21%21PDF-E.pdf) | Terrestrial-FSO prediction method covering atmospheric attenuation, turbulence/scintillation, link geometry, misalignment, ambient light, and local weather. | A structured channel-budget and planning checklist. | Experimental validation of this project's parameters. It is a prediction recommendation, not a measurement dataset. |

The 1.7 km experiment's 10 km result is explicitly an extrapolation, not a 10 km field measurement. Its optical terminal also differs from this repository's assumed optics. The 10 km model segment is therefore **within a published extrapolation range**, but not validated for this simulator.

## Applied-device model boundary

`physics.py` now consumes the selected ID281 fiber-detector values (0.93 SDE and 70 cps as a conservative upper bound). Its separate FSO model remains at 785 nm with a 0.60 detector-efficiency scenario value and assumed 100 Hz baseline dark count. That FSO detector is **not** characterized by the selected 1550 nm device profile. The FSO channel keeps weather-dependent availability, seasonal atmospheric transmission, diurnal turbulence, and daylight background. Fiber QBER no longer receives an ambient-time-of-day dark-count multiplier. The resulting QBER and SKR values are device-informed simulation outputs, not measured performance or security-proof key rates.

## Topology and distance audit (`topology.py`)

Computed directly from `build_topology()`:

The audit is reproducible with `python audit_physics_topology.py`.

- 289 nodes, 319 total edges: 87 fiber edges and 232 FSO edges.
- Every fiber edge is assigned 80 km.
- There are 29 FSO detours, each consisting of eight 10 km links: 80 km of modeled FSO path per bypass of one 80 km fiber hop.
- The default all-fiber Delhi→Chennai route is 23 hops / 1,840 modeled km.
- Every modeled FSO segment is 10 km. Each complete detour is 80 km; the code models its eight constituent links separately.

The listed city coordinates are approximate. The following check compares each corridor's assigned total fiber distance with the straight-line geodesic between its endpoint city coordinates. Geodesic distance is a lower bound on a real surface route; a modeled corridor shorter than that lower bound cannot be geographically valid.

| Corridor | Assigned fiber distance | Endpoint geodesic | Assigned / geodesic | Audit |
|---|---:|---:|---:|---|
| Delhi–Jaipur | 240 km | 235 km | 1.02 | Plausible lower-bound check only |
| Jaipur–Mumbai | 720 km | 921 km | 0.78 | **Impossible as drawn**: assigned route is shorter than endpoint geodesic |
| Mumbai–Bangalore | 480 km | 845 km | 0.57 | **Impossible as drawn**: assigned route is shorter than endpoint geodesic |
| Bangalore–Chennai | 400 km | 290 km | 1.38 | Possible, but not an actual surveyed route |
| Delhi–Hyderabad | 1,280 km | 1,255 km | 1.02 | Plausible lower-bound check only |
| Hyderabad–Chennai | 640 km | 515 km | 1.24 | Possible, but not an actual surveyed route |
| Delhi–Kolkata | 1,360 km | 1,304 km | 1.04 | Plausible lower-bound check only |
| Kolkata–Chennai | 1,840 km | 1,358 km | 1.35 | Possible, but not an actual surveyed route |

Thus the current graph is a controlled abstract topology, not a geographically consistent India fiber map. Its FSO detours are also synthetic parallel chains, not surveyed line-of-sight corridors. The 80 km fiber and FSO spans should not be interpreted as validated deployment distances.

## Required work before claiming empirical realism

1. Select one reference QKD system and protocol, with published device settings and measured QBER/SKR data. Keep the 810 nm Ecker et al. experiment as a candidate profile only; it does not match current assumptions without changes.
2. Replace city-center straight-line corridors with surveyed fiber routes or explicitly label the graph synthetic. Ensure every route distance is at least the endpoint geodesic.
3. Use India-specific, time-resolved weather/visibility data for each modeled FSO corridor; document source, time span, spatial resolution, and missing-data treatment. Fit outage/attenuation distributions and temporal correlation from those data.
4. Derive QBER and SKR from the selected protocol/device model and measured channel loss/count data. Keep measured points separate from extrapolated ranges.
5. Reproduce published reference curves/points and validate on held-out sites or time periods before training or interpreting routing choices.

Until these steps are complete, policy results should be described as performance in the **assumption-based simulator**, not as expected real Indian QKD network performance.

## Follow-up implementation after the archived 50-epoch run

The run above used code from the pre-experiment baseline commit. Subsequent
debugging added a monotone log-SKR action prior, distance-normalized SKR reward,
route-cost and hop-aware behavior-cloning tie breaks, and episode-persistent
FSO corridor outage draws. These changes address policy sensitivity, reward
inflation from segmenting a detour, and routes stranded by a per-step outage
redraw. The archived metrics above are unchanged and do not measure these new
methods; small regression and retraining checks are recorded in the Git history.

Randomized endpoint evaluation now records the actual source and destination
for every episode. `configs/defence_monsoon_night_50ep_idq.yaml` randomizes
endpoints in behavior-cloning demonstrations while retaining the protected
Delhi-to-Chennai evaluation condition. This broadens supervision but does not
make the synthetic topology geographically valid or calibrate its channel
physics.

## Updated 50-epoch rerun (2026-09-28)

The updated approach was trained with `configs/defence_monsoon_night_50ep_idq.yaml` for 15 behavior-cloning epochs and 50 PPO epochs. Run artifacts are in `experiments/runs/defence_monsoon_night_50ep_idq_20260928T044715Z_c84be8c5/`. The run completed all 50 epochs and evaluated every five epochs.

At epoch 50, evaluation completed 16/16 Delhi-to-Chennai episodes (four seeds in each of four seasons). Mean successful route length was 23.125 hops against a 23-hop all-fiber reference. Normal, Summer, and Monsoon averaged 23 hops; Winter averaged 23.5. All 370/370 traversed links met the simulator's QBER/SKR edge constraints, with no fallback actions. The representative route had 23 fiber links, no FSO links, and no revisits. Dijkstra-by-kilometers and BFS-by-hops each also succeeded in 16/16 episodes, with 23.25 mean hops; Random and Max-SKR had no successful episodes.

The controlled link-choice diagnostic sampled 32 matched-progress decisions for each of 12 season/hour conditions (384 total). On clear FSO samples, the FSO SKR proxy exceeded the fiber proxy in all 384 comparisons and was selected in all 384. Mean FSO action probability was about 0.790 across conditions. This tests relative preference conditional on clear FSO and matched route context; it does not represent marginal weather availability or prove that full routes should use FSO. The logged representative end-to-end route still used fiber.

The run exposed a checkpoint bookkeeping defect: because evaluations are scheduled every five epochs, unscheduled epochs had compared rollout metrics against scheduled-evaluation metrics. The saved `best_by_reward.pt` and `best_by_success_rate.pt` therefore both point to epoch 27 and should not be used for this run. `latest.pt` and the epoch-50 evaluation are the valid final artifacts. The checkpoint-selection fix in the subsequent commit only updates best checkpoints on scheduled evaluations and resets stored selection scores when resuming with a different selection mode. This fix was added after the experiment; it did not change the completed run's training or metrics.

These are results in the assumption-based simulator, not measured QKD performance or validation of a geographically realistic Indian network. The randomized behavior-cloning endpoints are not an independent held-out evaluation set.
