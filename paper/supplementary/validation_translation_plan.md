# Validation and translation plan

This project is suitable for an NQM research setting without proprietary
experimental traces if every result is labelled by its evidence level. NQM
explicitly targets secure quantum communication, inter-city QKD, single-photon
sources/detectors, and indigenous quantum devices, so the simulator should be
used as a design and uncertainty-analysis tool until hardware traces exist.

## Evidence levels

### Level 0: implementation correctness

- Unit tests for attenuation, QBER, click probabilities, finite-key bounds,
  outage sampling, temporal correlation, and route constraints.
- Limiting cases: zero loss, zero dark counts, no turbulence, full outage,
  QBER at the strict threshold, and depleted key pool.
- Deterministic seeds and serialized configurations.

### Level 1: analytical validation

- Compare fiber attenuation and dark-count QBER against closed-form equations.
- Compare log-normal turbulence moments and AR(1) autocorrelation against the
  configured distribution.
- Compare decoy-state bounds against hand-calculated and independent reference
  implementations.
- Verify that uncertainty intervals widen as pulse count decreases.

### Level 2: literature and vendor calibration

- Use only traceable papers, standards, or vendor specifications.
- Record wavelength, detector efficiency, dark counts, timing jitter, dead
  time, afterpulsing, source intensity, and uncertainty for every profile.
- Reproduce published fiber and FSO curves before running routing experiments.
- Label every parameter as measured, published, vendor-reported, inferred, or
  assumed.

### Level 3: synthetic trace stress testing

- Generate held-out correlated turbulence and outage traces from fitted priors.
- Test unseen weather mixtures, non-stationary means, burst outages, and
  parameter shifts.
- Report calibration error, coverage of uncertainty intervals, outage
  probability, QBER violations, secure-key yield, and route cost.

### Level 4: hardware-in-the-loop or field validation

- Replace priors with timestamped detector and channel traces.
- Refit parameters without changing the routing code.
- Hold out complete days/sites for evaluation.
- Report sim-to-real error separately from routing-policy error.

## What can be claimed now

The current results support a physics-informed, uncertainty-aware routing
simulation study. They do not support a claim of field performance, certified
secure-key generation, or commercial device suitability. The finite-key path
is an engineering estimator until independently checked against a formal
composable proof and measured count data.

## Commercial characterization plan

For a future product, create one versioned profile per device and test:

1. Source: wavelength, pulse rate, intensity distribution, decoy stability,
   extinction ratio, spectral width, polarization extinction, and drift.
2. Receiver: detection efficiency versus wavelength/bias, dark-count rate,
   timing jitter, dead time, afterpulsing, saturation, and basis imbalance.
3. Optics: insertion loss, coupling loss, aperture, beam waist, pointing
   jitter, tracking bandwidth, and boresight error.
4. Environment: temperature, humidity, visibility, solar background, wind,
   vibration, and turbulence/scintillation traces.
5. Protocol: finite-key block size, error-correction leakage, authentication,
   privacy amplification, key-management interface, and failure semantics.

Each test should have a calibration artifact, uncertainty budget, acceptance
threshold, and raw-data hash. The simulator should consume the profile without
silently changing defaults.

## NQM and standards alignment

The network/system framing should be aligned with [ITU-T Y.3800](https://www.itu.int/epublications/publication/itu-t-y-3800-2019-10-overview-on-networks-supporting-quantum-key-distribution) and [ITU-T Y.3801](https://www.itu.int/rec/T-REC-Y.3801-202004-I), and the protocol/security discussion with [ETSI GS QKD 005](https://www.etsi.org/deliver/etsi_gs/QKD/001_099/005/01.01.01_60/gs_QKD005v010101p.pdf). [C-DOT's public QKD system specification](https://www.cdot.in/cdotweb/assets/docs/products/optical/qkd.pdf) is useful for Indian system interface and interoperability requirements. These references guide requirements traceability; they do not substitute for a device measurement.
