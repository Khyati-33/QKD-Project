"""Finite-key decoy-state BB84 and device-noise primitives.

This module is deliberately explicit about its scope: it provides a
conservative engineering estimator for finite-key decoy BB84. It is not a
replacement for a complete composable security proof or a certification
implementation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import replace
from typing import Any


@dataclass(frozen=True)
class DecoyBB84Profile:
    signal_mu: float = 0.50
    decoy_mu: float = 0.10
    vacuum_probability: float = 0.10
    decoy_probability: float = 0.20
    signal_probability: float = 0.70
    pulses: int = 1_000_000
    error_correction_efficiency: float = 1.16
    security_epsilon: float = 1e-10
    correctness_epsilon: float = 1e-10
    intensity_relative_uncertainty: float = 0.02

    def __post_init__(self) -> None:
        if not (0 < self.decoy_mu < self.signal_mu):
            raise ValueError("decoy_mu must be positive and below signal_mu")
        if abs(self.signal_probability + self.decoy_probability +
               self.vacuum_probability - 1.0) > 1e-9:
            raise ValueError("source probabilities must sum to one")
        if self.pulses < 1 or self.error_correction_efficiency < 1:
            raise ValueError("invalid finite-key profile")
        if self.intensity_relative_uncertainty < 0:
            raise ValueError("intensity uncertainty must be non-negative")


@dataclass(frozen=True)
class DetectorNoiseProfile:
    efficiency: float = 0.60
    dark_count_hz: float = 100.0
    repetition_rate_hz: float = 100e6
    dead_time_s: float = 30e-9
    afterpulse_probability: float = 0.005
    timing_jitter_s: float = 100e-12
    background_count_hz: float = 0.0
    efficiency_relative_uncertainty: float = 0.05
    dark_count_relative_uncertainty: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")
        if min(self.dark_count_hz, self.repetition_rate_hz, self.dead_time_s,
               self.timing_jitter_s, self.background_count_hz) < 0:
            raise ValueError("detector noise values must be non-negative")
        if not 0 <= self.afterpulse_probability < 1:
            raise ValueError("afterpulse probability must be in [0, 1)")
        if min(self.efficiency_relative_uncertainty,
               self.dark_count_relative_uncertainty) < 0:
            raise ValueError("relative uncertainties must be non-negative")


def binary_entropy(error_rate: float) -> float:
    q = min(max(float(error_rate), 1e-15), 1.0 - 1e-15)
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def detector_click_probability(transmission: float, profile: DetectorNoiseProfile) -> float:
    """Probability of a click including dark/background and dead-time loss."""
    transmission = min(max(float(transmission), 0.0), 1.0)
    optical = profile.efficiency * transmission
    dark = (profile.dark_count_hz + profile.background_count_hz) / profile.repetition_rate_hz
    dead_time_factor = 1.0 / (1.0 + (optical + dark) * profile.repetition_rate_hz * profile.dead_time_s)
    return min(1.0, (1.0 - (1.0 - optical) * (1.0 - min(dark, 1.0))) * dead_time_factor +
               profile.afterpulse_probability * (1.0 - dead_time_factor))


def _count_interval(counts: int, trials: int, epsilon: float) -> tuple[float, float]:
    """Hoeffding confidence interval for an observed Bernoulli gain."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    radius = math.sqrt(math.log(2.0 / epsilon) / (2.0 * trials))
    estimate = counts / trials
    return max(0.0, estimate - radius), min(1.0, estimate + radius)


def _sample_binomial(rng: random.Random, trials: int, probability: float) -> int:
    """Fast reproducible binomial draw using exact small-n or normal large-n."""
    probability = min(max(probability, 0.0), 1.0)
    if trials < 1_000:
        return sum(rng.random() < probability for _ in range(trials))
    mean = trials * probability
    sigma = math.sqrt(max(0.0, trials * probability * (1.0 - probability)))
    return int(min(trials, max(0.0, round(rng.gauss(mean, sigma)))))


def finite_key_decoy_bb84(*, transmission: float, qber: float,
                          source: DecoyBB84Profile | None = None,
                          detector: DetectorNoiseProfile | None = None,
                          rng: random.Random | None = None) -> dict[str, Any]:
    """Estimate finite-key decoy BB84 rate from sampled signal/decoy counts.

    Vacuum and two non-zero intensities are used. Confidence intervals are
    Hoeffding bounds and the single-photon yield/error bounds are conservative
    clipped forms of the standard vacuum+weak-decoy estimator.
    """
    source = source or DecoyBB84Profile()
    detector = detector or DetectorNoiseProfile()
    rng = rng or random.Random()
    # Conservative device/source realization for the current finite-key block:
    # lower efficiency, higher dark counts, and widened intensity bounds.
    detector = replace(
        detector,
        efficiency=max(1e-12, detector.efficiency *
                       (1.0 - 2.0 * detector.efficiency_relative_uncertainty)),
        dark_count_hz=detector.dark_count_hz *
                      (1.0 + 2.0 * detector.dark_count_relative_uncertainty),
    )
    click = detector_click_probability(transmission, detector)
    error = min(max(float(qber), 0.0), 0.5)
    by_intensity = {
        "signal": (source.signal_mu, source.signal_probability),
        "decoy": (source.decoy_mu, source.decoy_probability),
        "vacuum": (0.0, source.vacuum_probability),
    }
    observations: dict[str, dict[str, float]] = {}
    for label, (mu, probability) in by_intensity.items():
        trials = max(1, int(source.pulses * probability))
        poisson_photon = 1.0 - math.exp(-mu * max(transmission, 0.0))
        gain = min(1.0, detector_click_probability(poisson_photon, detector))
        counts = _sample_binomial(rng, min(trials, 2_000_000), gain)
        # Scale-free sampling keeps large configured pulse counts practical.
        sampled_trials = min(trials, 2_000_000)
        err_counts = _sample_binomial(
            rng, sampled_trials, min(0.5, error * gain + 0.5 * (1 - gain)))
        q_lower, q_upper = _count_interval(counts, sampled_trials, source.security_epsilon)
        observations[label] = {"trials": sampled_trials, "gain": counts / sampled_trials,
            "gain_lower": q_lower, "gain_upper": q_upper,
            "qber": err_counts / max(counts, 1)}
    qs = observations["signal"]["gain_lower"]
    qd = observations["decoy"]["gain_lower"]
    q0_upper = observations["vacuum"]["gain_upper"]
    mu_s = source.signal_mu * (1.0 + source.intensity_relative_uncertainty)
    mu_d = source.decoy_mu * (1.0 - source.intensity_relative_uncertainty)
    denominator = mu_s * mu_d - mu_d ** 2
    y1_lower = (mu_s / denominator) * (
        qd * math.exp(mu_d) - (mu_d ** 2 / mu_s ** 2) * qs * math.exp(mu_s) -
        ((mu_s ** 2 - mu_d ** 2) / mu_s ** 2) * q0_upper)
    y1_lower = min(1.0, max(0.0, y1_lower))
    ed_upper = (observations["decoy"]["qber"] * qd * math.exp(mu_d) +
                0.5 * q0_upper) / max(mu_d * y1_lower, 1e-15)
    ed_upper = min(0.5, max(0.0, ed_upper))
    single_photon_fraction = source.signal_probability * mu_s * math.exp(-mu_s)
    privacy = single_photon_fraction * y1_lower * (1.0 - binary_entropy(ed_upper))
    reconciliation = source.error_correction_efficiency * observations["signal"]["gain"] * binary_entropy(
        min(0.5, observations["signal"]["qber"]))
    finite_penalty = (math.log2(2.0 / source.security_epsilon) +
                      math.log2(2.0 / source.correctness_epsilon)) / source.pulses
    rate = max(0.0, privacy - reconciliation - finite_penalty)
    return {"secure_key_rate": rate, "single_photon_yield_lower": y1_lower,
            "single_photon_error_upper": ed_upper, "finite_key_penalty": finite_penalty,
            "observations": observations, "protocol": "vacuum_weak_decoy_bb84_engineering_estimator",
            "security_note": "Hoeffding bounds and clipped decoy estimator; not a certification proof"}
