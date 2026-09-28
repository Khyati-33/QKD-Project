"""Link-level QBER, availability, and secure-key-rate models."""
from __future__ import annotations

import math
import random
from typing import Literal

from quantum_protocol import (DecoyBB84Profile, DetectorNoiseProfile,
                              finite_key_decoy_bb84)

QBER_HARD = 0.11
FIBER_OUTAGE_PROB = 0.01
SEASONS = ("normal", "summer", "winter", "monsoon")
FIBER_HARDWARE_PROFILE_ID = "idq_id281_snspd_1550_uk_ireland_characterized"

FSO_VIABILITY_ANCHORS = {
    # (10 km, 15 km, 30 km, 78 km), chosen within the target seasonal ranges.
    "normal": (0.93, 0.82, 0.54, 0.01),
    "summer": (0.90, 0.76, 0.48, 0.005),
    "winter": (0.92, 0.80, 0.52, 0.008),
    "monsoon": (0.88, 0.70, 0.43, 0.001),
}

# Correlated atmospheric state: the simulator advances this latent state once
# per environment decision instead of drawing independent turbulence for every
# FSO observation. This is a model parameter until it is fitted to a measured
# site time series.
FSO_TURBULENCE_CORRELATION = 0.85

# 1550 nm fiber receiver: ID Quantique ID281 channel characterized in the
# UK-Ireland undersea-link experiment (93.0% SDE, <70 dark counts/s). We use
# the reported upper bound of 70 cps conservatively. These values apply only
# to this 1550 nm fiber receiver, not the separate 785 nm FSO receiver below.
FIBER_DETECTOR_EFFICIENCY = 0.93
FIBER_DARK_COUNT_HZ = 70.0
FIBER_VISIBILITY = 0.98
FIBER_PULSE_RATE_HZ = 100e6
FIBER_ATTENUATION_DB_PER_KM = 0.2
FIBER_EXCESS_LOSS_DB = 0.46
FIBER_CROSSTALK_DB = -30.0


def temperature_at_time(time_of_day_hours: float) -> float:
    """Notebook diurnal temperature proxy: 293 K ± 3 K over 24 hours."""
    phase = 2.0 * math.pi * (time_of_day_hours % 24.0) / 24.0
    return 293.0 + 3.0 * math.sin(phase)


def fiber_qber(distance_km: float, time_of_day_hours: float) -> float:
    """Fiber QBER proxy using the selected 1550 nm ID281 characterization.

    The SNSPD is cryogenically temperature-controlled, so ambient time of day
    does not modulate its dark-count rate. The parameter remains in the
    function signature for the environment API, but only FSO sunlight and
    turbulence vary diurnally. This is still a simplified QBER model: source
    photon statistics, basis errors, finite-key effects, and measured counts
    are not represented.
    """
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")
    dark_count = FIBER_DARK_COUNT_HZ
    transmittance = (10.0 ** (-(FIBER_ATTENUATION_DB_PER_KM * distance_km +
                                FIBER_EXCESS_LOSS_DB) / 10.0) *
                     FIBER_DETECTOR_EFFICIENCY)
    optical_error = (1.0 - FIBER_VISIBILITY) / 2.0
    noise_error = (dark_count / FIBER_PULSE_RATE_HZ) / (transmittance + 1e-12)
    crosstalk_error = 0.5 * 10.0 ** (FIBER_CROSSTALK_DB / 10.0)
    return min(1.0, optical_error + noise_error + crosstalk_error)


def fso_viability_probability(distance_km: float, season: str) -> float:
    """Weather-dependent FSO availability calibrated across detour lengths.

    Piecewise log interpolation matches the specified viability bands and
    keeps short detour links useful for bridging occasional fiber outages.
    """
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")
    if season not in FSO_VIABILITY_ANCHORS:
        raise ValueError(f"unknown season {season!r}; expected one of {SEASONS}")
    if distance_km <= 2.0:
        return 1.0
    p10, p15, p30, p78 = FSO_VIABILITY_ANCHORS[season]
    anchors = ((2.0, 1.0), (10.0, p10), (15.0, p15),
               (30.0, p30), (78.0, p78))
    for (left_km, left_p), (right_km, right_p) in zip(anchors, anchors[1:]):
        if distance_km <= right_km:
            fraction = (distance_km - left_km) / (right_km - left_km)
            return math.exp(math.log(left_p) + fraction *
                            (math.log(right_p) - math.log(left_p)))
    return p78


def sample_correlated_fso_cn2(season: str, time_of_day_hours: float,
                              rng: random.Random,
                              previous_log_cn2: float | None = None,
                              correlation: float = FSO_TURBULENCE_CORRELATION
                              ) -> tuple[float, float]:
    """Draw log-normal turbulence with an AR(1) temporal correlation.

    Returns ``(Cn2, log(Cn2))``. The seasonal and diurnal mean follows the
    existing channel model; only the innovation process is correlated. This
    keeps the marginal distribution unchanged while avoiding independent
    five-minute atmospheric draws.
    """
    if season not in SEASONS:
        raise ValueError(f"unknown season {season!r}; expected one of {SEASONS}")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    weather_factors = {"monsoon": 5.0, "winter": 0.5,
                       "summer": 2.0, "normal": 1.0}
    hour = time_of_day_hours % 24.0
    cn2_base = max(1e-14 * math.exp(-((hour - 13.0) ** 2) / 18.0), 1e-17)
    mean_log = math.log(cn2_base * weather_factors[season])
    sigma = 0.5
    innovation = rng.normalvariate(0.0, sigma)
    if previous_log_cn2 is None:
        log_cn2 = mean_log + innovation
    else:
        log_cn2 = (mean_log + correlation * (previous_log_cn2 - mean_log) +
                   math.sqrt(1.0 - correlation ** 2) * innovation)
    return math.exp(log_cn2), log_cn2


def sample_link_state(link_type: Literal["fiber", "fso"], distance_km: float,
                      season: str = "normal", time_of_day_hours: float = 12.0,
                      rng: random.Random | None = None,
                      fiber_outage_prob: float = FIBER_OUTAGE_PROB,
                      qber_hard: float = QBER_HARD,
                      outage_uniform: float | None = None,
                      outage_distance_km: float | None = None,
                      cn2_override: float | None = None,
                      key_rate_model: Literal["asymptotic_proxy", "finite_key_decoy_bb84"] =
                      "asymptotic_proxy",
                      decoy_profile: DecoyBB84Profile | None = None,
                      detector_profile: DetectorNoiseProfile | None = None) -> dict[str, float | bool]:
    """Sample availability and return QBER, SKR, and outage information."""
    rng = rng or random.Random()
    if link_type == "fiber":
        if not 0.0 <= fiber_outage_prob <= 1.0:
            raise ValueError("fiber_outage_prob must be in [0, 1]")
        outage = rng.random() < fiber_outage_prob
        qber = fiber_qber(distance_km, time_of_day_hours)
        transmission = (10.0 ** (-(FIBER_ATTENUATION_DB_PER_KM * distance_km +
                                   FIBER_EXCESS_LOSS_DB) / 10.0) *
                       FIBER_DETECTOR_EFFICIENCY)
        conditions = {"detector_dark_count_hz": FIBER_DARK_COUNT_HZ,
                      "detector_sde": FIBER_DETECTOR_EFFICIENCY,
                      "wavelength_nm": 1550.0}
    elif link_type == "fso":
        if season not in FSO_VIABILITY_ANCHORS:
            raise ValueError(f"unknown season {season!r}; expected one of {SEASONS}")
        outage_draw = rng.random() if outage_uniform is None else float(outage_uniform)
        if not 0.0 <= outage_draw <= 1.0:
            raise ValueError("outage_uniform must be in [0, 1]")
        availability_distance = (distance_km if outage_distance_km is None
                                 else float(outage_distance_km))
        if availability_distance <= 0:
            raise ValueError("outage_distance_km must be positive")
        outage = outage_draw >= fso_viability_probability(availability_distance, season)
        # Mirror the reference notebook's FSOLink conditions and optical
        # budget: diurnal Cn2, seasonal turbulence/atmospheric factors,
        # diffraction, scintillation loss, pointing, and solar background.
        hour = time_of_day_hours % 24.0
        weather_factors = {
            "monsoon": (5.0, 0.5, 1.0, 2.0),
            "winter": (0.5, 0.8, 1.0, 1.0),
            "summer": (2.0, 1.0, 1.0, 1.0),
            "normal": (1.0, 1.0, 1.0, 1.0),
        }
        cn2_factor, atm_factor, cn2_spread, atm_spread = weather_factors[season]
        cn2_base = max(1e-14 * math.exp(-((hour - 13.0) ** 2) / 18.0), 1e-17)
        cn2_mean = cn2_base * cn2_factor
        atm_mean = min(0.9 * atm_factor, 0.98)
        cn2 = (float(cn2_override) if cn2_override is not None else
               rng.lognormvariate(math.log(cn2_mean), 0.5 * cn2_spread))
        eta_atm = min(1.0, max(0.05, rng.normalvariate(atm_mean, 0.02 * atm_spread)))
        sun_elevation = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))

        wavelength = 785e-9
        beam_waist = 80e-3
        receiver_radius = 200e-3
        detector_efficiency = 0.60
        propagation_m = distance_km * 1000.0
        rayleigh_m = math.pi * beam_waist ** 2 / wavelength
        beam_radius = beam_waist * math.sqrt(1.0 + (propagation_m / rayleigh_m) ** 2)
        eta_diffraction = 1.0 - math.exp(-2.0 * receiver_radius ** 2 / beam_radius ** 2)
        wave_number = 2.0 * math.pi / wavelength
        # The reference notebook's unscaled Cn2=1e-14 produces Rytov
        # variance well above 1 at 9–13 km (near-zero transmittance), which
        # contradicts this project's calibrated 9–13 km detour viability
        # target. Apply an effective-path calibration while retaining the
        # notebook's time/season dependence and exact optical formula.
        effective_cn2 = cn2 * 1e-3
        rytov_variance = (1.23 * effective_cn2 * wave_number ** (7.0 / 6.0) *
                          propagation_m ** (11.0 / 6.0))
        eta_turbulence = math.exp(-rytov_variance / 2.0)
        divergence = wavelength / (math.pi * beam_waist)
        eta_pointing = math.exp(-(1e-6 ** 2) / divergence ** 2)
        transmittance = min(1.0, max(0.0, eta_diffraction * eta_atm *
                                    eta_turbulence * eta_pointing * detector_efficiency))
        transmission = transmittance

        h_planck, speed_of_light = 6.626e-34, 3e8
        photon_energy = h_planck * speed_of_light / wavelength
        background_rate = (1.5e3 * sun_elevation * 0.2 * detector_efficiency * 0.5 *
                           math.pi * receiver_radius ** 2 * 0.1e-9 * (100e-6) ** 2 *
                           0.5e-9 / photon_energy)
        dark_rate = 100.0 + background_rate
        qber = min(1.0, (1.0 - 0.98) / 2.0 +
                   (dark_rate / 100e6) / (transmittance + 1e-12))
        conditions = {"hour": hour, "Cn2": cn2, "eta_atm": eta_atm,
                      "effective_Cn2": effective_cn2, "sun_elevation": sun_elevation,
                      "transmittance": transmittance,
                      "rytov_variance": rytov_variance}
    else:
        raise ValueError(f"unknown link type {link_type!r}")
    if key_rate_model == "asymptotic_proxy":
        skr = secure_key_rate(qber, distance_km, outage=outage,
                              qber_hard=qber_hard, transmission=transmission)
        finite_key = None
    elif key_rate_model == "finite_key_decoy_bb84":
        finite_key = finite_key_decoy_bb84(
            transmission=transmission, qber=qber,
            source=decoy_profile, detector=detector_profile, rng=rng)
        skr = 0.0 if outage else float(finite_key["secure_key_rate"])
    else:
        raise ValueError(f"unknown key_rate_model {key_rate_model!r}")
    if finite_key is not None:
        conditions["finite_key"] = finite_key
    return {"qber": qber, "skr": skr, "outage": outage,
            "conditions": conditions, "time_of_day_hours": time_of_day_hours % 24.0}


def secure_key_rate(qber: float, distance_km: float, *, outage: bool = False,
                    qber_hard: float = QBER_HARD,
                    transmission: float | None = None) -> float:
    """A compact asymptotic BB84-style proxy, positive only for viable links."""
    if outage or qber >= qber_hard:
        return 0.0
    # Binary entropy penalty; the distance factor represents transmission loss.
    q = min(max(qber, 1e-12), 1.0 - 1e-12)
    entropy = -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)
    transmission = (10.0 ** (-distance_km / 100.0) if transmission is None
                    else max(0.0, float(transmission)))
    return max(0.0, transmission * (1.0 - 2.0 * entropy))


def edge_is_valid(qber: float, skr: float, qber_hard: float = QBER_HARD) -> bool:
    """Shared physical validity rule used by the environment and policy masks."""
    return qber < qber_hard and skr > 0.0


def chain_parity_error(hop_qbers: list[float] | tuple[float, ...]) -> float:
    """Cumulative parity-error probability for independent hop errors."""
    parity = 0.0
    for qber in hop_qbers:
        if not 0.0 <= qber <= 1.0:
            raise ValueError("each QBER must be in [0, 1]")
        parity = parity * (1.0 - qber) + (1.0 - parity) * qber
    return parity


def sample_fiber_outage(rng: random.Random | None = None,
                        fiber_outage_prob: float = FIBER_OUTAGE_PROB) -> bool:
    rng = rng or random.Random()
    return rng.random() < fiber_outage_prob
