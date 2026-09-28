"""Compute device-informed fiber and scenario FSO values reproducibly."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from physics import (
    FIBER_ATTENUATION_DB_PER_KM,
    FIBER_DARK_COUNT_HZ,
    FIBER_DETECTOR_EFFICIENCY,
    FIBER_EXCESS_LOSS_DB,
    FIBER_PULSE_RATE_HZ,
    FIBER_VISIBILITY,
    FSO_VIABILITY_ANCHORS,
    fiber_qber,
    fso_viability_probability,
    sample_link_state,
)


def derived_report(samples: int = 1000) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    fiber_results = {}
    for distance_km in (78.0, 80.0):
        loss_db = FIBER_ATTENUATION_DB_PER_KM * distance_km + FIBER_EXCESS_LOSS_DB
        channel_transmission = 10 ** (-loss_db / 10)
        detection_probability = channel_transmission * FIBER_DETECTOR_EFFICIENCY
        qber = fiber_qber(distance_km, 22.0)
        # secure_key_rate() uses a normalized asymptotic proxy, not bits/s.
        from physics import secure_key_rate
        fiber_results[str(int(distance_km))] = {
            "distance_km": distance_km,
            "wavelength_nm": 1550,
            "fiber_loss_db_including_excess": loss_db,
            "channel_transmission_before_detector": channel_transmission,
            "system_detection_efficiency": FIBER_DETECTOR_EFFICIENCY,
            "single_photon_detection_probability_per_incident_photon": detection_probability,
            "idealized_detection_events_per_second_at_100_MHz_one_photon_pulses":
                detection_probability * FIBER_PULSE_RATE_HZ,
            "detector_dark_count_upper_bound_hz": FIBER_DARK_COUNT_HZ,
            "visibility_assumption": FIBER_VISIBILITY,
            "qber_proxy": qber,
            "normalized_skr_proxy": secure_key_rate(qber, distance_km,
                                                       transmission=detection_probability),
        }

    fso_results = {}
    for season_index, season in enumerate(FSO_VIABILITY_ANCHORS):
        for hour in (2.0, 14.0, 22.0):
            rng = random.Random(20260927 + season_index * 100 + int(hour))
            states = [sample_link_state("fso", 10.0, season, hour, rng)
                      for _ in range(samples)]
            conditions = [state["conditions"] for state in states]
            fso_results[f"{season}_{int(hour):02d}h"] = {
                "distance_km": 10.0,
                "wavelength_nm": 785,
                "weather_availability_probability": fso_viability_probability(10.0, season),
                "empirical_availability": sum(not state["outage"] for state in states) / samples,
                "qber_proxy_median_conditional_on_samples": statistics.median(
                    state["qber"] for state in states if not state["outage"]),
                "normalized_skr_proxy_median_conditional_on_samples": statistics.median(
                    state["skr"] for state in states if not state["outage"]),
                "transmittance_median": statistics.median(c["transmittance"] for c in conditions),
                "cn2_median_per_m_to_minus_two_thirds": statistics.median(c["Cn2"] for c in conditions),
                "effective_cn2_scale_assumption": 1e-3,
                "rytov_variance_median": statistics.median(c["rytov_variance"] for c in conditions),
                "fso_detector_efficiency_assumption": 0.60,
                "baseline_dark_count_assumption_hz": 100.0,
            }
    return {
        "scope": "derived simulator outputs; not measured end-to-end performance",
        "fiber_detector_profile": "IDQ ID281, characterized 1550 nm channel",
        "fiber": fiber_results,
        "fso": fso_results,
        "interpretation": {
            "fiber_count_rate": "idealized one-photon-per-pulse bound; QKD weak-pulse source statistics omitted",
            "skr": "dimensionless normalized proxy, not bits/s or finite-key secure rate",
            "fso": "weather/turbulence scenario; current FSO detector and effective-Cn2 scale are not device-calibrated",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = derived_report(args.samples)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf8")
    print(rendered)


if __name__ == "__main__":
    main()
