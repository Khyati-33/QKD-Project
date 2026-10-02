"""Independent formula and limiting-trend checks for the configured link model.

This validates internal consistency and monotonic limiting behavior only. It
does not calibrate either model against measured device or weather traces.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics import (FIBER_ATTENUATION_DB_PER_KM, FIBER_CROSSTALK_DB,
    FIBER_DARK_COUNT_HZ, FIBER_DETECTOR_EFFICIENCY, FIBER_EXCESS_LOSS_DB,
    FIBER_PULSE_RATE_HZ, FIBER_VISIBILITY, FSO_TURBULENCE_CORRELATION,
    FSO_VIABILITY_ANCHORS, SEASONS, fiber_qber, fso_viability_probability,
    sample_correlated_fso_cn2, secure_key_rate)


def entropy(q: float) -> float:
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def main() -> None:
    distances = np.linspace(0.0, 200.0, 201)
    rows = []
    for distance in distances:
        # Independent transcription of the stated model equations.
        loss_db = FIBER_ATTENUATION_DB_PER_KM * distance + FIBER_EXCESS_LOSS_DB
        transmission = 10.0 ** (-loss_db / 10.0) * FIBER_DETECTOR_EFFICIENCY
        q = min(1.0, (1.0 - FIBER_VISIBILITY) / 2.0 +
                (FIBER_DARK_COUNT_HZ / FIBER_PULSE_RATE_HZ) / (transmission + 1e-12) +
                0.5 * 10.0 ** (FIBER_CROSSTALK_DB / 10.0))
        rate = (max(0.0, transmission * (1.0 - 2.0 * entropy(min(max(q, 1e-12),
                    1.0 - 1e-12)))) if q < 0.11 else 0.0)
        code_q = fiber_qber(float(distance), 12.0)
        code_rate = secure_key_rate(code_q, float(distance), transmission=transmission)
        rows.append({"distance_km": float(distance), "loss_db": loss_db,
                     "transmission": transmission, "qber": q, "rate_proxy": rate,
                     "code_qber": code_q, "code_rate_proxy": code_rate})

    qber_errors = [abs(row["qber"] - row["code_qber"]) for row in rows]
    rate_errors = [abs(row["rate_proxy"] - row["code_rate_proxy"]) for row in rows]
    if any(b["transmission"] > a["transmission"] + 1e-12 for a, b in zip(rows, rows[1:])):
        raise AssertionError("fiber transmission must decrease with distance")
    if any(b["qber"] < a["qber"] - 1e-12 for a, b in zip(rows, rows[1:])):
        raise AssertionError("fiber QBER must not decrease with distance")
    if any(b["rate_proxy"] > a["rate_proxy"] + 1e-12 for a, b in zip(rows, rows[1:])):
        raise AssertionError("fiber rate proxy must not increase with distance")
    if max(qber_errors) > 1e-12 or max(rate_errors) > 1e-12:
        raise AssertionError("independent QBER/rate transcription differs from code")

    anchor_errors = []
    for season in SEASONS:
        for distance, expected in zip((10.0, 15.0, 30.0, 78.0),
                                      FSO_VIABILITY_ANCHORS[season]):
            actual = fso_viability_probability(distance, season)
            anchor_errors.append(abs(actual - expected))
    if max(anchor_errors) > 1e-12:
        raise AssertionError("FSO interpolation does not reproduce its configured anchors")

    rng = random.Random(20261002)
    log_samples = []
    previous = None
    for _ in range(20_000):
        _, previous = sample_correlated_fso_cn2("normal", 22.0, rng, previous)
        log_samples.append(previous)
    empirical_correlation = float(np.corrcoef(log_samples[:-1], log_samples[1:])[0, 1])
    standard_error = (1.0 - FSO_TURBULENCE_CORRELATION ** 2) / math.sqrt(len(log_samples))
    if abs(empirical_correlation - FSO_TURBULENCE_CORRELATION) > 5 * standard_error:
        raise AssertionError("sampled FSO AR(1) correlation differs from configured target")

    report = {
        "status": "internal_analytical_consistency_only",
        "external_measurement_validation": "not_performed_no_matched_trace_or_device_dataset",
        "seed": 20261002,
        "fiber_distance_sweep_km": [0, 200, 1],
        "fiber_sweep_points": len(rows),
        "independent_formula_max_abs_error": {
            "qber": max(qber_errors),
            "normalized_rate_proxy": max(rate_errors)},
        "monotonic_checks": {"transmission_nonincreasing": True,
            "qber_nondecreasing": True, "rate_proxy_nonincreasing": True},
        "fso_availability_anchor_max_abs_error": max(anchor_errors),
        "fso_log_cn2_ar1": {"configured": FSO_TURBULENCE_CORRELATION,
            "sampled": empirical_correlation,
            "samples": len(log_samples), "five_se_tolerance": 5 * standard_error},
        "scope_note": "Checks code against an independent transcription of its own assumptions; does not establish physical accuracy or QKD security.",
        "fiber_samples": [rows[index] for index in (0, 10, 40, 80, 120, 160, 200)],
    }
    output = ROOT / "paper" / "supplementary" / "physics_analytical_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
