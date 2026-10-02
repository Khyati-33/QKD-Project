"""Validate model-derived FSO availability against configured anchors."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics import FSO_VIABILITY_ANCHORS, SEASONS, sample_link_state


def main() -> None:
    rows = []
    for season in SEASONS:
        for distance in (10.0, 15.0, 30.0, 78.0):
            rng = random.Random(20260928 + int(distance) + len(season))
            states = [sample_link_state("fso", distance, season, 22.0, rng=rng)
                      for _ in range(2000)]
            availability = float(np.mean([not s["outage"] for s in states]))
            target_availability = FSO_VIABILITY_ANCHORS[season][
                {10.0: 0, 15.0: 1, 30.0: 2, 78.0: 3}[distance]]
            availability_se = float(np.sqrt(target_availability *
                                            (1.0 - target_availability) / len(states)))
            trans = [s["conditions"]["transmittance"] for s in states]
            rows.append({"season": season, "distance_km": distance,
                "samples": len(states),
                "configured_availability": target_availability,
                "sampled_availability": availability,
                "availability_error": availability - target_availability,
                "target_binomial_standard_error": availability_se,
                "error_in_target_standard_errors": (
                    (availability - target_availability) / availability_se
                    if availability_se else None),
                "sampled_availability_approx_95pct_interval": [
                    max(0.0, availability - 1.96 * math.sqrt(
                        availability * (1.0 - availability) / len(states))),
                    min(1.0, availability + 1.96 * math.sqrt(
                        availability * (1.0 - availability) / len(states)))],
                "median_transmittance": float(np.median(trans)),
                "p05_transmittance": float(np.percentile(trans, 5)),
                "p95_transmittance": float(np.percentile(trans, 95))})
    report = {"status": "model_internal_validation_only",
              "external_trace_calibration": "not_available",
              "samples_per_condition": 2000,
              "rows": rows}
    output = ROOT / "experiments" / "fso_model_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"rows": len(rows), "samples_per_condition": 2000,
        "max_abs_anchor_error": max(abs(r["availability_error"]) for r in rows),
        "max_abs_error_in_target_se": max(abs(r["error_in_target_standard_errors"])
                                           for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
