"""Validate model-derived FSO availability against configured anchors."""
from __future__ import annotations

import json
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
            trans = [s["conditions"]["transmittance"] for s in states]
            rows.append({"season": season, "distance_km": distance,
                "configured_availability": FSO_VIABILITY_ANCHORS[season][
                    {10.0: 0, 15.0: 1, 30.0: 2, 78.0: 3}[distance]],
                "sampled_availability": availability,
                "availability_error": availability - FSO_VIABILITY_ANCHORS[season][
                    {10.0: 0, 15.0: 1, 30.0: 2, 78.0: 3}[distance]],
                "median_transmittance": float(np.median(trans)),
                "p05_transmittance": float(np.percentile(trans, 5)),
                "p95_transmittance": float(np.percentile(trans, 95))})
    report = {"status": "model_internal_validation_only",
              "external_trace_calibration": "not_available",
              "rows": rows}
    output = ROOT / "experiments" / "fso_model_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"rows": len(rows), "max_abs_anchor_error": max(
        abs(r["availability_error"]) for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
