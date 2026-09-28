"""Calibration utilities for measured FSO turbulence traces."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable


def fit_log_cn2_ar1(values: Iterable[float]) -> dict[str, float | int]:
    """Fit log-normal marginal parameters and AR(1) correlation from Cn2 data."""
    values = [float(v) for v in values if float(v) > 0]
    if len(values) < 3:
        raise ValueError("at least three positive Cn2 samples are required")
    logs = [math.log(v) for v in values]
    mean = statistics.fmean(logs)
    sigma = statistics.pstdev(logs)
    numerator = sum((logs[i - 1] - mean) * (logs[i] - mean)
                    for i in range(1, len(logs)))
    denominator = sum((v - mean) ** 2 for v in logs[:-1])
    correlation = numerator / denominator if denominator else 0.0
    return {"samples": len(values), "log_cn2_mean": mean,
            "log_cn2_sigma": sigma, "ar1_correlation": max(-1.0, min(1.0, correlation)),
            "cn2_geometric_mean": math.exp(mean)}


def fit_csv(path: str | Path, *, cn2_column: str = "cn2") -> dict:
    path = Path(path)
    with path.open(newline="", encoding="utf8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or cn2_column not in rows.fieldnames:
            raise ValueError(f"CSV must contain a {cn2_column!r} column")
        report = fit_log_cn2_ar1(float(row[cn2_column]) for row in rows)
    report.update({"input": str(path), "source_status": "measured_trace_pending_review"})
    return report
