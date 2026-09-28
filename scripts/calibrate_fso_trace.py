"""Fit FSO log-Cn2/temporal-correlation parameters from a measured CSV trace."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration import fit_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV containing positive Cn2 samples")
    parser.add_argument("--output", default="experiments/fso_calibration_report.json")
    parser.add_argument("--column", default="cn2")
    args = parser.parse_args()
    report = fit_csv(args.input, cn2_column=args.column)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
