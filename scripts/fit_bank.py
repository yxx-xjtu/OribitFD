"""Fit a frozen OrbitFD source bank from source-fit and source-calibration feature NPZs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import fit_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit frozen OrbitBank from source-only feature NPZs.")
    parser.add_argument("--fit", required=True, help="Source training feature .npz (z, y).")
    parser.add_argument("--cal", required=True, help="Disjoint source calibration feature .npz (z, y[, context_id]).")
    parser.add_argument("--out", required=True, help="Output bank .npz path.")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--semantic-penalty", type=float, default=0.2)
    parser.add_argument("--normalization-shrinkage", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--radius-multiplier", type=float, default=1.5)
    parser.add_argument("--context-drift-allowance", type=float, default=0.2)
    args = parser.parse_args()

    if Path(args.fit).resolve() == Path(args.cal).resolve():
        parser.error("Fit and calibration must be disjoint files.")
    with np.load(args.fit, allow_pickle=False) as f, np.load(args.cal, allow_pickle=False) as c:
        if "record_id" in f and "record_id" in c and np.intersect1d(f["record_id"], c["record_id"]).size:
            parser.error("Fit and calibration record_id arrays overlap.")
        context_calibrated = "context_id" in c
        bank = fit_bank(
            f["z"],
            f["y"],
            c["z"],
            c["y"],
            rank=args.rank,
            semantic_penalty=args.semantic_penalty,
            normalization_shrinkage=args.normalization_shrinkage,
            alpha=args.alpha,
            radius_multiplier=args.radius_multiplier,
            calibration_context=c["context_id"] if context_calibrated else None,
            context_drift_allowance=args.context_drift_allowance,
        )

    bank.save(args.out)
    summary = {k: v for k, v in bank.__dict__.items() if not hasattr(v, "shape")}
    summary.update(
        {
            "rank": bank.rank,
            "num_classes": bank.num_classes,
            "context_calibrated": context_calibrated,
            "arguments": vars(args),
        }
    )
    Path(str(args.out) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
