"""Synthetic sanity checks and structural counterexample demo for OrbitFD.

Runs completely standalone on CPU in a few seconds without external datasets.
Demonstrates:
1. Off-orbit unknown rejection under smooth operating-condition drift.
2. Unknown-only stream resilience (zero false state commits).
3. Single-class known-stream state tracking.
4. Expected geometric limitation on deliberately on-orbit unknown bursts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import evaluate_stream, fit_bank, open_set_metrics


def make_data(seed: int = 7, d: int = 16, block_size: int = 32, blocks: int = 64):
    rng = np.random.default_rng(seed)
    orth, _ = np.linalg.qr(rng.normal(size=(d, d)))
    basis_true = orth[:, :2]
    centers = np.stack(
        [4.0 * (np.cos(a) * orth[:, 2] + np.sin(a) * orth[:, 3]) for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    )

    def source(n: int):
        y = rng.integers(0, 3, size=n)
        a = rng.normal(size=(n, 2))
        z = centers[y] + a @ basis_true.T + 0.07 * rng.normal(size=(n, d))
        return z, y

    fit_z, fit_y = source(3600)
    cal_zs, cal_ys, cal_context = [], [], []
    for group in range(72):
        labels = np.full(32, group % 3, dtype=int)
        a = rng.normal(size=2)
        z = (
            centers[labels]
            + (a[None, :] + 0.10 * rng.normal(size=(32, 2))) @ basis_true.T
            + 0.07 * rng.normal(size=(32, d))
        )
        cal_zs.append(z)
        cal_ys.append(labels)
        cal_context.append(np.full(32, group))
    cal_z, cal_y = np.concatenate(cal_zs), np.concatenate(cal_ys)
    cal_context_arr = np.concatenate(cal_context)

    streams = {}
    scenarios = (
        "off_orbit_unknown",
        "on_orbit_unknown",
        "abrupt_on_orbit_unknown",
        "unknown_only",
        "single_class_known",
    )
    for scenario in scenarios:
        zs, ys, ids = [], [], []
        for t in range(blocks):
            a = np.array([2.6 * np.sin(np.pi * t / (blocks - 1)), 0.6 * np.sin(2 * np.pi * t / (blocks - 1))])
            y = np.full(block_size, (t // 12) % 3, dtype=int)
            z = (
                centers[y]
                + (a[None, :] + 0.10 * rng.normal(size=(block_size, 2))) @ basis_true.T
                + 0.07 * rng.normal(size=(block_size, d))
            )
            if scenario == "off_orbit_unknown":
                mask = rng.random(block_size) < (0.65 if 24 <= t < 32 else 0.15)
                z[mask] += 3.2 * orth[:, 4]
                y[mask] = -1
            elif scenario in ("on_orbit_unknown", "abrupt_on_orbit_unknown") and 24 <= t < 40:
                offset = 0.8 * min(1.0, (t - 23) / 8) if scenario == "on_orbit_unknown" else 0.8
                z = (
                    centers[0]
                    + (a[None, :] + np.array([offset, 0.0]) + 0.10 * rng.normal(size=(block_size, 2)))
                    @ basis_true.T
                    + 0.07 * rng.normal(size=(block_size, d))
                )
                y[:] = -1
            elif scenario == "unknown_only":
                z += 3.2 * orth[:, 4]
                y[:] = -1
            zs.append(z)
            ys.append(y)
            ids.append(np.full(block_size, t))
        streams[scenario] = (np.concatenate(zs), np.concatenate(ys), np.concatenate(ids))
    return (fit_z, fit_y, cal_z, cal_y), streams, cal_context_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OrbitFD synthetic sanity checks.")
    parser.add_argument("--out", default="results/demo", help="Output directory for demo artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, streams, cal_context = make_data(seed=args.seed, block_size=args.block_size)
    bank = fit_bank(*data, rank=2, semantic_penalty=0.2, calibration_context=cal_context)
    bank.save(out / "bank.npz")

    results = {
        "notice": "Synthetic feature sanity checks (including expected on-orbit counterexamples).",
        "seed": args.seed,
        "block_size": args.block_size,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "within_energy": bank.within_energy,
        "between_energy": bank.between_energy,
        "scenarios": {},
    }
    for scenario, (z, y, ids) in streams.items():
        results["scenarios"][scenario] = {}
        for mode in ("static", "normal_only", "frozen_local", "ungated", "online"):
            prediction, logs = evaluate_stream(bank, z, ids, mode=mode)
            metrics = open_set_metrics(y, prediction["forced_prediction"], prediction["label"], prediction["score"])
            metrics["committed_updates"] = int(sum(log["updated"] for log in logs))
            metrics["mean_state_step"] = float(
                np.mean([np.linalg.norm(np.asarray(log["state_after"]) - log["state_used"]) for log in logs])
            )
            results["scenarios"][scenario][mode] = metrics
        m = results["scenarios"][scenario]["online"]
        print(
            f"[{scenario:25s}] AUROC={str(round(m['auroc_unknown_positive'], 4)) if m['auroc_unknown_positive'] is not None else 'N/A':>6s} | "
            f"KCA={str(round(m['known_correct_acceptance'], 4)) if m['known_correct_acceptance'] is not None else 'N/A':>6s} | "
            f"U-Reject={str(round(m['unknown_rejection_including_defer'], 4)) if m['unknown_rejection_including_defer'] is not None else 'N/A':>6s} | "
            f"Commits={m['committed_updates']}"
        )

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved synthetic evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
