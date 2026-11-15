"""Run causal block-stream inference using a frozen OrbitBank."""
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

from orbitfd import OnlineConfig, OrbitBank, benchmark_metrics, evaluate_stream, open_set_metrics
from orbitfd.stream import make_global_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological source-free OrbitFD stream deployment.")
    parser.add_argument("--bank", required=True, help="Path to frozen OrbitBank .npz.")
    parser.add_argument("--target", required=True, help="Chronological target feature .npz (z[, block_id, y]).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--mode",
        choices=["online", "frozen_local", "frozen_global", "static", "normal_only", "ungated"],
        default="online",
    )
    parser.add_argument("--cal", help="Calibration feature .npz (required only when --mode frozen_global).")
    parser.add_argument("--block-size", type=int, default=32, help="Block length (default: 32).")
    parser.add_argument("--config", help="Optional JSON file overriding OnlineConfig fields.")
    args = parser.parse_args()

    if args.block_size < 1:
        parser.error("block-size must be positive.")
    torch.set_num_threads(1)
    bank = OrbitBank.load(args.bank)
    config = OnlineConfig(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else OnlineConfig()

    global_bank = None
    if args.mode == "frozen_global":
        if not args.cal:
            parser.error("--cal is required when --mode frozen_global.")
        with np.load(args.cal, allow_pickle=False) as cf:
            global_bank = make_global_bank(bank, cf["z"], cf["y"])

    with np.load(args.target, allow_pickle=False) as f:
        z = f["z"].copy()
        ids = f["block_id"].copy() if "block_id" in f else np.arange(len(z)) // args.block_size
        labels = f["y"].copy() if "y" in f else None
        if "timestamp" in f and np.any(np.diff(f["timestamp"]) < 0):
            parser.error("Target timestamps are not chronological. Do not shuffle the deployment stream.")

    result, logs = evaluate_stream(
        bank, z, ids, block_size=args.block_size, mode=args.mode, config=config, global_bank=global_bank
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", **result)
    (out / "updates.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps({"mode": args.mode, "block_size": args.block_size, **config.__dict__}, indent=2),
        encoding="utf-8",
    )

    if labels is not None:
        bm = benchmark_metrics(labels, result["forced_prediction"], result["score"])
        om = open_set_metrics(labels, result["forced_prediction"], result["label"], result["score"])
        metrics = {**om, "benchmark": bm, "committed_updates": int(sum(l["updated"] for l in logs))}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Saved unlabeled predictions and state update log to {out}")


if __name__ == "__main__":
    main()
