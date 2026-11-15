"""Cross-platform single-thread CPU latency, memory footprint, and FLOPs profiler (E4).

Benchmarks the native deployment OrbitFD pipeline (FP32 WaveEncoder + FP64 OrbitOnline)
on 32-window blocks using a single CPU thread without GPU acceleration.
Works out-of-the-box with synthetic streams or a trained run directory.
"""
from __future__ import annotations

import os

for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_env_key] = "1"

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import OrbitBank, OrbitOnline, WaveEncoder, fit_bank


def count_encoder_macs(encoder: nn.Module, x: torch.Tensor) -> int:
    total_macs = 0
    handles = []

    def hook(m: nn.Module, _inp, out: torch.Tensor) -> None:
        nonlocal total_macs
        if isinstance(m, nn.Conv1d):
            total_macs += int(out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size))
        elif isinstance(m, nn.Linear):
            total_macs += int(out.numel() * m.in_features)

    for module in encoder.modules():
        handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        encoder(x)
    for h in handles:
        h.remove()
    return total_macs


def bank_storage_bytes(bank: OrbitBank) -> int:
    total = 0
    for k in ("mean", "scale", "prototypes", "basis", "coordinate_scale"):
        t = getattr(bank, k)
        total += int(t.numel() * t.element_size())
    # 7 FP64 scalar thresholds/metadata
    total += 7 * 8
    return total


def summarize_ms(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_blocks": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark single-thread CPU latency and FLOPs for OrbitFD.")
    parser.add_argument("--channels", type=int, choices=[1, 3], default=1, help="1 (JNU/CWRU/SelfA) or 3 (HUST).")
    parser.add_argument("--blocks", type=int, default=64, help="Number of 32-window blocks per stream repeat.")
    parser.add_argument("--repeats", type=int, default=3, help="Full-stream timing repeats.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    encoder = WaveEncoder(channels=args.channels, feature_dim=64).cpu().eval()
    params = sum(p.numel() for p in encoder.parameters())
    macs_per_window = count_encoder_macs(encoder, torch.zeros(1, args.channels, 1024))
    flops_per_window = 2 * macs_per_window

    rng = np.random.default_rng(0)
    fit_z = rng.normal(size=(1200, 64))
    fit_y = rng.integers(0, 3, size=1200)
    cal_z = rng.normal(size=(384, 64))
    cal_y = np.tile(np.arange(3), 128)
    cal_ctx = np.repeat(np.arange(12), 32)
    bank = fit_bank(fit_z, fit_y, cal_z, cal_y, rank=2, semantic_penalty=0.2, calibration_context=cal_ctx)

    stream_x = torch.from_numpy(
        rng.normal(size=(args.blocks * 32, args.channels, 1024)).astype(np.float32)
    )

    # Warm-up pass
    warm_engine = OrbitOnline(bank)
    for start in range(0, min(len(stream_x), 10 * 32), 32):
        z = encoder(stream_x[start : start + 32]).numpy()
        warm_engine.step(z)

    enc_times, eng_times, tot_times = [], [], []
    for _ in range(args.repeats):
        engine = OrbitOnline(bank)
        for start in range(0, len(stream_x), 32):
            xb = stream_x[start : start + 32]
            t0 = time.perf_counter_ns()
            z = encoder(xb).numpy()
            t1 = time.perf_counter_ns()
            engine.step(z)
            t2 = time.perf_counter_ns()
            enc_times.append((t1 - t0) / 1e6)
            eng_times.append((t2 - t1) / 1e6)
            tot_times.append((t2 - t0) / 1e6)

    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "channels": args.channels,
        "deployment_parameters": int(params),
        "macs_per_window": int(macs_per_window),
        "flops_per_window": int(flops_per_window),
        "source_bank_bytes": bank_storage_bytes(bank),
        "shared_state_bytes": int(bank.rank * 8),
        "block_size": 32,
        "encoding_latency": summarize_ms(enc_times),
        "online_engine_latency": summarize_ms(eng_times),
        "total_block_latency": summarize_ms(tot_times),
        "amortized_ms_per_window": float(np.mean(tot_times) / 32.0),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
