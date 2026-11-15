"""Source-only encoder training and full OrbitFD task pipeline.

Supports two workflows:
1. Benchmark task mode (Section IV-B / Supplementary Section III-A protocol):
   - 1,020 class-balanced source batches (batch size 64), 30 validation checks
   - Checkpoint selected by highest source validation macro-F1 (tie-break: lower CE)
   - Fits frozen OrbitBank on Train + Cal and evaluates chronological 32-window blocks
   Example:
     python scripts/train_backbone.py --prepared prepared --task-id T3-U1 --seed 0 --out runs/T3-U1_seed0

2. Standalone custom NPZ mode (for custom vibration datasets):
   - Expects source ``train.npz`` and ``val.npz`` with ``x`` [N, C, L] and ``y`` [N] in 0..K-1
   Example:
     python scripts/train_backbone.py --train source_train.npz --val source_val.npz --out runs/custom/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import OrbitOnline, SourceClassifier, benchmark_metrics, fit_bank, open_set_metrics


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def extract_all(model: SourceClassifier, x: torch.Tensor, device: str, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    zs, logits = [], []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        z = model.encoder(xb)
        zs.append(z.cpu().numpy())
        logits.append(model.head(z).cpu().numpy())
    return {"z": np.concatenate(zs), "logits": np.concatenate(logits)}


def make_balanced_loader(x: torch.Tensor, y: np.ndarray, total_batches: int, batch_size: int = 64) -> DataLoader:
    num_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=num_classes)
    weights = 1.0 / counts[y]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        total_batches * batch_size,
        replacement=True,
    )
    return DataLoader(TensorDataset(x, torch.from_numpy(y)), batch_size=batch_size, sampler=sampler, num_workers=0)


def run_benchmark_task(args: argparse.Namespace) -> None:
    set_deterministic_seed(args.seed)
    prepared = Path(args.prepared)
    manifest = json.loads((prepared / "tasks" / f"{args.task_id}.json").read_text(encoding="utf-8"))
    task = manifest["task"]
    known = task["id_classes"]
    all_x = np.load(prepared / f"{task['dataset']}.npy", mmap_mode="r")
    roles = manifest["roles"]

    train_raw = np.asarray(all_x[[r["row"] for r in roles["train"]]], dtype=np.float32)
    mean = train_raw.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = train_raw.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32).clip(1e-6)

    splits: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
    for role in ("train", "val", "cal"):
        rr = roles[role]
        raw = train_raw if role == "train" else np.asarray(all_x[[r["row"] for r in rr]], dtype=np.float32)
        yy = np.array([known.index(r["label"]) for r in rr], dtype=np.int64)
        splits[role] = (torch.from_numpy(((raw - mean) / scale).astype(np.float32)), yy)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = splits["train"][0].shape[1]
    num_classes = len(known)
    config = {"channels": int(channels), "feature_dim": args.feature_dim, "num_classes": num_classes}
    model = SourceClassifier(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps = args.steps
    check_period = max(1, steps // 30)
    iterator = iter(make_balanced_loader(splits["train"][0], splits["train"][1], steps, args.batch_size))
    best_score = (-1.0, -float("inf"))
    history = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        xb, yb = next(iterator)
        xb, yb = xb.to(args.device), yb.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()

        if step % check_period == 0 or step == steps:
            val_out = extract_all(model, splits["val"][0], args.device)
            yv = splits["val"][1]
            val_f1 = float(
                f1_score(yv, val_out["logits"].argmax(1), labels=list(range(num_classes)), average="macro", zero_division=0)
            )
            val_ce = float(F.cross_entropy(torch.from_numpy(val_out["logits"]), torch.from_numpy(yv)))
            row = {
                "source_batch": step,
                "train_loss": float(loss.detach().cpu()),
                "source_val_f1": val_f1,
                "source_val_ce": val_ce,
                "elapsed_seconds": time.perf_counter() - t0,
            }
            history.append(row)
            if (val_f1, -val_ce) > best_score:
                best_score = (val_f1, -val_ce)
                torch.save(
                    {
                        "config": config,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "normalization_mean": mean,
                        "normalization_scale": scale,
                        "selection": row,
                        "seed": args.seed,
                    },
                    out_dir / "best.pt",
                )

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    train_feat = extract_all(model, splits["train"][0], args.device)
    cal_feat = extract_all(model, splits["cal"][0], args.device)
    cal_context = np.array([r["record_id"] for r in roles["cal"]])
    bank = fit_bank(
        train_feat["z"],
        splits["train"][1],
        cal_feat["z"],
        splits["cal"][1],
        rank=2,
        semantic_penalty=0.2,
        calibration_context=cal_context,
    )
    bank.save(out_dir / "bank.npz")
    np.savez_compressed(out_dir / "train_features.npz", **train_feat, y=splits["train"][1], record_id=np.array([r["record_id"] for r in roles["train"]]))
    np.savez_compressed(out_dir / "cal_features.npz", **cal_feat, y=splits["cal"][1], context_id=cal_context)

    for pool in ("dev", "eval"):
        rr = roles[pool]
        raw_pool = np.asarray(all_x[[r["row"] for r in rr]], dtype=np.float32)
        x_pool = torch.from_numpy(((raw_pool - mean) / scale).astype(np.float32))
        feat_pool = extract_all(model, x_pool, args.device)
        engine = OrbitOnline(bank)
        preds, logs = [], []
        for start in range(0, len(feat_pool["z"]), 32):
            step_out = engine.step(feat_pool["z"][start : start + 32])
            preds.append({k: step_out[k].cpu().numpy() for k in ("score", "forced_prediction", "label")})
            logs.append(
                {
                    "start": start,
                    "end": min(start + 32, len(feat_pool["z"])),
                    "updated": bool(step_out["updated"]),
                    "reason": step_out["reason"],
                    "state_used": step_out["state_used"].cpu().tolist(),
                    "state_after": step_out["state_after"].cpu().tolist(),
                }
            )
        merged = {k: np.concatenate([p[k] for p in preds]) for k in preds[0]}
        y_true = np.array([known.index(r["label"]) if r["label"] in known else -1 for r in rr], dtype=np.int64)
        bm = benchmark_metrics(y_true, merged["forced_prediction"], merged["score"])
        om = open_set_metrics(y_true, merged["forced_prediction"], merged["label"], merged["score"])
        summary = {
            "task_id": args.task_id,
            "pool": pool,
            "seed": args.seed,
            **bm,
            "known_correct_acceptance": om["known_correct_acceptance"],
            "unknown_rejection": om["unknown_rejection_including_defer"],
            "open_set_macro_f1": om["open_set_macro_f1"],
            "committed_updates": sum(l["updated"] for l in logs),
            "blocks": len(logs),
        }
        np.savez_compressed(
            out_dir / f"{pool}_predictions.npz",
            **merged,
            z=feat_pool["z"],
            logits=feat_pool["logits"],
            y=y_true,
        )
        (out_dir / f"{pool}_updates.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        (out_dir / f"{pool}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if pool == "eval":
            print(
                f"[{args.task_id} seed={args.seed} eval] "
                f"Known Acc={bm['known_acc']*100:.2f}% | AUROC={bm['auroc']*100:.2f}% | "
                f"AUPR={bm['aupr']*100:.2f}% | FPR95={bm['fpr95']*100:.2f}% | "
                f"Commits={summary['committed_updates']}"
            )


def read_npz_signals(path: str):
    with np.load(path, allow_pickle=False) as f:
        x = np.asarray(f["x"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        records = f["record_id"].copy() if "record_id" in f else None
    if x.ndim != 3 or y.shape != (len(x),) or not np.isfinite(x).all() or (y < 0).any():
        raise ValueError("Source NPZ requires finite x [N, C, L] and y [N] in 0..K-1.")
    return x, y, records


def run_standalone_npz(args: argparse.Namespace) -> None:
    if Path(args.train).resolve() == Path(args.val).resolve():
        raise ValueError("Training and validation files must be disjoint splits.")
    set_deterministic_seed(args.seed)
    x, y, records = read_npz_signals(args.train)
    xv, yv, records_v = read_npz_signals(args.val)
    if records is not None and records_v is not None and np.intersect1d(records, records_v).size:
        raise ValueError("Train and validation record_id arrays overlap.")
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))) or not np.array_equal(np.unique(yv), classes):
        raise ValueError("Train and validation must share contiguous labels 0..K-1.")

    config = {"channels": int(x.shape[1]), "feature_dim": args.feature_dim, "num_classes": len(classes)}
    model = SourceClassifier(**config).to(args.device)
    with torch.no_grad():
        model.encoder.input_mean.copy_(torch.from_numpy(x.mean(axis=(0, 2), keepdims=True)))
        model.encoder.input_scale.copy_(torch.from_numpy(x.std(axis=(0, 2), keepdims=True).clip(1e-6)))

    steps = args.steps
    check_period = max(1, steps // 30)
    iterator = iter(make_balanced_loader(torch.from_numpy(x), y, steps, args.batch_size))
    xv_tensor = torch.from_numpy(xv)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = (-1.0, -float("inf"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps + 1):
        model.train()
        xb, yb = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb.to(args.device)), yb.to(args.device))
        loss.backward()
        optimizer.step()
        if step % check_period == 0 or step == steps:
            val_out = extract_all(model, xv_tensor, args.device)
            val_f1 = float(
                f1_score(yv, val_out["logits"].argmax(1), labels=list(range(len(classes))), average="macro", zero_division=0)
            )
            val_ce = float(F.cross_entropy(torch.from_numpy(val_out["logits"]), torch.from_numpy(yv)))
            if (val_f1, -val_ce) > best_score:
                best_score = (val_f1, -val_ce)
                torch.save(
                    {
                        "config": config,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "source_val_f1": val_f1,
                        "source_val_ce": val_ce,
                        "seed": args.seed,
                    },
                    out_path,
                )
            print(f"step={step:4d}/{steps} val_macro_f1={val_f1:.4f} val_ce={val_ce:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OrbitFD source-only encoder.")
    parser.add_argument("--prepared", help="Prepared benchmark directory (from prepare_benchmarks.py).")
    parser.add_argument("--task-id", help="Benchmark task ID, e.g., T3-U1.")
    parser.add_argument("--train", help="Standalone mode: source training .npz.")
    parser.add_argument("--val", help="Standalone mode: source validation .npz.")
    parser.add_argument("--out", required=True, help="Output run directory (benchmark mode) or .pt path (NPZ mode).")
    parser.add_argument("--steps", type=int, default=1020, help="Source training batches (default: 1020).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.prepared and args.task_id:
        run_benchmark_task(args)
    elif args.train and args.val:
        run_standalone_npz(args)
    else:
        parser.error("Provide either (--prepared and --task-id) OR (--train and --val).")


if __name__ == "__main__":
    main()
