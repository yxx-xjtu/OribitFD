"""Prepare non-overlapping Train/Val/Cal/Audit/Dev/Eval splits for JNU, HUST, and CWRU.

Implements the exact temporal partitioning protocol from Section IV-A and
Supplementary Section II-A/B:
- Source recordings: 60% Train / 15% Val / 15% Cal / 10% Audit
- Target recordings: 30% Dev / 70% Eval
- 1,024 raw points are purged on both sides of every internal split boundary
  prior to non-overlapping 1,024-point window extraction.
- Enforces zero raw-sample overlap and zero content hash collision across
  selected roles for every task instance (T1-U1 .. T7-U3).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

WINDOW_LENGTH = 1024
ROLES = ("train", "val", "cal", "audit", "dev", "eval")

PUBLIC_GROUPS = [
    ("T1", "JNU", [600, 1000], [800], "Interpolation: 600, 1000 -> 800 rpm"),
    ("T2", "HUST", [20, 40, 60, 80], [25, 30, 35, 65, 70, 75], "Interpolation: 20, 40, 60, 80 -> 25..75 Hz"),
    ("T3", "CWRU48", [0, 1, 3], [2], "Interpolation: 0, 1, 3 -> 2 HP"),
    ("T4", "JNU", [600, 800], [1000], "Extrapolation (high-speed): 600, 800 -> 1000 rpm"),
    ("T5", "HUST", [30, 35, 40, 60, 65, 70], [20, 25], "Extrapolation (low-speed): 30..70 -> 20, 25 Hz"),
    ("T6", "CWRU48", [0, 1, 2], [3], "Extrapolation (high-load): 0, 1, 2 -> 3 HP"),
    ("T7", "CWRU48", [1, 2, 3], [0], "Extrapolation (low-load): 1, 2, 3 -> 0 HP"),
]


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def role_intervals(n: int, g: int = WINDOW_LENGTH) -> dict[str, tuple[int, int]]:
    a, b, c, d = int(0.60 * n), int(0.75 * n), int(0.90 * n), int(0.30 * n)
    return {
        "train": (0, a - g),
        "val": (a + g, b - g),
        "cal": (b + g, c - g),
        "audit": (c + g, n),
        "dev": (0, d - g),
        "eval": (d + g, n),
    }


def validate_task_selection(selected: dict[str, list[dict]]) -> None:
    by_record: dict[str, list[tuple[int, int, str]]] = {}
    hashes: dict[str, str] = {}
    for role, rows in selected.items():
        for r in rows:
            by_record.setdefault(r["record_id"], []).append((r["start"], r["end"], role))
            old = hashes.setdefault(r["content_sha256"], role)
            if old != role:
                raise ValueError(f"Duplicate window content across roles: {old} vs {role}")
    for rid, spans in by_record.items():
        spans.sort()
        for s1, s2 in zip(spans, spans[1:]):
            if s1[1] > s2[0]:
                raise ValueError(f"Overlapping raw interval in {rid}: {s1} vs {s2}")


def scan_dataset_entries(dataset: str, raw_dir: Path) -> list[dict]:
    entries: list[dict] = []
    if dataset == "CWRU48":
        reg_path = raw_dir / "registry.json"
        if not reg_path.exists():
            raise FileNotFoundError(
                f"Missing {reg_path}. Run `python scripts/download_cwru.py --out {raw_dir}` first."
            )
        return json.loads(reg_path.read_text(encoding="utf-8"))
    if dataset == "JNU":
        for p in sorted(raw_dir.glob("*.csv")):
            m = re.fullmatch(r"(n|ib|ob|tb)(600|800|1000).*\.csv", p.name)
            if not m:
                continue
            cls = {"n": 0, "ib": 1, "ob": 2, "tb": 3}[m[1]]
            entries.append(
                {
                    "file": p.name,
                    "label": cls,
                    "condition": int(m[2]),
                    "severity": "healthy" if cls == 0 else "fixed",
                    "sampling_rate": 50000,
                }
            )
        if len(entries) != 12:
            raise ValueError(f"Expected 12 JNU .csv files in {raw_dir}, found {len(entries)}.")
        return entries
    if dataset == "HUST":
        for p in sorted(raw_dir.glob("*.xls")):
            m = re.fullmatch(r"(0\.5X_)?([HIOB])_(\d+)[Hh][Zz]\.xls", p.name)
            if not m:
                continue
            cls = {"H": 0, "I": 1, "O": 2, "B": 3}[m[2]]
            entries.append(
                {
                    "file": p.name,
                    "label": cls,
                    "condition": int(m[3]),
                    "severity": "healthy" if cls == 0 else ("medium" if m[1] else "severe"),
                    "sampling_rate": 25600,
                }
            )
        if len(entries) != 70:
            raise ValueError(f"Expected 70 HUST .xls files in {raw_dir}, found {len(entries)}.")
        return entries
    raise ValueError(f"Unsupported public dataset: {dataset}")


def load_raw_signal(dataset: str, raw_dir: Path, entry: dict) -> np.ndarray:
    path = raw_dir / entry["file"]
    if "sha256" in entry and sha256_file(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    if dataset == "JNU":
        return np.loadtxt(path, dtype=np.float32).reshape(1, -1)
    if dataset == "CWRU48":
        mat = loadmat(path, variable_names=[entry["channel"]])
        return mat[entry["channel"]].reshape(1, -1).astype(np.float32)
    if dataset == "HUST":
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        start = next(i for i, s in enumerate(lines) if s.strip() == "Data") + 1
        arr = np.loadtxt(io.StringIO("\n".join(lines[start:])), usecols=(0, 2, 3, 4), dtype=np.float64)
        if arr.shape != (262144, 4) or not np.all(np.diff(arr[:, 0]) > 0):
            raise ValueError(f"Unexpected HUST raw array shape or timestamps in {path}")
        return arr[:, 1:].T.astype(np.float32)
    raise ValueError(dataset)


def prepare_dataset(dataset: str, raw_dir: Path, out_dir: Path) -> None:
    entries = scan_dataset_entries(dataset, raw_dir)
    windows: list[np.ndarray] = []
    rows: list[dict] = []
    for entry in entries:
        path = raw_dir / entry["file"]
        raw_hash = sha256_file(path)
        sig = load_raw_signal(dataset, raw_dir, entry)
        if not np.isfinite(sig).all():
            raise ValueError(f"Non-finite values in {path}")
        rid = f"{dataset}:{path.stem}"
        for role, (lo, hi) in role_intervals(sig.shape[1]).items():
            for start in range(lo, hi - WINDOW_LENGTH + 1, WINDOW_LENGTH):
                win = sig[:, start : start + WINDOW_LENGTH].copy()
                idx = len(rows)
                windows.append(win)
                rows.append(
                    {
                        "row": idx,
                        "record_id": rid,
                        "raw_sha256": raw_hash,
                        "start": start,
                        "end": start + WINDOW_LENGTH,
                        "label": entry["label"],
                        "condition": entry["condition"],
                        "severity": entry["severity"],
                        "role": role,
                        "content_sha256": hashlib.sha256(win.tobytes()).hexdigest(),
                        "sample_id": f"{rid}:{start}:{start + WINDOW_LENGTH}",
                    }
                )
    arr = np.stack(windows).astype(np.float32)
    np.save(out_dir / f"{dataset}.npy", arr)
    dump_json(out_dir / f"{dataset}_records.json", rows)
    print(f"Prepared {dataset}: {len(entries)} files -> {len(rows)} role windows ({arr.shape})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public open-set bearing benchmarks (T1-T7).")
    parser.add_argument("--out", default="prepared", help="Output prepared directory.")
    parser.add_argument("--jnu", type=Path, help="Path to raw JNU directory (.csv files).")
    parser.add_argument("--hust", type=Path, help="Path to raw HUST directory (.xls files).")
    parser.add_argument("--cwru", type=Path, help="Path to raw CWRU48 directory (with registry.json).")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provided = {"JNU": args.jnu, "HUST": args.hust, "CWRU48": args.cwru}
    active_datasets = {ds for ds, p in provided.items() if p is not None}
    if not active_datasets:
        parser.error("Provide at least one dataset path via --jnu, --hust, or --cwru.")

    for ds in sorted(active_datasets):
        prepare_dataset(ds, provided[ds], out)

    manifest_summary = []
    for group_id, dataset, source_conds, target_conds, desc in PUBLIC_GROUPS:
        if dataset not in active_datasets:
            continue
        rows = json.loads((out / f"{dataset}_records.json").read_text(encoding="utf-8"))
        for u_idx, unknown_cls in enumerate((1, 2, 3), start=1):
            task_id = f"{group_id}-U{u_idx}"
            known_classes = [c for c in range(4) if c != unknown_cls]
            selected: dict[str, list[dict]] = {}
            for role in ROLES:
                is_target = role in ("dev", "eval")
                selected[role] = sorted(
                    [
                        r
                        for r in rows
                        if r["role"] == role
                        and r["condition"] in (target_conds if is_target else source_conds)
                        and (is_target or r["label"] in known_classes)
                    ],
                    key=lambda r: (r["condition"], r["record_id"], r["start"]),
                )
            validate_task_selection(selected)
            counts = {role: len(rs) for role, rs in selected.items()}
            counts["dev_id"] = sum(r["label"] in known_classes for r in selected["dev"])
            counts["dev_ood"] = sum(r["label"] == unknown_cls for r in selected["dev"])
            counts["eval_id"] = sum(r["label"] in known_classes for r in selected["eval"])
            counts["eval_ood"] = sum(r["label"] == unknown_cls for r in selected["eval"])
            task_meta = {
                "task_id": task_id,
                "group_id": group_id,
                "dataset": dataset,
                "description": desc,
                "id_classes": known_classes,
                "ood_classes": [unknown_cls],
                "train": {"conditions": source_conds},
                "val": {"conditions": source_conds},
                "cal": {"conditions": source_conds},
                "test": {"conditions": target_conds},
                "counts": counts,
            }
            dump_json(out / "tasks" / f"{task_id}.json", {"task": task_meta, "counts": counts, "roles": selected})
            manifest_summary.append({"task_id": task_id, "dataset": dataset, "counts": counts})
            print(
                f"  Task {task_id}: train={counts['train']}, val={counts['val']}, cal={counts['cal']}, "
                f"eval_id={counts['eval_id']}, eval_ood={counts['eval_ood']}"
            )

    dump_json(out / "READY.json", {"prepared_datasets": sorted(active_datasets), "tasks": manifest_summary})
    print(f"All requested task manifests written to {out / 'tasks'}")


if __name__ == "__main__":
    main()
