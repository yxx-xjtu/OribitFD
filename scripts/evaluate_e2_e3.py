"""Fixed-feature scoring rules (E2-A) and mechanistic ablations (E3).

Evaluates alternative scoring and adaptation rules on identical frozen features
from a trained OrbitFD run directory (produced by ``scripts/train_backbone.py``):
- E2-A rules: MSP, Energy, Mahalanobis-LW, ViM, Static prototype, Online Orbit
- E3 rules:   Static prototype, Normal-only, Frozen-local Orbit,
              Frozen-global Orbit, Source-context union, Ungated causal, Online Orbit
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import OrbitBank, OrbitOnline, benchmark_metrics, open_set_metrics

ALL_RULES = (
    "msp",
    "energy",
    "mahalanobis_lw",
    "vim",
    "static",
    "normal_only",
    "frozen_local",
    "frozen_global",
    "source_union",
    "ungated",
    "online",
)


def fit_frozen_rules(
    bank: OrbitBank,
    train_features: np.ndarray,
    train_y: np.ndarray,
    train_logits: np.ndarray,
    cal_features: np.ndarray,
    cal_y: np.ndarray,
    cal_logits: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    record_ids: np.ndarray,
) -> dict:
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_y, dtype=int)
    means = np.stack([x[y == c].mean(0) for c in range(bank.num_classes)])
    lw = LedoitWolf(assume_centered=True).fit(x - means[y])

    u = -np.linalg.pinv(np.asarray(head_weight, dtype=np.float64)) @ np.asarray(head_bias, dtype=np.float64)
    xx = x - u
    _, eigenvectors = np.linalg.eigh(xx.T @ xx / len(xx))
    residual_basis = eigenvectors[:, : x.shape[1] // 2]
    residual = np.linalg.norm(xx @ residual_basis, axis=1)
    if residual.mean() <= 1e-12:
        raise ValueError("ViM source residual collapsed.")
    vim_alpha = float(np.asarray(train_logits).max(1).mean() / residual.mean())

    calibrated = bank.components(bank.normalize(cal_features), torch.zeros(bank.rank, dtype=torch.float64))
    cy = torch.as_tensor(cal_y, dtype=torch.long)
    idx = torch.arange(len(cy))
    global_bank = copy.deepcopy(bank)
    global_bank.tau_parallel = max(
        float(torch.quantile(calibrated["parallel"][idx, cy], 0.975, interpolation="higher")),
        1e-8,
    )

    source_coords = bank.components(bank.normalize(train_features), torch.zeros(bank.rank, dtype=torch.float64))[
        "coordinates"
    ]
    q = source_coords[torch.arange(len(y)), torch.from_numpy(y)].numpy()
    ids = np.asarray(record_ids)
    centers = np.stack([np.median(q[ids == r], axis=0) for r in sorted(set(ids))])
    n_clusters = min(3, len(centers))
    codebook = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(centers).cluster_centers_

    fitted = {
        "bank": bank,
        "global_bank": global_bank,
        "maha_means": means,
        "maha_precision": lw.precision_,
        "vim_origin": u,
        "vim_basis": residual_basis,
        "vim_alpha": vim_alpha,
        "codebook": codebook,
    }
    per = union_per_class(fitted, cal_features)
    fitted["union_threshold"] = max(
        float(np.quantile(per[np.arange(len(cy)), np.asarray(cal_y)], 0.95, method="higher")),
        1e-8,
    )
    fitted["generic_thresholds"] = {
        r: float(np.quantile(generic_score(fitted, r, cal_features, cal_logits), 0.95, method="higher"))
        for r in ("msp", "energy", "mahalanobis_lw", "vim")
    }
    return fitted


def generic_score(fitted: dict, rule: str, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if rule == "msp":
        return 1.0 - softmax(logits, axis=1).max(1)
    if rule == "energy":
        return -logsumexp(logits, axis=1)
    x = np.asarray(features, dtype=np.float64)
    if rule == "mahalanobis_lw":
        d = x[:, None] - fitted["maha_means"][None]
        return np.einsum("ncd,de,nce->nc", d, fitted["maha_precision"], d).min(1)
    if rule == "vim":
        norm = np.linalg.norm((x - fitted["vim_origin"]) @ fitted["vim_basis"], axis=1)
        return norm * fitted["vim_alpha"] - logsumexp(logits, axis=1)
    raise ValueError(rule)


def union_per_class(fitted: dict, features: np.ndarray) -> np.ndarray:
    b = fitted["bank"]
    comp = b.components(b.normalize(features), torch.zeros(b.rank, dtype=torch.float64))
    n = comp["normal"].numpy() / b.tau_normal
    p = (
        np.linalg.norm(comp["coordinates"].numpy()[:, :, None, :] - fitted["codebook"][None, None], axis=-1)
        / math.sqrt(b.rank)
        / b.tau_parallel
    )
    best_context = np.maximum(n[:, :, None], p).argmin(2)
    pp = np.take_along_axis(p, best_context[:, :, None], axis=2)[:, :, 0]
    return np.maximum(n, pp)


def rollout_rule(fitted: dict, rule: str, features: np.ndarray, logits: np.ndarray):
    if rule in ("msp", "energy", "mahalanobis_lw", "vim"):
        sc = generic_score(fitted, rule, features, logits)
        pred = np.asarray(logits).argmax(1)
        lbl = np.where(sc <= fitted["generic_thresholds"][rule], pred, -1)
        return {"score": sc, "forced_prediction": pred, "label": lbl}, []

    bank = fitted["bank"]
    engine = OrbitOnline(bank)
    parts, logs = [], []
    for start in range(0, len(features), 32):
        f = features[start : start + 32]
        if rule == "frozen_global":
            out = fitted["global_bank"].predict(f)
            out.update(updated=False, reason="frozen_global")
        else:
            out = engine.step(f, adapt=(rule == "online"))

        if rule == "static":
            comp = bank.components(bank.normalize(f), engine.state)
            sc, pr = (comp["full"] / bank.tau_static).min(1)
            out.update(score=sc, forced_prediction=pr, label=torch.where(sc <= 1.0, pr, -torch.ones_like(pr)))
        elif rule == "normal_only":
            sc, pr = out["normal_score"], out["normal_class"]
            out.update(score=sc, forced_prediction=pr, label=torch.where(sc <= 1.0, pr, -torch.ones_like(pr)))
        elif rule == "source_union":
            per = union_per_class(fitted, f) / fitted["union_threshold"]
            pr = per.argmin(1)
            sc = per[np.arange(len(f)), pr]
            out.update(
                score=torch.from_numpy(sc),
                forced_prediction=torch.from_numpy(pr),
                label=torch.from_numpy(np.where(sc <= 1.0, pr, -1)),
            )
        elif rule == "ungated":
            proposal = out["coordinate_observation"].mean(0)
            step = proposal - engine.state
            step *= min(1.0, engine.cfg.max_step / max(float(torch.linalg.vector_norm(step)), 1e-12))
            engine.state = engine.state + step
            out.update(updated=True, reason="ungated_mean_update")

        parts.append({k: out[k].cpu().numpy() for k in ("score", "forced_prediction", "label")})
        logs.append({"start": start, "end": start + len(f), "updated": bool(out["updated"]), "reason": out["reason"]})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2/E3 fixed-feature scoring and ablation controls.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by train_backbone.py.")
    parser.add_argument("--pool", choices=["dev", "eval"], default="eval")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bank = OrbitBank.load(run_dir / "bank.npz")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    head_w = ckpt["state_dict"]["head.weight"].numpy()
    head_b = ckpt["state_dict"]["head.bias"].numpy()

    with np.load(run_dir / "train_features.npz", allow_pickle=False) as tf, np.load(
        run_dir / "cal_features.npz", allow_pickle=False
    ) as cf, np.load(run_dir / f"{args.pool}_predictions.npz", allow_pickle=False) as pf:
        fitted = fit_frozen_rules(
            bank,
            tf["z"],
            tf["y"],
            tf["logits"],
            cf["z"],
            cf["y"],
            cf["logits"],
            head_w,
            head_b,
            tf["record_id"],
        )
        target_z, target_logits, y_true = pf["z"], pf["logits"], pf["y"]

    results = {}
    for rule in ALL_RULES:
        pred, logs = rollout_rule(fitted, rule, target_z, target_logits)
        bm = benchmark_metrics(y_true, pred["forced_prediction"], pred["score"])
        om = open_set_metrics(y_true, pred["forced_prediction"], pred["label"], pred["score"])
        results[rule] = {
            **bm,
            "open_set_macro_f1": om["open_set_macro_f1"],
            "committed_updates": sum(l["updated"] for l in logs),
        }
        print(
            f"[{rule:15s}] Acc={bm['known_acc']*100:6.2f}% | AUROC={bm['auroc']*100:6.2f}% | "
            f"AUPR={bm['aupr']*100:6.2f}% | FPR95={bm['fpr95']*100:6.2f}%"
        )

    out_file = run_dir / f"{args.pool}_e2_e3_controls.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved E2/E3 control metrics to {out_file}")


if __name__ == "__main__":
    main()
