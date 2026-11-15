"""Chronological block-stream evaluation helpers for OrbitFD and geometric controls."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from .bank import OrbitBank, empirical_quantile
from .online import OnlineConfig, OrbitOnline

MODES = ("online", "frozen_orbit", "frozen_local", "frozen_global", "static", "normal_only", "ungated")


def make_global_bank(bank: OrbitBank, cal_features: Any, cal_y: Any, alpha: float = 0.05) -> OrbitBank:
    """Build the Frozen-global Orbit control (97.5th percentile at zero state)."""
    global_bank = copy.deepcopy(bank)
    zc = bank.normalize(cal_features)
    comp = bank.components(zc, torch.zeros(bank.rank, dtype=torch.float64, device=zc.device))
    yc = torch.as_tensor(cal_y, dtype=torch.long, device=zc.device)
    idx = torch.arange(len(yc), device=zc.device)
    global_bank.tau_parallel = empirical_quantile(comp["parallel"][idx, yc], 1 - alpha / 2)
    return global_bank


def evaluate_stream(
    bank: OrbitBank,
    z: Any,
    block_id: Any | None = None,
    *,
    block_size: int = 32,
    mode: str = "online",
    config: OnlineConfig | None = None,
    global_bank: OrbitBank | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Evaluate a chronological feature stream without target ground truth.

    ``block_id`` defines operational block chunking (default: fixed 32-window blocks).
    State is never reset across blocks within a stream.
    """
    z = np.asarray(z)
    if z.ndim != 2 or len(z) == 0:
        raise ValueError("Expected z of shape [N, D] with N >= 1.")
    if block_id is None:
        block_id = np.arange(len(z)) // block_size
    else:
        block_id = np.asarray(block_id)
    if block_id.shape != (len(z),):
        raise ValueError("Expected block_id of shape [N].")
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}.")
    if mode == "frozen_global" and global_bank is None:
        raise ValueError("mode='frozen_global' requires passing global_bank.")

    ends = np.r_[np.flatnonzero(block_id[1:] != block_id[:-1]) + 1, len(z)]
    engine = OrbitOnline(bank, config)
    parts: dict[str, list[np.ndarray]] = {
        "score": [],
        "forced_prediction": [],
        "label": [],
        "normal_score": [],
        "parallel_score": [],
    }
    logs: list[dict[str, Any]] = []
    begin = 0
    for block_number, end in enumerate(ends):
        batch = z[begin:end]
        if mode == "frozen_global":
            assert global_bank is not None
            out = global_bank.predict(batch)
            out["state_used"] = engine.state.clone()
            out["state_after"] = engine.state.clone()
            out["updated"] = False
            out["reason"] = "frozen_global"
        else:
            out = engine.step(batch, adapt=(mode == "online"))

        if mode == "static":
            zn = bank.normalize(batch)
            d = torch.cdist(zn, bank.prototypes) / np.sqrt(zn.shape[1])
            value, pred = d.min(1)
            out["score"], out["forced_prediction"] = value / bank.tau_static, pred
            out["label"] = torch.where(out["score"] <= 1.0, pred, -torch.ones_like(pred))
        elif mode == "normal_only":
            out["score"] = out["normal_score"]
            out["forced_prediction"] = out["normal_class"]
            out["label"] = torch.where(
                out["score"] <= 1.0, out["normal_class"], -torch.ones_like(out["normal_class"])
            )
        elif mode == "ungated":
            proposal = out["coordinate_observation"].mean(0)
            cfg = engine.cfg
            step = proposal - engine.state
            step *= min(1.0, cfg.max_step / max(float(torch.linalg.vector_norm(step)), 1e-12))
            engine.state = engine.state + step
            out["state_after"] = engine.state.clone()
            out["updated"] = True
            out["reason"] = "ungated_mean_update"

        for k in parts:
            parts[k].append(out[k].detach().cpu().numpy())
        log = {k: v for k, v in out.items() if not isinstance(v, torch.Tensor)}
        log.update(
            {
                "block": block_number,
                "start": int(begin),
                "end": int(end),
                "state_used": out["state_used"].cpu().tolist(),
                "state_after": out["state_after"].cpu().tolist(),
            }
        )
        logs.append(log)
        begin = end
    return {k: np.concatenate(v) for k, v in parts.items()}, logs
