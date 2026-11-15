"""Frozen source geometry and class-consistent finite-support calibration for OrbitFD.

Only labeled source-training (fit) and disjoint source-calibration features enter
``fit_bank()``. Network parameters, prototypes, shared variation basis, coordinate
scales, and calibrated support widths remain strictly frozen during deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def features_tensor(x: Any, *, device: torch.device | str = "cpu") -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64, device=device).detach()
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError("Features must be a nonempty [N, D] array with D >= 2.")
    if not torch.isfinite(x).all():
        raise ValueError("Features contain NaN or infinity.")
    return x


def labels_tensor(y: Any, n: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    raw = torch.as_tensor(y, device=device).reshape(-1)
    if len(raw) != n or not torch.isfinite(raw).all():
        raise ValueError("Labels must be finite and have N entries.")
    if not torch.equal(raw, raw.long().to(raw.dtype)):
        raise ValueError("Labels must be integers.")
    return raw.long()


def empirical_quantile(x: torch.Tensor, q: float) -> float:
    """Empirical order statistic with 'higher' interpolation, lower-bounded by 1e-8."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1).")
    return max(float(torch.quantile(x.double(), q, interpolation="higher")), 1e-8)


@dataclass
class OrbitBank:
    """Frozen source prototype bank and calibrated finite-support bounds."""

    mean: torch.Tensor              # [D], frozen source feature mean
    scale: torch.Tensor             # [D], shrinkage-regularized source feature scale
    prototypes: torch.Tensor        # [C, D], standardized class means
    basis: torch.Tensor             # [D, r], orthonormal within-class variation basis U
    coordinate_scale: torch.Tensor  # [r], within-class standard deviation along U
    tau_normal: float               # Normal residual support width tau_N
    tau_parallel: float             # Along-orbit residual support width tau_P
    tau_static: float               # Isotropic static-prototype control width
    margin_gate: float              # Normal-residual margin gate gamma for state evidence
    state_radius: float             # Source-calibrated trust-region radius R
    within_energy: float = 0.0
    between_energy: float = 0.0
    format_version: int = 1

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def num_classes(self) -> int:
        return self.prototypes.shape[0]

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def normalize(self, raw: Any) -> torch.Tensor:
        x = features_tensor(raw, device=self.device)
        if x.shape[1] != self.mean.numel():
            raise ValueError("Feature dimension differs from source bank.")
        return (x - self.mean) / self.scale

    def components(self, z: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decompose standardized features relative to all class orbits at fixed ``state``.

        Because ``(I - U U^T) B = 0``, the normal residual ``N_c(z)`` is strictly
        invariant to ``state``; only the parallel residual ``P_c(z, a)`` translates.
        """
        state = torch.as_tensor(state, dtype=z.dtype, device=z.device)
        if state.shape != (self.rank,) or not torch.isfinite(state).all():
            raise ValueError("State must be a finite [r] vector.")
        residual0 = z[:, None, :] - self.prototypes[None, :, :]
        tangent0 = residual0 @ self.basis                          # [B, C, r]
        orthogonal = residual0 - tangent0 @ self.basis.T
        coordinates = tangent0 / self.coordinate_scale
        normal = torch.linalg.vector_norm(orthogonal, dim=-1) / math.sqrt(z.shape[-1] - self.rank)
        parallel = torch.linalg.vector_norm(coordinates - state, dim=-1) / math.sqrt(self.rank)
        residual = residual0 - (self.basis @ (self.coordinate_scale * state))[None, None, :]
        full = torch.linalg.vector_norm(residual, dim=-1) / math.sqrt(z.shape[-1])
        return {
            "normal": normal,
            "parallel": parallel,
            "coordinates": coordinates,
            "full": full,
        }

    def predict_normalized(self, z: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        comp = self.components(z, state)
        # Class-consistent max-min score: both residuals must hold for the SAME class c.
        per_class = torch.maximum(
            comp["normal"] / self.tau_normal,
            comp["parallel"] / self.tau_parallel,
        )
        score, forced = per_class.min(dim=1)
        normal_min, normal_class = comp["normal"].min(dim=1)
        top2 = torch.topk(comp["normal"], 2, dim=1, largest=False).values
        margin = (top2[:, 1] - top2[:, 0]) / top2[:, 1].clamp_min(1e-8)
        label = forced.clone()
        outside = score > 1.0
        label[outside & (normal_min > self.tau_normal)] = -1   # Unknown candidate (off-orbit)
        label[outside & (normal_min <= self.tau_normal)] = -2  # Context deferral (along-orbit mismatch)
        index = torch.arange(len(z), device=z.device)
        return {
            "score": score,
            "forced_prediction": forced,
            "label": label,
            "normal_score": normal_min / self.tau_normal,
            "chosen_class_normal": comp["normal"][index, forced] / self.tau_normal,
            "parallel_score": comp["parallel"][index, forced] / self.tau_parallel,
            "normal_class": normal_class,
            "normal_margin": margin,
            "coordinate_observation": comp["coordinates"][index, normal_class],
        }

    def predict(self, raw: Any, state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if state is None:
            state = torch.zeros(self.rank, dtype=self.mean.dtype, device=self.device)
        return self.predict_normalized(self.normalize(raw), state)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for k, v in self.__dict__.items():
            data[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "OrbitBank":
        with np.load(path, allow_pickle=False) as f:
            if int(f["format_version"]) != 1:
                raise ValueError("Unsupported bank version.")
            tensors = {
                k: torch.as_tensor(f[k].copy(), dtype=torch.float64, device=device)
                for k in ("mean", "scale", "prototypes", "basis", "coordinate_scale")
            }
            scalars = {
                k: float(f[k])
                for k in (
                    "tau_normal",
                    "tau_parallel",
                    "tau_static",
                    "margin_gate",
                    "state_radius",
                    "within_energy",
                    "between_energy",
                )
            }
        return cls(**tensors, **scalars)


@torch.no_grad()
def fit_bank(
    source_fit: Any,
    y_fit: Any,
    source_calibration: Any,
    y_calibration: Any,
    rank: int = 2,
    semantic_penalty: float = 0.2,
    normalization_shrinkage: float = 0.2,
    alpha: float = 0.05,
    radius_multiplier: float = 1.5,
    calibration_context: Any | None = None,
    context_drift_allowance: float = 0.2,
) -> OrbitBank:
    """Fit frozen OrbitBank strictly from source training and calibration features.

    Solves the trace-normalized generalized scatter objective:
        max_{U^T U = I_r} tr(U^T [S_w / tr(S_w) - kappa * S_b / tr(S_b)] U).
    """
    x = features_tensor(source_fit)
    xc = features_tensor(source_calibration)
    y = labels_tensor(y_fit, len(x))
    yc = labels_tensor(y_calibration, len(xc))
    if xc.shape[1] != x.shape[1]:
        raise ValueError("Fit and calibration dimensions differ.")
    unique = torch.unique(y).sort().values
    if len(unique) < 2 or not torch.equal(unique, torch.arange(len(unique))):
        raise ValueError("Source-known labels must be contiguous 0..C-1, C >= 2.")
    if not torch.equal(torch.unique(yc).sort().values, unique):
        raise ValueError("Calibration must contain exactly the source-known classes.")
    if any(int((y == c).sum()) < 3 or int((yc == c).sum()) < 3 for c in unique):
        raise ValueError("Need >= 3 fit and calibration samples per known class.")
    if not 1 <= rank < x.shape[1] or semantic_penalty < 0:
        raise ValueError("Require 1 <= rank < D and semantic_penalty >= 0.")
    if not 0 <= normalization_shrinkage <= 1 or not 0 < alpha < 1 or radius_multiplier <= 0:
        raise ValueError("Invalid normalization, alpha, or radius multiplier.")

    mean, variance = x.mean(0), x.var(0, unbiased=False)
    if float(variance.mean()) < 1e-12:
        raise ValueError("Collapsed features: no source variation to estimate.")
    scale = (
        (1 - normalization_shrinkage) * variance
        + normalization_shrinkage * variance.mean()
    ).clamp_min(1e-10).sqrt()
    z, zc = (x - mean) / scale, (xc - mean) / scale

    mu = torch.stack([z[y == c].mean(0) for c in unique])
    sw = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for c in unique:
        residual = z[y == c] - mu[c]
        sw += residual.T @ residual / len(residual) / len(unique)
    center = mu - mu.mean(0)
    sb = center.T @ center / len(unique)
    if float(torch.trace(sw)) < 1e-12:
        raise ValueError("No within-class variance: latent orbit is unidentifiable.")

    objective = (
        sw / torch.trace(sw).clamp_min(1e-12)
        - semantic_penalty * sb / torch.trace(sb).clamp_min(1e-12)
    )
    _, eigenvectors = torch.linalg.eigh((objective + objective.T) / 2)
    basis = eigenvectors[:, -rank:].flip(1).contiguous()
    # Deterministic eigenvector sign convention (largest-magnitude entry positive).
    for j in range(rank):
        pivot = basis[:, j].abs().argmax()
        if basis[pivot, j] < 0:
            basis[:, j] *= -1

    coordinate_scale = torch.diag(basis.T @ sw @ basis).clamp_min(1e-8).sqrt()
    bank = OrbitBank(
        mean,
        scale,
        mu,
        basis,
        coordinate_scale,
        1.0,
        1.0,
        1.0,
        0.05,
        4.0,
        float(torch.trace(basis.T @ sw @ basis) / torch.trace(sw)),
        float(torch.trace(basis.T @ sb @ basis) / torch.trace(sb).clamp_min(1e-12)),
    )

    comp = bank.components(zc, torch.zeros(rank, dtype=torch.float64))
    index = torch.arange(len(zc))
    bank.tau_normal = empirical_quantile(comp["normal"][index, yc], 1 - alpha / 2)
    bank.tau_parallel = empirical_quantile(comp["parallel"][index, yc], 1 - alpha / 2)
    bank.tau_static = empirical_quantile(comp["full"][index, yc], 1 - alpha)

    if calibration_context is not None:
        if context_drift_allowance < 0:
            raise ValueError("context_drift_allowance must be nonnegative.")
        groups = np.asarray(calibration_context)
        if groups.shape != (len(zc),):
            raise ValueError("calibration_context must have shape [N_cal].")
        coordinate = comp["coordinates"][index, yc]
        innovations = []
        for group in np.unique(groups):
            idx = torch.as_tensor(np.flatnonzero(groups == group), dtype=torch.long)
            if len(idx) < 16:
                continue
            cut = len(idx) // 2
            context_estimate = coordinate[idx[:cut]].median(0).values
            difference = coordinate[idx[cut + 1 :]] - context_estimate
            innovations.append(torch.linalg.vector_norm(difference, dim=1) / math.sqrt(rank))
        if not innovations:
            raise ValueError("Need at least one calibration context with >= 16 windows.")
        bank.tau_parallel = (
            empirical_quantile(torch.cat(innovations), 1 - alpha / 2) + context_drift_allowance
        )

    pred = bank.predict_normalized(zc, torch.zeros(rank, dtype=torch.float64))
    correct = pred["normal_class"] == yc
    if correct.any():
        bank.margin_gate = max(0.02, min(0.4, float(torch.quantile(pred["normal_margin"][correct], 0.05))))
    norms = torch.linalg.vector_norm(comp["coordinates"][index, yc], dim=1)
    bank.state_radius = max(1.0, radius_multiplier * empirical_quantile(norms, 0.995))
    return bank
