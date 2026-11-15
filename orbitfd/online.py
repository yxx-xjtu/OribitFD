"""Causal shared-state adaptation engine for OrbitFD deployment.

Processes ordered target blocks via Predict -> Propose -> Check -> Commit:
1. Block t is predicted and recorded using the prior state a_{t-1}.
2. State-invariant normal-residual gates select orbit-consistent evidence.
3. Temporal split separates fit (earlier half) and check (later half) sets.
4. Huber IRLS location estimation, consistency/trust-region checks, step-size
   bounding, and held-out loss improvement gate whether a_t is committed for
   block t + 1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .bank import OrbitBank


@dataclass
class OnlineConfig:
    """Hyperparameters for causal shared-state estimation (fixed across all tasks)."""

    min_fit: int = 8
    min_check: int = 8
    purge_gap: int = 1
    prior_weight: float = 0.05
    huber_delta: float = 1.5
    irls_steps: int = 6
    max_step: float = 0.25
    max_innovation: float = 0.5
    split_agreement: float = 0.75
    min_relative_gain: float = 0.02
    min_absolute_gain: float = 1e-4

    def __post_init__(self) -> None:
        if self.min_fit < 1 or self.min_check < 1 or self.purge_gap < 0 or self.irls_steps < 1:
            raise ValueError("Invalid sample counts or iteration count.")
        if (
            self.prior_weight < 0
            or self.huber_delta <= 0
            or self.max_step <= 0
            or self.split_agreement <= 0
            or self.max_innovation <= 0
        ):
            raise ValueError("Invalid state-estimation parameters.")
        if self.min_relative_gain < 0 or self.min_absolute_gain < 0:
            raise ValueError("Gains must be nonnegative.")


def huber_vector_loss(obs: torch.Tensor, state: torch.Tensor, delta: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(obs - state, dim=-1) / math.sqrt(obs.shape[-1])
    return torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta)).mean()


def robust_location(obs: torch.Tensor, prior: torch.Tensor, cfg: OnlineConfig) -> torch.Tensor:
    """Coordinatewise median initialization followed by weakly anchored Huber IRLS."""
    estimate = obs.median(0).values
    for _ in range(cfg.irls_steps):
        norm = torch.linalg.vector_norm(obs - estimate, dim=1) / math.sqrt(obs.shape[1])
        weights = (cfg.huber_delta / norm.clamp_min(1e-9)).clamp_max(1.0)
        estimate = (
            (weights[:, None] * obs).mean(0) + cfg.prior_weight * prior
        ) / (weights.mean() + cfg.prior_weight)
    return estimate


class OrbitOnline:
    """Stateful causal deployment engine operating on a frozen ``OrbitBank``."""

    def __init__(self, bank: OrbitBank, config: OnlineConfig | None = None):
        self.bank = bank
        self.cfg = config or OnlineConfig()
        self.state = torch.zeros(bank.rank, dtype=bank.mean.dtype, device=bank.device)
        self.accepted_updates = 0
        self.num_blocks = 0

    def reset(self) -> None:
        """Reset deployment state to zero at the start of an independent stream."""
        self.state.zero_()
        self.accepted_updates = 0
        self.num_blocks = 0

    @torch.no_grad()
    def step(self, features: Any, *, adapt: bool = True) -> dict[str, Any]:
        z = self.bank.normalize(features)
        prior = self.state.clone()
        # Causal guarantee: block outputs are computed with a_{t-1} and never revised.
        out: dict[str, Any] = self.bank.predict_normalized(z, prior)
        out["state_used"] = prior
        out["state_after"] = prior.clone()
        out["environment_displacement"] = float(
            torch.linalg.vector_norm(self.bank.coordinate_scale * prior)
        )
        out["updated"] = False
        out["reason"] = "adaptation_disabled"
        out["n_fit"] = 0
        out["n_check"] = 0
        out["eligible_mask"] = torch.zeros(len(z), dtype=torch.bool, device=z.device)
        out["fit_indices"] = torch.empty(0, dtype=torch.long, device=z.device)
        out["check_indices"] = torch.empty(0, dtype=torch.long, device=z.device)
        self.num_blocks += 1
        if not adapt:
            return out

        eligible = (out["normal_score"] <= 1.0) & (out["normal_margin"] >= self.bank.margin_gate)
        out["eligible_mask"] = eligible.clone()
        split = len(z) // 2
        left = torch.arange(0, split, device=z.device)
        right = torch.arange(min(split + self.cfg.purge_gap, len(z)), len(z), device=z.device)
        left = left[eligible[left]]
        right = right[eligible[right]]
        out["fit_indices"], out["check_indices"] = left.clone(), right.clone()
        out["n_fit"], out["n_check"] = len(left), len(right)
        if len(left) < self.cfg.min_fit:
            out["reason"] = "insufficient_fit_evidence"
            return out
        if len(right) < self.cfg.min_check:
            out["reason"] = "insufficient_check_evidence"
            return out

        observations = out["coordinate_observation"]
        candidate_raw = robust_location(observations[left], prior, self.cfg)
        fit_norm = torch.linalg.vector_norm(observations[left] - candidate_raw, dim=1) / math.sqrt(
            self.bank.rank
        )
        out["fit_weights"] = (self.cfg.huber_delta / fit_norm.clamp_min(1e-9)).clamp_max(1.0)
        check_estimate = robust_location(observations[right], prior, self.cfg)
        disagreement = torch.linalg.vector_norm(candidate_raw - check_estimate) / math.sqrt(
            self.bank.rank
        )
        out["split_disagreement"] = float(disagreement)
        if not torch.isfinite(candidate_raw).all() or not torch.isfinite(check_estimate).all():
            out["reason"] = "nonfinite_proposal"
            return out
        if float(disagreement) > self.cfg.split_agreement:
            out["reason"] = "inconsistent_context"
            return out
        if float(torch.linalg.vector_norm(candidate_raw)) > self.bank.state_radius:
            out["reason"] = "outside_source_trust_region"
            return out

        raw_innovation = float(torch.linalg.vector_norm(candidate_raw - prior))
        out["raw_state_innovation"] = raw_innovation
        if raw_innovation > self.cfg.max_innovation:
            out["reason"] = "innovation_exceeds_declared_drift_budget"
            return out

        step = candidate_raw - prior
        step *= min(1.0, self.cfg.max_step / max(float(torch.linalg.vector_norm(step)), 1e-12))
        candidate = prior + step
        old_loss = huber_vector_loss(observations[right], prior, self.cfg.huber_delta)
        new_loss = huber_vector_loss(observations[right], candidate, self.cfg.huber_delta)
        gain = float(old_loss - new_loss)
        out["holdout_gain"] = gain
        required_gain = max(self.cfg.min_absolute_gain, self.cfg.min_relative_gain * float(old_loss))
        if gain <= required_gain:
            out["reason"] = "no_out_of_fit_gain"
            return out

        self.state = candidate.detach().clone()
        self.accepted_updates += 1
        out["state_after"] = self.state.clone()
        out["updated"] = True
        out["reason"] = "committed_for_next_block"
        return out
