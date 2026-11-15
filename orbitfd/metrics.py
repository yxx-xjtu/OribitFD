"""Open-set fault diagnosis evaluation metrics.

Target ground-truth labels (0..K-1 for known classes, -1 for unknown faults)
are accessed exclusively in this module after prediction and online adaptation.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def benchmark_metrics(y, forced_prediction, score) -> dict[str, float | int | None]:
    """Compute the four primary benchmark metrics reported in Tables II-VI.

    - ``known_acc``: Known-class classification accuracy before rejection.
    - ``auroc``: Area under ROC curve with unknown faults as the positive class.
    - ``aupr``: Average precision with unknown faults as the positive class.
    - ``fpr95``: Fraction of unknown samples accepted at the first empirical
      threshold retaining at least 95% of known samples (ties retained).
    """
    y, pred, score = map(np.asarray, (y, forced_prediction, score))
    if not (y.ndim == pred.ndim == score.ndim == 1 and len(y) == len(pred) == len(score)):
        raise ValueError("Aligned 1D arrays required.")
    if not np.isfinite(score).all() or not ((y >= 0) | (y == -1)).all():
        raise ValueError("Invalid score or label.")
    known, unknown = y >= 0, y == -1
    result: dict[str, float | int | None] = {
        "known_acc": float((pred[known] == y[known]).mean()) if known.any() else None,
        "auroc": None,
        "aupr": None,
        "fpr95": None,
        "id_count": int(known.sum()),
        "ood_count": int(unknown.sum()),
        "fpr95_threshold": None,
        "fpr95_actual_id_retention": None,
    }
    if known.any() and unknown.any():
        threshold = float(np.sort(score[known])[math.ceil(0.95 * known.sum()) - 1])
        result.update(
            auroc=float(roc_auc_score(unknown, score)),
            aupr=float(average_precision_score(unknown, score)),
            fpr95=float((score[unknown] <= threshold).mean()),
            fpr95_threshold=threshold,
            fpr95_actual_id_retention=float((score[known] <= threshold).mean()),
        )
        fpr, tpr, _ = roc_curve(known, -score, drop_intermediate=False)
        assert np.isclose(result["fpr95"], fpr[np.flatnonzero(tpr >= 0.95)[0]])
    return result


def open_set_metrics(y, forced_prediction, label, score) -> dict[str, float | None]:
    """Comprehensive threshold-free and source-calibrated (S = 1) operating-point metrics."""
    y, pred, decision, score = map(np.asarray, (y, forced_prediction, label, score))
    if not (
        y.ndim == pred.ndim == decision.ndim == score.ndim == 1
        and len(y) == len(pred) == len(decision) == len(score)
    ):
        raise ValueError("Metric arguments must be aligned 1D arrays.")
    known, unknown = y >= 0, y == -1
    if not (known | unknown).all() or not np.isfinite(score).all():
        raise ValueError("Ground truth uses 0..C-1 and -1; scores must be finite.")
    accept = decision >= 0
    result: dict[str, float | None] = {}
    result["known_accuracy_before_rejection"] = (
        float((pred[known] == y[known]).mean()) if known.any() else None
    )
    result["known_acceptance"] = float(accept[known].mean()) if known.any() else None
    result["known_correct_acceptance"] = (
        float(((pred == y) & accept)[known].mean()) if known.any() else None
    )
    selected = known & accept
    result["known_selective_accuracy"] = (
        float((pred[selected] == y[selected]).mean()) if selected.any() else None
    )
    result["unknown_rejection_including_defer"] = (
        float((~accept[unknown]).mean()) if unknown.any() else None
    )
    result["unknown_candidate_rate"] = float((decision[unknown] == -1).mean()) if unknown.any() else None
    result["known_defer_rate"] = float((decision[known] == -2).mean()) if known.any() else None
    result["unknown_defer_rate"] = float((decision[unknown] == -2).mean()) if unknown.any() else None
    result["overall_defer_rate"] = float((decision == -2).mean()) if len(y) else None

    if known.any() and unknown.any():
        result["auroc_unknown_positive"] = float(roc_auc_score(unknown, score))
        result["aupr_unknown_positive"] = float(average_precision_score(unknown, score))
        fpr, tpr, _ = roc_curve(unknown, score, drop_intermediate=False)
        result["known_false_rejection_at_95_unknown_recall"] = float(
            fpr[np.flatnonzero(tpr >= 0.95)[0]]
        )
        thr = float(np.sort(score[known])[math.ceil(0.95 * known.sum()) - 1])
        result["unknown_acceptance_at_95_known_retention"] = float((score[unknown] <= thr).mean())
        ccr = result["known_correct_acceptance"]
        ur = result["unknown_rejection_including_defer"]
        result["h_correct_acceptance_unknown_rejection"] = (
            float(2 * ccr * ur / (ccr + ur)) if ccr + ur > 0 else 0.0
        )
        # Open-set macro-F1 over K known classes + 1 unified unknown class (-1) at S = 1.
        y_open = np.where(known, y, -1)
        pred_open = np.where(accept, decision, -1)
        classes_open = sorted(set(y_open.tolist()))
        result["open_set_macro_f1"] = float(
            f1_score(y_open, pred_open, labels=classes_open, average="macro", zero_division=0)
        )
        # Area under OSCR curve
        order = np.argsort(score, kind="stable")
        sc = score[order]
        endpoints = np.r_[np.flatnonzero(sc[1:] != sc[:-1]), len(sc) - 1]
        ccr_curve = np.r_[0.0, np.cumsum((known & (pred == y))[order])[endpoints] / known.sum()]
        uar_curve = np.r_[0.0, np.cumsum(unknown[order])[endpoints] / unknown.sum()]
        integrate = getattr(np, "trapezoid", np.trapz)
        result["auoscr"] = float(integrate(ccr_curve, uar_curve))
    else:
        for key in (
            "auroc_unknown_positive",
            "aupr_unknown_positive",
            "known_false_rejection_at_95_unknown_recall",
            "unknown_acceptance_at_95_known_retention",
            "h_correct_acceptance_unknown_rejection",
            "open_set_macro_f1",
            "auoscr",
        ):
            result[key] = None
    return result
