from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _ensure_binary_inputs(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_prob), dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise RuntimeError("y_true and y_prob must be 1D")
    if len(yt) != len(yp):
        raise RuntimeError("y_true and y_prob must have the same length")
    if len(yt) == 0:
        raise RuntimeError("Empty inputs")
    if not np.isin(yt, [0, 1]).all():
        raise RuntimeError("y_true must contain only 0/1")
    if not np.isfinite(yp).all():
        raise RuntimeError("y_prob contains non-finite values")
    return yt, yp


def roc_auc_binary(y_true: Sequence[int], y_prob: Sequence[float]) -> float | None:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    pos = int((yt == 1).sum())
    neg = int((yt == 0).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(yp, kind="mergesort")
    ranks = np.empty(len(yp), dtype=np.float64)

    i = 0
    rank = 1.0
    while i < len(yp):
        j = i
        while j + 1 < len(yp) and yp[order[j + 1]] == yp[order[i]]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i)))
        ranks[order[i : j + 1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    rank_sum_pos = float(ranks[yt == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def average_precision_binary(
    y_true: Sequence[int], y_score: Sequence[float], positive_label: int = 1
) -> float | None:
    yt, ys = _ensure_binary_inputs(y_true, y_score)
    if positive_label not in (0, 1):
        raise RuntimeError("positive_label must be 0 or 1")

    y_pos = (yt == positive_label).astype(np.int64)
    n_pos = int(y_pos.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-ys, kind="mergesort")
    y_sorted = y_pos[order]

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    for i, y in enumerate(y_sorted, start=1):
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        if y == 1:
            ap += precision * (recall - prev_recall)
            prev_recall = recall

    return float(ap)


def fpr_at_tpr(
    y_true: Sequence[int], y_prob: Sequence[float], target_tpr: float = 0.95
) -> float | None:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    pos = int((yt == 1).sum())
    neg = int((yt == 0).sum())
    if pos == 0 or neg == 0:
        return None

    thresholds = np.unique(yp)
    thresholds = np.concatenate(([1.0 + 1e-12], thresholds[::-1], [-1e-12]))

    best = None
    for thr in thresholds:
        pred = (yp >= thr).astype(np.int64)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, neg)
        if tpr >= target_tpr and (best is None or fpr < best):
            best = fpr
    return None if best is None else float(best)


def negative_log_likelihood(
    y_true: Sequence[int], y_prob: Sequence[float], eps: float = 1e-12
) -> float:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    yp = np.clip(yp, eps, 1.0 - eps)
    nll = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)).mean()
    return float(nll)


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    return float(np.mean((yp - yt) ** 2))


def brier_skill_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float | None:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    bs = brier_score(yt, yp)
    base = float(np.mean(yt))
    bs_ref = float(np.mean((base - yt) ** 2))
    if bs_ref == 0:
        return None
    return float(1.0 - bs / bs_ref)


def fixed_ece(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 15
) -> float:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yt)

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (yp >= lo) & (yp < hi if i < n_bins - 1 else yp <= hi)
        if not mask.any():
            continue
        conf = float(yp[mask].mean())
        acc = float(yt[mask].mean())
        ece += abs(acc - conf) * (mask.sum() / n)
    return float(ece)


def adaptive_ece(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 15
) -> float:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    order = np.argsort(yp)
    yt = yt[order]
    yp = yp[order]
    n = len(yt)

    ece = 0.0
    for i in range(n_bins):
        s = round(i * n / n_bins)
        e = round((i + 1) * n / n_bins)
        if e <= s:
            continue
        yt_bin = yt[s:e]
        yp_bin = yp[s:e]
        acc = float(yt_bin.mean())
        conf = float(yp_bin.mean())
        ece += abs(acc - conf) * (len(yt_bin) / n)
    return float(ece)


def binary_metrics_at_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> dict[str, Any]:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    pred = (yp >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())

    acc = _safe_div(tp + tn, len(yt))
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    tnr = _safe_div(tn, tn + fp)
    bal_acc = 0.5 * (rec + tnr)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def sweep_thresholds(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> list[dict[str, Any]]:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)
    uniq = np.unique(yp)
    thresholds = [0.0, 0.5, 1.0]
    thresholds.extend(float(x) for x in uniq.tolist())
    thresholds = sorted(set(thresholds))
    return [binary_metrics_at_threshold(yt, yp, thr) for thr in thresholds]


def best_threshold_by_accuracy(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> dict[str, Any]:
    sweep = sweep_thresholds(y_true, y_prob)
    best = max(sweep, key=lambda x: (x["accuracy"], -abs(x["threshold"] - 0.5)))
    return best


def best_threshold_by_balanced_accuracy(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> dict[str, Any]:
    sweep = sweep_thresholds(y_true, y_prob)
    best = max(
        sweep, key=lambda x: (x["balanced_accuracy"], -abs(x["threshold"] - 0.5))
    )
    return best


def compute_all_binary_metrics(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> dict[str, Any]:
    yt, yp = _ensure_binary_inputs(y_true, y_prob)

    default_thr = binary_metrics_at_threshold(yt, yp, 0.5)
    best_acc = best_threshold_by_accuracy(yt, yp)
    best_bal = best_threshold_by_balanced_accuracy(yt, yp)

    metrics = {
        "num_rows": len(yt),
        "num_correct": int((yt == 1).sum()),
        "num_incorrect": int((yt == 0).sum()),
        "threshold_0_5": default_thr,
        "best_accuracy_threshold": best_acc,
        "best_balanced_accuracy_threshold": best_bal,
        "roc_auc": roc_auc_binary(yt, yp),
        "aupr_correct": average_precision_binary(yt, yp, positive_label=1),
        "aupr_incorrect": average_precision_binary(1 - yt, 1.0 - yp, positive_label=1),
        "fpr_at_95_tpr": fpr_at_tpr(yt, yp, target_tpr=0.95),
        "nll": negative_log_likelihood(yt, yp),
        "brier": brier_score(yt, yp),
        "brier_skill_score": brier_skill_score(yt, yp),
        "ece_fixed_15": fixed_ece(yt, yp, n_bins=15),
        "ece_adaptive_15": adaptive_ece(yt, yp, n_bins=15),
    }
    return metrics


def save_threshold_sweep_json(
    path: Path, y_true: Sequence[int], y_prob: Sequence[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sweep = sweep_thresholds(y_true, y_prob)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)


def save_probability_distribution_plots(
    out_dir: Path, y_true: Sequence[int], y_prob: Sequence[float], prefix: str = "aux"
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    yt, yp = _ensure_binary_inputs(y_true, y_prob)

    correct_probs = yp[yt == 1]
    incorrect_probs = yp[yt == 0]

    raw_path = out_dir / f"{prefix}_prob_distribution_counts.png"
    density_path = out_dir / f"{prefix}_prob_distribution_density.png"

    plt.figure(figsize=(8, 5))
    plt.hist(correct_probs, bins=20, alpha=0.7, label="actual correct")
    plt.hist(incorrect_probs, bins=20, alpha=0.7, label="actual incorrect")
    plt.xlabel("head_prob_correct")
    plt.ylabel("count")
    plt.title("Probability distribution by actual correctness (counts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(raw_path, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(correct_probs, bins=20, alpha=0.7, label="actual correct", density=True)
    plt.hist(
        incorrect_probs, bins=20, alpha=0.7, label="actual incorrect", density=True
    )
    plt.xlabel("head_prob_correct")
    plt.ylabel("density")
    plt.title("Probability distribution by actual correctness (density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(density_path, dpi=160)
    plt.close()

    return {
        "raw_count_plot": str(raw_path),
        "density_plot": str(density_path),
    }


def metrics_report_text(metrics: dict[str, Any]) -> str:
    t05 = metrics["threshold_0_5"]
    best_acc = metrics["best_accuracy_threshold"]
    best_bal = metrics["best_balanced_accuracy_threshold"]

    lines = [
        f"num_rows: {metrics['num_rows']}",
        f"num_correct: {metrics['num_correct']}",
        f"num_incorrect: {metrics['num_incorrect']}",
        "",
        f"roc_auc: {metrics['roc_auc']}",
        f"aupr_correct: {metrics['aupr_correct']}",
        f"aupr_incorrect: {metrics['aupr_incorrect']}",
        f"fpr_at_95_tpr: {metrics['fpr_at_95_tpr']}",
        f"nll: {metrics['nll']}",
        f"brier: {metrics['brier']}",
        f"brier_skill_score: {metrics['brier_skill_score']}",
        f"ece_fixed_15: {metrics['ece_fixed_15']}",
        f"ece_adaptive_15: {metrics['ece_adaptive_15']}",
        "",
        f"threshold@0.5 accuracy: {t05['accuracy']}",
        f"threshold@0.5 precision: {t05['precision']}",
        f"threshold@0.5 recall: {t05['recall']}",
        f"threshold@0.5 f1: {t05['f1']}",
        f"threshold@0.5 balanced_accuracy: {t05['balanced_accuracy']}",
        "",
        f"best_accuracy threshold: {best_acc['threshold']}",
        f"best_accuracy value: {best_acc['accuracy']}",
        f"best_accuracy precision: {best_acc['precision']}",
        f"best_accuracy recall: {best_acc['recall']}",
        f"best_accuracy f1: {best_acc['f1']}",
        "",
        f"best_balanced_accuracy threshold: {best_bal['threshold']}",
        f"best_balanced_accuracy value: {best_bal['balanced_accuracy']}",
        f"best_balanced_accuracy precision: {best_bal['precision']}",
        f"best_balanced_accuracy recall: {best_bal['recall']}",
        f"best_balanced_accuracy f1: {best_bal['f1']}",
    ]
    return "\n".join(lines)
