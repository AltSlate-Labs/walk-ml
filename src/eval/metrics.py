from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def topk_accuracy(topk_predictions: np.ndarray, true_labels: np.ndarray) -> float:
    if topk_predictions.ndim != 2:
        raise ValueError("topk_predictions should have shape [N, K]")
    if true_labels.ndim != 1:
        raise ValueError("true_labels should have shape [N]")
    if topk_predictions.shape[0] != true_labels.shape[0]:
        raise ValueError("Mismatched N between topk_predictions and true_labels")
    matches = (topk_predictions == true_labels[:, None]).any(axis=1)
    return float(np.mean(matches))


def per_user_accuracy(true_labels: Sequence[int], pred_labels: Sequence[int]) -> Dict[str, float]:
    hits = defaultdict(int)
    totals = defaultdict(int)
    for truth, pred in zip(true_labels, pred_labels):
        key = str(int(truth))
        totals[key] += 1
        if int(truth) == int(pred):
            hits[key] += 1
    return {
        user_id: float(hits[user_id] / totals[user_id])
        for user_id in sorted(totals.keys(), key=int)
    }


def confusion_matrix_rows(
    true_labels: Iterable[int],
    pred_labels: Iterable[int],
) -> List[Dict[str, int]]:
    matrix = defaultdict(int)
    for truth, pred in zip(true_labels, pred_labels):
        matrix[(int(truth), int(pred))] += 1
    rows: List[Dict[str, int]] = []
    for (truth, pred), count in sorted(matrix.items()):
        rows.append({"true_label": truth, "pred_label": pred, "count": int(count)})
    return rows


def compute_threshold_metrics(
    max_scores: np.ndarray,
    true_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    accepted = max_scores >= threshold
    far = float(np.mean(impostor_scores >= threshold)) if impostor_scores.size else 0.0
    frr = float(np.mean(true_scores < threshold)) if true_scores.size else 0.0
    accept_rate = float(np.mean(accepted)) if accepted.size else 0.0
    return {"far": far, "frr": frr, "accept_rate": accept_rate}
