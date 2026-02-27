from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _candidate_thresholds(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> np.ndarray:
    combined = np.concatenate([genuine_scores, impostor_scores])
    unique = np.unique(combined)
    if unique.size <= 1024:
        base = unique
    else:
        quantiles = np.linspace(0.0, 1.0, 1024)
        base = np.quantile(unique, quantiles)

    score_dtype = np.result_type(genuine_scores.dtype, impostor_scores.dtype, np.float32)
    max_observed = np.array(np.max(combined), dtype=score_dtype).item()
    above_max = np.nextafter(
        np.array(max_observed, dtype=score_dtype),
        np.array(np.inf, dtype=score_dtype),
    ).item()
    if not np.isfinite(float(above_max)) or float(above_max) <= float(max_observed):
        if np.issubdtype(score_dtype, np.floating):
            eps = np.finfo(score_dtype).eps
            scale = max(1.0, abs(float(max_observed)))
            above_max = np.array(float(max_observed) + float(eps) * scale, dtype=score_dtype).item()
        else:
            above_max = np.array(float(max_observed) + 1.0, dtype=np.float64).item()
    candidates = np.append(np.asarray(base, dtype=np.float64), above_max)
    return np.unique(candidates)


def compute_far_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    far = float(np.mean(impostor_scores >= threshold)) if impostor_scores.size else 0.0
    frr = float(np.mean(genuine_scores < threshold)) if genuine_scores.size else 0.0
    return {"far": far, "frr": frr}


def calibrate_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: Optional[float] = None,
) -> Dict[str, float]:
    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError("Need both genuine and impostor scores for calibration.")

    thresholds = _candidate_thresholds(genuine_scores, impostor_scores)
    far_curve = []
    frr_curve = []

    for threshold in thresholds:
        rates = compute_far_frr(genuine_scores, impostor_scores, float(threshold))
        far_curve.append(rates["far"])
        frr_curve.append(rates["frr"])

    far_curve = np.asarray(far_curve, dtype=np.float32)
    frr_curve = np.asarray(frr_curve, dtype=np.float32)
    diff = np.abs(far_curve - frr_curve)
    eer_idx = int(np.argmin(diff))
    eer_threshold = float(thresholds[eer_idx])
    eer = float((far_curve[eer_idx] + frr_curve[eer_idx]) * 0.5)

    if target_far is None:
        selected_threshold = eer_threshold
    else:
        feasible = np.where(far_curve <= float(target_far))[0]
        if feasible.size == 0:
            selected_threshold = float(thresholds[-1])
        else:
            selected_threshold = float(thresholds[int(feasible[0])])

    selected_rates = compute_far_frr(genuine_scores, impostor_scores, selected_threshold)
    return {
        "threshold": selected_threshold,
        "far": selected_rates["far"],
        "frr": selected_rates["frr"],
        "eer": eer,
        "eer_threshold": eer_threshold,
    }
