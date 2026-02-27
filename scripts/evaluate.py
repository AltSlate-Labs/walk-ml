#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.calibration import calibrate_threshold
from src.eval.metrics import (
    compute_threshold_metrics,
    confusion_matrix_rows,
    per_user_accuracy,
)
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gait identification predictions")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.csv")
    parser.add_argument("--embeddings", default=None, help="Optional embeddings .npz")
    parser.add_argument(
        "--calibrate-from",
        default=None,
        help="Optional .npz file with genuine_scores and impostor_scores",
    )
    parser.add_argument("--threshold", default="auto", help="Float or 'auto'")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--out-dir", default=None, help="Output directory")
    return parser.parse_args()


def _read_predictions(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_confusion_csv(path: Path, rows: List[Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_label", "pred_label", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _build_model_card(report: Dict[str, object]) -> str:
    eer_value = report.get("eer")
    eer_text = f"{float(eer_value):.4f}" if eer_value is not None else "n/a"
    lines = [
        "# Model Card (Evaluation Snapshot)",
        "",
        "## Summary",
        f"- Top-1 accuracy: {report['top1']:.4f}",
        f"- Threshold: {report['threshold']:.4f}",
        f"- FAR: {report['far']:.4f}",
        f"- FRR: {report['frr']:.4f}",
        f"- EER: {eer_text}",
        "",
        "## Notes",
        "- This report is generated from prediction artifacts.",
        "- Review per-user metrics before promotion to production.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions).resolve()
    rows = _read_predictions(predictions_path)
    if not rows:
        raise ValueError("Predictions file is empty.")

    true_labels = np.array([int(row["true_label"]) for row in rows], dtype=np.int32)
    pred_labels = np.array([int(row["pred_label"]) for row in rows], dtype=np.int32)
    max_scores = np.array([float(row["max_score"]) for row in rows], dtype=np.float32)
    true_scores = np.array([float(row["true_score"]) for row in rows], dtype=np.float32)
    impostor_scores = np.array([float(row["impostor_score"]) for row in rows], dtype=np.float32)

    if args.threshold == "auto":
        if args.calibrate_from:
            with np.load(Path(args.calibrate_from), allow_pickle=False) as payload:
                cal_genuine = payload["genuine_scores"].astype(np.float32)
                cal_impostor = payload["impostor_scores"].astype(np.float32)
        else:
            cal_genuine = true_scores
            cal_impostor = impostor_scores

        calibration = calibrate_threshold(
            genuine_scores=cal_genuine,
            impostor_scores=cal_impostor,
            target_far=float(args.target_far),
        )
        threshold = float(calibration["threshold"])
        eer = float(calibration["eer"])
    else:
        threshold = float(args.threshold)
        calibration = {"threshold": threshold}
        eer = None

    top1 = float(np.mean(true_labels == pred_labels))
    threshold_metrics = compute_threshold_metrics(
        max_scores=max_scores,
        true_scores=true_scores,
        impostor_scores=impostor_scores,
        threshold=threshold,
    )

    per_user = per_user_accuracy(true_labels.tolist(), pred_labels.tolist())
    confusion = confusion_matrix_rows(true_labels.tolist(), pred_labels.tolist())

    embedding_shape = None
    if args.embeddings:
        with np.load(Path(args.embeddings), allow_pickle=False) as payload:
            embedding_shape = list(payload["embeddings"].shape)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else predictions_path.parent / "evaluation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "top1": top1,
        "threshold": threshold,
        "far": float(threshold_metrics["far"]),
        "frr": float(threshold_metrics["frr"]),
        "accept_rate": float(threshold_metrics["accept_rate"]),
        "eer": eer,
        "num_samples": int(true_labels.shape[0]),
        "per_user_accuracy": per_user,
        "embedding_shape": embedding_shape,
    }

    save_json(out_dir / "evaluation_report.json", report)
    save_json(out_dir / "thresholds.json", calibration)
    _write_confusion_csv(out_dir / "confusion_matrix.csv", confusion)

    model_card = _build_model_card(report)
    (out_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Evaluation artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
