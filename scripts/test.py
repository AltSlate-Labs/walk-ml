#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ClipDataset, build_label_map, load_index
from src.data.splits import build_splits
from src.engine.test_loop import (
    build_centroid_gallery,
    build_prediction_rows,
    collect_embeddings,
    predict_closed_set,
)
from src.eval.calibration import calibrate_threshold
from src.eval.metrics import compute_threshold_metrics, per_user_accuracy, topk_accuracy
from src.models.gait_encoder_mlx import create_model, load_weights
from src.utils.io import load_config, save_json, utc_timestamp
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MLX gait identification model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-root", required=True, help="Dataset root path")
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", default="auto", help="Float or 'auto'")
    return parser.parse_args()


def _load_label_maps(run_dir: Path, splits) -> Dict[str, Dict]:
    label_map_path = run_dir / "label_map.json"
    index_to_user_path = run_dir / "index_to_user.json"
    if label_map_path.exists() and index_to_user_path.exists():
        with label_map_path.open("r", encoding="utf-8") as handle:
            label_map = json.load(handle)
        with index_to_user_path.open("r", encoding="utf-8") as handle:
            raw_index_to_user = json.load(handle)
        index_to_user = {int(k): str(v) for k, v in raw_index_to_user.items()}
        return {"label_map": label_map, "index_to_user": index_to_user}

    label_map = build_label_map(splits["train"] + splits["val"] + splits["test"])
    index_to_user = {index: user for user, index in label_map.items()}
    return {"label_map": label_map, "index_to_user": index_to_user}


def _write_predictions_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError("No prediction rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            return parent.parent
    return checkpoint_path.parent


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 42))
    set_seed(seed)

    data_root = Path(args.data_root).resolve()
    records = load_index(
        index_path=config["data"]["index_file"],
        data_root=data_root,
        min_quality=float(config["data"].get("min_quality", 0.0)),
    )
    splits = build_splits(
        records=records,
        train_sessions=config["data"]["train_sessions"],
        val_from_sessions=config["data"]["val_from_sessions"],
        test_sessions=config["data"]["test_sessions"],
        val_fraction=float(config["data"]["val_fraction"]),
        seed=seed,
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = _resolve_run_dir(checkpoint_path)

    maps = _load_label_maps(run_dir, splits)
    label_map = maps["label_map"]
    index_to_user = maps["index_to_user"]

    config["model"]["num_classes"] = len(label_map)
    batch_size = int(args.batch_size or config["data"].get("batch_size", 16))

    train_gallery_dataset = ClipDataset(
        records=splits["train"],
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=False,
        seed=seed,
    )
    eval_gallery_records = (
        splits["train"] + splits["val"]
        if args.split == "test"
        else splits["train"]
    )
    eval_gallery_dataset = ClipDataset(
        records=eval_gallery_records,
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=False,
        seed=seed,
    )
    eval_dataset = ClipDataset(
        records=splits[args.split],
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=False,
        seed=seed,
    )

    model = create_model(config, num_classes=len(label_map))
    load_weights(model, checkpoint_path)

    eval_gallery_output = collect_embeddings(model, eval_gallery_dataset, batch_size=batch_size, seed=seed)
    eval_gallery = build_centroid_gallery(
        embeddings=np.asarray(eval_gallery_output["embeddings"], dtype=np.float32),
        labels=np.asarray(eval_gallery_output["labels"], dtype=np.int32),
    )

    eval_output = collect_embeddings(model, eval_dataset, batch_size=batch_size, seed=seed + 1)
    pred = predict_closed_set(
        embeddings=np.asarray(eval_output["embeddings"], dtype=np.float32),
        true_labels=np.asarray(eval_output["labels"], dtype=np.int32),
        gallery_labels=np.asarray(eval_gallery["gallery_labels"], dtype=np.int32),
        gallery_vectors=np.asarray(eval_gallery["gallery_vectors"], dtype=np.float32),
        top_k=config["evaluation"].get("top_k", [1, 5]),
    )

    calibration_source = pred
    if args.threshold == "auto":
        if args.split == "val":
            calibration_source = pred
        else:
            calibration_gallery_output = collect_embeddings(
                model,
                train_gallery_dataset,
                batch_size=batch_size,
                seed=seed + 2,
            )
            calibration_gallery = build_centroid_gallery(
                embeddings=np.asarray(calibration_gallery_output["embeddings"], dtype=np.float32),
                labels=np.asarray(calibration_gallery_output["labels"], dtype=np.int32),
            )
            val_dataset = ClipDataset(
                records=splits["val"],
                label_map=label_map,
                seq_len=int(config["data"]["seq_len"]),
                image_size=config["data"]["image_size"],
                augment=False,
                seed=seed,
            )
            val_output = collect_embeddings(model, val_dataset, batch_size=batch_size, seed=seed + 3)
            calibration_source = predict_closed_set(
                embeddings=np.asarray(val_output["embeddings"], dtype=np.float32),
                true_labels=np.asarray(val_output["labels"], dtype=np.int32),
                gallery_labels=np.asarray(calibration_gallery["gallery_labels"], dtype=np.int32),
                gallery_vectors=np.asarray(calibration_gallery["gallery_vectors"], dtype=np.float32),
                top_k=config["evaluation"].get("top_k", [1, 5]),
            )

        calibration = calibrate_threshold(
            genuine_scores=np.asarray(calibration_source["true_scores"], dtype=np.float32),
            impostor_scores=np.asarray(calibration_source["impostor_scores"], dtype=np.float32),
            target_far=float(config["evaluation"].get("target_far", 0.01)),
        )
        threshold = float(calibration["threshold"])
    else:
        threshold = float(args.threshold)
        calibration = {"threshold": threshold, "eer": None}

    top1 = float(
        np.mean(
            np.asarray(pred["pred_labels"], dtype=np.int32)
            == np.asarray(eval_output["labels"], dtype=np.int32)
        )
    )
    top5 = None
    if 5 in pred["topk"]:
        top5 = topk_accuracy(
            np.asarray(pred["topk"][5], dtype=np.int32),
            np.asarray(eval_output["labels"], dtype=np.int32),
        )

    threshold_metrics = compute_threshold_metrics(
        max_scores=np.asarray(pred["max_scores"], dtype=np.float32),
        true_scores=np.asarray(pred["true_scores"], dtype=np.float32),
        impostor_scores=np.asarray(pred["impostor_scores"], dtype=np.float32),
        threshold=threshold,
    )

    rows = build_prediction_rows(
        true_labels=np.asarray(eval_output["labels"], dtype=np.int32),
        pred_labels=np.asarray(pred["pred_labels"], dtype=np.int32),
        max_scores=np.asarray(pred["max_scores"], dtype=np.float32),
        true_scores=np.asarray(pred["true_scores"], dtype=np.float32),
        impostor_scores=np.asarray(pred["impostor_scores"], dtype=np.float32),
        clip_paths=eval_output["clip_paths"],
        index_to_user=index_to_user,
        topk={int(k): np.asarray(v, dtype=np.int32) for k, v in pred["topk"].items()},
    )

    run_test_dir = run_dir / "test" / f"{args.split}-{utc_timestamp()}"
    run_test_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_test_dir / "test_predictions.csv"
    _write_predictions_csv(predictions_path, rows)

    np.savez(
        run_test_dir / "embeddings_test.npz",
        embeddings=np.asarray(eval_output["embeddings"], dtype=np.float32),
        labels=np.asarray(eval_output["labels"], dtype=np.int32),
        pred_labels=np.asarray(pred["pred_labels"], dtype=np.int32),
        max_scores=np.asarray(pred["max_scores"], dtype=np.float32),
        true_scores=np.asarray(pred["true_scores"], dtype=np.float32),
        impostor_scores=np.asarray(pred["impostor_scores"], dtype=np.float32),
    )

    metrics = {
        "split": args.split,
        "top1": float(top1),
        "top5": float(top5) if top5 is not None else None,
        "threshold": threshold,
        "far": float(threshold_metrics["far"]),
        "frr": float(threshold_metrics["frr"]),
        "accept_rate": float(threshold_metrics["accept_rate"]),
        "eer": calibration.get("eer"),
        "per_user_accuracy": per_user_accuracy(
            true_labels=np.asarray(eval_output["labels"], dtype=np.int32).tolist(),
            pred_labels=np.asarray(pred["pred_labels"], dtype=np.int32).tolist(),
        ),
        "predictions_path": str(predictions_path),
    }

    save_json(run_test_dir / "test_metrics.json", metrics)
    save_json(
        run_test_dir / "calibration_scores.json",
        {
            "threshold": threshold,
            "num_genuine": int(np.asarray(calibration_source["true_scores"]).shape[0]),
            "num_impostor": int(np.asarray(calibration_source["impostor_scores"]).shape[0]),
        },
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Test artifacts: {run_test_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
