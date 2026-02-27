from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.data.dataset import ClipDataset, iter_batches
from src.engine.test_loop import build_centroid_gallery, collect_embeddings, predict_closed_set
from src.eval.calibration import calibrate_threshold
from src.eval.metrics import compute_threshold_metrics, topk_accuracy
from src.losses.metric_losses import combined_loss
from src.models.gait_encoder_mlx import require_mlx, save_weights
from src.utils.io import append_jsonl, save_json

try:
    import mlx.optimizers as optim  # type: ignore
except ImportError:
    optim = None


def _to_float(value) -> float:
    return float(np.asarray(value).item())


def _scheduled_lr(base_lr: float, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_optimizer_lr(optimizer, lr: float) -> None:
    if hasattr(optimizer, "learning_rate"):
        optimizer.learning_rate = lr
        return
    if hasattr(optimizer, "set_learning_rate"):
        optimizer.set_learning_rate(lr)


def evaluate_validation(
    model,
    gallery_dataset: ClipDataset,
    val_dataset: ClipDataset,
    batch_size: int,
    seed: int,
    top_k,
    target_far: Optional[float],
) -> Dict[str, object]:
    gallery_output = collect_embeddings(model, gallery_dataset, batch_size=batch_size, seed=seed)
    gallery = build_centroid_gallery(
        embeddings=np.asarray(gallery_output["embeddings"], dtype=np.float32),
        labels=np.asarray(gallery_output["labels"], dtype=np.int32),
    )

    val_output = collect_embeddings(model, val_dataset, batch_size=batch_size, seed=seed + 1)
    pred = predict_closed_set(
        embeddings=np.asarray(val_output["embeddings"], dtype=np.float32),
        true_labels=np.asarray(val_output["labels"], dtype=np.int32),
        gallery_labels=np.asarray(gallery["gallery_labels"], dtype=np.int32),
        gallery_vectors=np.asarray(gallery["gallery_vectors"], dtype=np.float32),
        top_k=top_k,
    )

    calibration = calibrate_threshold(
        genuine_scores=np.asarray(pred["true_scores"], dtype=np.float32),
        impostor_scores=np.asarray(pred["impostor_scores"], dtype=np.float32),
        target_far=target_far,
    )

    threshold_metrics = compute_threshold_metrics(
        max_scores=np.asarray(pred["max_scores"], dtype=np.float32),
        true_scores=np.asarray(pred["true_scores"], dtype=np.float32),
        impostor_scores=np.asarray(pred["impostor_scores"], dtype=np.float32),
        threshold=float(calibration["threshold"]),
    )

    metrics: Dict[str, object] = {
        "threshold": float(calibration["threshold"]),
        "far": float(threshold_metrics["far"]),
        "frr": float(threshold_metrics["frr"]),
        "accept_rate": float(threshold_metrics["accept_rate"]),
        "eer": float(calibration["eer"]),
    }

    labels = np.asarray(val_output["labels"], dtype=np.int32)
    for k in sorted(set(int(v) for v in top_k)):
        topk_preds = np.asarray(pred["topk"][k], dtype=np.int32)
        metrics[f"top{k}"] = topk_accuracy(topk_preds, labels)

    metrics["genuine_scores"] = np.asarray(pred["true_scores"], dtype=np.float32)
    metrics["impostor_scores"] = np.asarray(pred["impostor_scores"], dtype=np.float32)
    return metrics


def train_model(
    model,
    train_dataset: ClipDataset,
    gallery_dataset: ClipDataset,
    val_dataset: ClipDataset,
    config: Dict[str, object],
    run_dir: Path,
) -> Dict[str, object]:
    if optim is None:
        raise RuntimeError("MLX optimizers are required. Install with: pip install mlx")

    mx, nn = require_mlx()

    train_cfg = config["training"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    data_cfg = config["data"]

    epochs = int(train_cfg["epochs"])
    batch_size = int(data_cfg["batch_size"])
    base_lr = float(train_cfg["learning_rate"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))

    optimizer = optim.AdamW(
        learning_rate=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    def loss_fn(model_instance, clips, labels):
        embeddings, logits = model_instance(clips)
        total, _ = combined_loss(embeddings, logits, labels, loss_cfg)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    train_log_path = run_dir / "metrics" / "train_log.jsonl"
    val_log_path = run_dir / "metrics" / "val_metrics.jsonl"

    best_top1 = -1.0
    best_eer = float("inf")
    best_epoch = -1
    best_threshold = 0.0

    for epoch in range(epochs):
        lr = _scheduled_lr(base_lr, epoch, epochs, warmup_epochs)
        _set_optimizer_lr(optimizer, lr)

        batch_losses = []
        for step, batch in enumerate(
            iter_batches(train_dataset, batch_size=batch_size, shuffle=True, seed=epoch), start=1
        ):
            clips = mx.array(batch["clips"])
            labels = mx.array(batch["labels"])

            loss_value, grads = loss_and_grad(model, clips, labels)
            optimizer.update(model, grads)

            try:
                mx.eval(model.parameters(), optimizer.state)
            except Exception:
                try:
                    mx.eval(model.parameters())
                except Exception:
                    pass

            loss_float = _to_float(loss_value)
            batch_losses.append(loss_float)

            if step % int(train_cfg.get("log_every", 10)) == 0:
                append_jsonl(
                    train_log_path,
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "lr": lr,
                        "loss": loss_float,
                    },
                )

        mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        val_metrics = evaluate_validation(
            model=model,
            gallery_dataset=gallery_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            seed=epoch,
            top_k=eval_cfg.get("top_k", [1, 5]),
            target_far=float(train_cfg.get("target_far", eval_cfg.get("target_far", 0.01))),
        )

        epoch_val = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": mean_loss,
            "top1": float(val_metrics.get("top1", 0.0)),
            "top5": float(val_metrics.get("top5", 0.0)),
            "far": float(val_metrics.get("far", 0.0)),
            "frr": float(val_metrics.get("frr", 0.0)),
            "eer": float(val_metrics.get("eer", 1.0)),
            "threshold": float(val_metrics.get("threshold", 0.0)),
            "accept_rate": float(val_metrics.get("accept_rate", 0.0)),
        }
        append_jsonl(val_log_path, epoch_val)

        save_weights(model, run_dir / "checkpoints" / "last_weights.npz")

        improved = epoch_val["top1"] > best_top1 or (
            math.isclose(epoch_val["top1"], best_top1) and epoch_val["eer"] < best_eer
        )
        if improved:
            best_top1 = epoch_val["top1"]
            best_eer = epoch_val["eer"]
            best_epoch = epoch + 1
            best_threshold = epoch_val["threshold"]
            save_weights(model, run_dir / "checkpoints" / "best_weights.npz")
            np.savez(
                run_dir / "metrics" / "best_val_scores.npz",
                genuine_scores=np.asarray(val_metrics["genuine_scores"], dtype=np.float32),
                impostor_scores=np.asarray(val_metrics["impostor_scores"], dtype=np.float32),
            )
            save_json(run_dir / "metrics" / "best_val_metrics.json", epoch_val)

    summary = {
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_eer": best_eer,
        "best_threshold": best_threshold,
        "best_checkpoint": str(run_dir / "checkpoints" / "best_weights.npz"),
        "last_checkpoint": str(run_dir / "checkpoints" / "last_weights.npz"),
    }
    save_json(run_dir / "metrics" / "training_summary.json", summary)
    return summary
