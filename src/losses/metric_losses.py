from __future__ import annotations

from typing import Any, Dict, Tuple

from src.models.gait_encoder_mlx import require_mlx


def arcface_style_cross_entropy(
    logits: Any,
    labels: Any,
    margin: float,
    scale: float,
) -> Any:
    mx, _ = require_mlx()
    num_classes = logits.shape[1]
    labels = labels.astype(mx.int32)
    one_hot = mx.eye(num_classes, dtype=logits.dtype)[labels]
    adjusted = (logits - margin * one_hot) * scale
    log_probs = adjusted - mx.logsumexp(adjusted, axis=1, keepdims=True)
    loss = -mx.sum(one_hot * log_probs, axis=1)
    return mx.mean(loss)


def batch_hard_triplet_loss(embeddings: Any, labels: Any, margin: float) -> Any:
    mx, _ = require_mlx()

    similarity = embeddings @ embeddings.T
    distances = 1.0 - similarity

    labels = labels.astype(mx.int32)
    same_label = labels[:, None] == labels[None, :]
    eye = mx.eye(labels.shape[0], dtype=mx.bool_)
    positive_mask = mx.logical_and(same_label, mx.logical_not(eye))
    negative_mask = mx.logical_not(same_label)

    neg_inf = mx.array(-1e9, dtype=distances.dtype)
    pos_inf = mx.array(1e9, dtype=distances.dtype)

    hardest_positive = mx.max(mx.where(positive_mask, distances, neg_inf), axis=1)
    hardest_negative = mx.min(mx.where(negative_mask, distances, pos_inf), axis=1)

    raw_loss = mx.maximum(hardest_positive - hardest_negative + margin, 0.0)
    valid = (mx.sum(positive_mask.astype(mx.int32), axis=1) > 0).astype(distances.dtype)
    denom = mx.maximum(mx.sum(valid), 1.0)
    return mx.sum(raw_loss * valid) / denom


def combined_loss(
    embeddings: Any,
    logits: Any,
    labels: Any,
    loss_cfg: Dict[str, float],
) -> Tuple[Any, Dict[str, Any]]:
    id_loss = arcface_style_cross_entropy(
        logits,
        labels,
        margin=float(loss_cfg["arcface_margin"]),
        scale=float(loss_cfg["arcface_scale"]),
    )
    triplet_loss = batch_hard_triplet_loss(
        embeddings,
        labels,
        margin=float(loss_cfg["triplet_margin"]),
    )

    total = (
        float(loss_cfg["id_weight"]) * id_loss
        + float(loss_cfg["triplet_weight"]) * triplet_loss
    )
    return total, {"id_loss": id_loss, "triplet_loss": triplet_loss}
