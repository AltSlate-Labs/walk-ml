from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.data.dataset import ClipDataset, iter_batches
from src.models.gait_encoder_mlx import require_mlx


def collect_embeddings(
    model,
    dataset: ClipDataset,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    mx, _ = require_mlx()

    embeddings_list: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    user_ids: List[str] = []
    clip_paths: List[str] = []

    for batch in iter_batches(dataset, batch_size=batch_size, shuffle=False, seed=seed):
        clips = mx.array(batch["clips"])
        labels = np.asarray(batch["labels"], dtype=np.int32)

        embeddings, logits = model(clips)
        embeddings_np = np.asarray(embeddings)
        logits_np = np.asarray(logits)

        embeddings_list.append(embeddings_np)
        logits_list.append(logits_np)
        labels_list.append(labels)
        user_ids.extend(batch["user_ids"])  # type: ignore[arg-type]
        clip_paths.extend(batch["clip_paths"])  # type: ignore[arg-type]

    if not embeddings_list:
        raise ValueError("No embeddings produced from dataset.")

    return {
        "embeddings": np.concatenate(embeddings_list, axis=0),
        "logits": np.concatenate(logits_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
        "user_ids": user_ids,
        "clip_paths": clip_paths,
    }


def build_centroid_gallery(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        subset = embeddings[labels == label]
        centroid = subset.mean(axis=0)
        norm = np.linalg.norm(centroid) + 1e-12
        centroids.append(centroid / norm)
    gallery_vectors = np.vstack(centroids).astype(np.float32)
    return {"gallery_labels": unique_labels.astype(np.int32), "gallery_vectors": gallery_vectors}


def _topk_labels(similarity: np.ndarray, gallery_labels: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-similarity, axis=1)
    topk_idx = order[:, :k]
    return gallery_labels[topk_idx]


def predict_closed_set(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_vectors: np.ndarray,
    top_k: Sequence[int],
) -> Dict[str, object]:
    similarity = embeddings @ gallery_vectors.T
    best_gallery_pos = np.argmax(similarity, axis=1)
    pred_labels = gallery_labels[best_gallery_pos]
    max_scores = similarity[np.arange(similarity.shape[0]), best_gallery_pos]

    label_to_pos = {int(label): idx for idx, label in enumerate(gallery_labels.tolist())}
    true_label_set = {int(label) for label in np.unique(true_labels).tolist()}
    missing_labels = sorted(true_label_set.difference(label_to_pos.keys()))
    if missing_labels:
        preview = ", ".join(str(label) for label in missing_labels[:10])
        suffix = "..." if len(missing_labels) > 10 else ""
        raise ValueError(
            "Gallery is missing centroids for true labels: "
            f"{preview}{suffix}. "
            "Verify split/user coverage so probe identities exist in gallery."
        )
    true_pos = np.array([label_to_pos[int(label)] for label in true_labels], dtype=np.int32)
    true_scores = similarity[np.arange(similarity.shape[0]), true_pos]

    impostor_similarity = similarity.copy()
    impostor_similarity[np.arange(similarity.shape[0]), true_pos] = -np.inf
    max_impostor_scores = np.max(impostor_similarity, axis=1)

    topk = {}
    for k in sorted(set(int(v) for v in top_k)):
        topk[k] = _topk_labels(similarity, gallery_labels, k)

    return {
        "pred_labels": pred_labels.astype(np.int32),
        "max_scores": max_scores.astype(np.float32),
        "true_scores": true_scores.astype(np.float32),
        "impostor_scores": max_impostor_scores.astype(np.float32),
        "topk": topk,
    }


def build_prediction_rows(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_scores: np.ndarray,
    true_scores: np.ndarray,
    impostor_scores: np.ndarray,
    clip_paths: Sequence[str],
    index_to_user: Dict[int, str],
    topk: Dict[int, np.ndarray] | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    topk = topk or {}
    top5 = topk.get(5)
    for i in range(true_labels.shape[0]):
        truth = int(true_labels[i])
        pred = int(pred_labels[i])
        top5_hit = None
        if top5 is not None:
            top5_hit = int((top5[i] == truth).any())
        rows.append(
            {
                "clip_path": clip_paths[i],
                "true_label": truth,
                "pred_label": pred,
                "true_user_id": index_to_user[truth],
                "pred_user_id": index_to_user[pred],
                "max_score": float(max_scores[i]),
                "true_score": float(true_scores[i]),
                "impostor_score": float(impostor_scores[i]),
                "is_correct": int(truth == pred),
                "top5_hit": top5_hit,
            }
        )
    return rows
