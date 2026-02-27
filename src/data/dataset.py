from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ClipRecord:
    clip_path: str
    user_id: str
    session_id: str
    camera_id: str
    timestamp: str
    quality_score: float


def load_index(index_path: str | Path, data_root: str | Path, min_quality: float = 0.0) -> List[ClipRecord]:
    index_path = Path(index_path)
    if not index_path.is_absolute():
        index_path = Path(data_root) / index_path
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    records: List[ClipRecord] = []
    with index_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_path", "user_id", "session_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required index columns: {sorted(missing)}")

        for row in reader:
            quality = float(row.get("quality_score") or 0.0)
            if quality < min_quality:
                continue
            clip_path = Path(row["clip_path"])
            if not clip_path.is_absolute():
                clip_path = (Path(data_root) / clip_path).resolve()
            records.append(
                ClipRecord(
                    clip_path=str(clip_path),
                    user_id=str(row["user_id"]),
                    session_id=str(row["session_id"]),
                    camera_id=str(row.get("camera_id") or "unknown"),
                    timestamp=str(row.get("timestamp") or ""),
                    quality_score=quality,
                )
            )

    if not records:
        raise ValueError("No records were loaded from index.csv after filtering.")
    return records


def build_label_map(records: Sequence[ClipRecord]) -> Dict[str, int]:
    user_ids = sorted({record.user_id for record in records})
    return {user_id: index for index, user_id in enumerate(user_ids)}


def _load_clip_array(clip_path: str | Path) -> np.ndarray:
    path = Path(clip_path)
    if not path.exists():
        raise FileNotFoundError(f"Clip file not found: {path}")

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "frames" in payload:
                clip = payload["frames"]
            else:
                first_key = payload.files[0]
                clip = payload[first_key]
    elif path.suffix.lower() == ".npy":
        clip = np.load(path, allow_pickle=False)
    else:
        raise ValueError(
            f"Unsupported clip format for {path}. Expected .npz or .npy"
        )

    if clip.ndim == 3:
        clip = clip[..., np.newaxis]
    if clip.ndim != 4:
        raise ValueError(
            f"Expected clip shape [T,H,W,C] or [T,H,W], got {clip.shape} for {path}"
        )
    if clip.shape[-1] == 1:
        clip = np.repeat(clip, 3, axis=-1)

    clip = clip.astype(np.float32)
    if clip.max() > 1.0:
        clip = clip / 255.0
    return np.clip(clip, 0.0, 1.0)


def _resize_nearest(clip: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    _, h, w, _ = clip.shape
    if h == target_h and w == target_w:
        return clip
    h_idx = np.linspace(0, h - 1, target_h).round().astype(np.int32)
    w_idx = np.linspace(0, w - 1, target_w).round().astype(np.int32)
    resized = np.take(clip, h_idx, axis=1)
    resized = np.take(resized, w_idx, axis=2)
    return resized


def _normalize_length(clip: np.ndarray, seq_len: int) -> np.ndarray:
    t = clip.shape[0]
    if t == seq_len:
        return clip
    if t > seq_len:
        start = max((t - seq_len) // 2, 0)
        return clip[start : start + seq_len]

    pad_count = seq_len - t
    if t == 0:
        raise ValueError("Encountered an empty clip with zero frames.")
    pad = np.repeat(clip[-1:, ...], pad_count, axis=0)
    return np.concatenate([clip, pad], axis=0)


def _augment_clip(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    brightness = rng.uniform(-0.08, 0.08)
    contrast = rng.uniform(0.9, 1.1)
    noise_std = rng.uniform(0.0, 0.02)

    augmented = (clip - 0.5) * contrast + 0.5 + brightness
    if noise_std > 0:
        augmented = augmented + rng.normal(0.0, noise_std, size=augmented.shape)
    return np.clip(augmented, 0.0, 1.0)


class ClipDataset:
    def __init__(
        self,
        records: Sequence[ClipRecord],
        label_map: Dict[str, int],
        seq_len: int,
        image_size: Sequence[int],
        augment: bool,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.label_map = label_map
        self.seq_len = int(seq_len)
        if len(image_size) != 2:
            raise ValueError("image_size must be [H, W]")
        self.target_h = int(image_size[0])
        self.target_w = int(image_size[1])
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        clip = _load_clip_array(record.clip_path)
        clip = _normalize_length(clip, self.seq_len)
        clip = _resize_nearest(clip, self.target_h, self.target_w)
        if self.augment:
            clip = _augment_clip(clip, self.rng)

        label = self.label_map.get(record.user_id)
        if label is None:
            raise KeyError(f"Unknown user_id {record.user_id} not found in label map")

        return {
            "clip": clip.astype(np.float32),
            "label": np.int32(label),
            "user_id": record.user_id,
            "session_id": record.session_id,
            "clip_path": record.clip_path,
        }


def iter_batches(
    dataset: ClipDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[Dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(dataset))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        clips: List[np.ndarray] = []
        labels: List[np.int32] = []
        user_ids: List[str] = []
        session_ids: List[str] = []
        clip_paths: List[str] = []

        for index in chunk:
            sample = dataset[int(index)]
            clips.append(sample["clip"])  # type: ignore[arg-type]
            labels.append(sample["label"])  # type: ignore[arg-type]
            user_ids.append(str(sample["user_id"]))
            session_ids.append(str(sample["session_id"]))
            clip_paths.append(str(sample["clip_path"]))

        yield {
            "clips": np.stack(clips, axis=0).astype(np.float32),
            "labels": np.array(labels, dtype=np.int32),
            "user_ids": user_ids,
            "session_ids": session_ids,
            "clip_paths": clip_paths,
        }


def summarize_records(records: Iterable[ClipRecord]) -> Dict[str, int]:
    user_ids = set()
    sessions = set()
    count = 0
    for record in records:
        count += 1
        user_ids.add(record.user_id)
        sessions.add(record.session_id)
    return {
        "num_clips": count,
        "num_users": len(user_ids),
        "num_sessions": len(sessions),
    }
