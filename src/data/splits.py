from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from src.data.dataset import ClipRecord


def _filter_by_sessions(records: Sequence[ClipRecord], sessions: Sequence[str]) -> List[ClipRecord]:
    allowed = {str(session) for session in sessions}
    return [record for record in records if str(record.session_id) in allowed]


def _sample_validation_records(
    candidates: Sequence[ClipRecord],
    val_fraction: float,
    seed: int,
) -> List[ClipRecord]:
    grouped: Dict[str, List[ClipRecord]] = defaultdict(list)
    for record in candidates:
        grouped[record.user_id].append(record)

    rng = np.random.default_rng(seed)
    chosen: List[ClipRecord] = []
    for user_id, items in grouped.items():
        if len(items) <= 1:
            continue
        n_val = max(1, int(round(len(items) * val_fraction)))
        n_val = min(n_val, len(items) - 1)
        selected_idx = rng.choice(len(items), size=n_val, replace=False)
        for idx in np.atleast_1d(selected_idx):
            chosen.append(items[int(idx)])

    return chosen


def validate_no_overlap(split_map: Dict[str, Sequence[ClipRecord]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, records in split_map.items():
        for record in records:
            if record.clip_path in seen:
                other = seen[record.clip_path]
                raise ValueError(
                    f"Split leakage detected: {record.clip_path} is in both {other} and {split_name}"
                )
            seen[record.clip_path] = split_name


def build_splits(
    records: Sequence[ClipRecord],
    train_sessions: Sequence[str],
    val_from_sessions: Sequence[str],
    test_sessions: Sequence[str],
    val_fraction: float,
    seed: int,
) -> Dict[str, List[ClipRecord]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    train_pool = _filter_by_sessions(records, train_sessions)
    test_records = _filter_by_sessions(records, test_sessions)
    val_candidates = _filter_by_sessions(train_pool, val_from_sessions)
    val_records = _sample_validation_records(val_candidates, val_fraction, seed)

    val_keys = {record.clip_path for record in val_records}
    train_records = [record for record in train_pool if record.clip_path not in val_keys]

    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    validate_no_overlap(split_map)

    if not split_map["train"]:
        raise ValueError("Train split is empty. Check session mapping in config.")
    if not split_map["val"]:
        raise ValueError("Validation split is empty. Increase val_fraction or capture more data.")
    if not split_map["test"]:
        raise ValueError("Test split is empty. Check test_sessions in config.")

    return split_map
