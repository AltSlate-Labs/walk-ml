from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
except ImportError:
    mx = None
    nn = None


def require_mlx() -> Tuple[Any, Any]:
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is required but not installed. Install with: pip install mlx"
        )
    return mx, nn


def _resolve_model_seq_len(config: Dict[str, Any]) -> int:
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    data_seq_len = data_cfg.get("seq_len")
    model_seq_len = model_cfg.get("seq_len")

    if data_seq_len is None and model_seq_len is None:
        raise KeyError("Missing seq_len in config. Set data.seq_len (and optionally model.seq_len).")

    if data_seq_len is None:
        return int(model_seq_len)
    if model_seq_len is None:
        return int(data_seq_len)

    data_seq_len_int = int(data_seq_len)
    model_seq_len_int = int(model_seq_len)
    if data_seq_len_int != model_seq_len_int:
        raise ValueError(
            "Config mismatch: data.seq_len="
            f"{data_seq_len_int} and model.seq_len={model_seq_len_int}. "
            "These must match to avoid encoder shape errors."
        )
    return data_seq_len_int


if nn is not None:

    class GaitEncoder(nn.Module):
        def __init__(
            self,
            seq_len: int,
            input_channels: int,
            hidden_dim: int,
            embedding_dim: int,
            num_classes: int,
        ) -> None:
            super().__init__()
            self.seq_len = int(seq_len)
            self.input_channels = int(input_channels)
            self.feature_dim = self.seq_len * self.input_channels
            self.fc1 = nn.Linear(self.feature_dim, int(hidden_dim))
            self.fc2 = nn.Linear(int(hidden_dim), int(embedding_dim))
            self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

        def encode(self, clips: Any) -> Any:
            pooled = mx.mean(clips, axis=(2, 3))
            flat = pooled.reshape((pooled.shape[0], -1))
            hidden = self.fc1(flat)
            hidden = mx.maximum(hidden, 0.0)
            embeddings = self.fc2(hidden)
            norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True) + 1e-12)
            return embeddings / norm

        def __call__(self, clips: Any) -> Tuple[Any, Any]:
            embeddings = self.encode(clips)
            logits = self.classifier(embeddings)
            return embeddings, logits

else:

    class GaitEncoder:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MLX is required but not installed. Install with: pip install mlx"
            )


def create_model(config: Dict[str, Any], num_classes: int) -> GaitEncoder:
    model_cfg = config["model"]
    seq_len = _resolve_model_seq_len(config)
    return GaitEncoder(
        seq_len=seq_len,
        input_channels=int(model_cfg.get("input_channels", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        num_classes=int(num_classes),
    )


def save_weights(model: GaitEncoder, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_weights"):
        raise RuntimeError("Model does not support save_weights in current runtime")
    model.save_weights(str(target))


def load_weights(model: GaitEncoder, path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found: {target}")
    if not hasattr(model, "load_weights"):
        raise RuntimeError("Model does not support load_weights in current runtime")
    model.load_weights(str(target))
