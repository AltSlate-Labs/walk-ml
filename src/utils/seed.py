from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx  # type: ignore

        mx.random.seed(seed)
    except Exception:
        # MLX may not be installed in every environment.
        pass
