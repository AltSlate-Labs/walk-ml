#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ClipDataset, build_label_map, load_index, summarize_records
from src.data.splits import build_splits
from src.engine.train_loop import train_model
from src.models.gait_encoder_mlx import create_model, load_weights
from src.utils.io import prepare_run_dir, save_json, save_yaml, load_config, utc_timestamp
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLX gait identification model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-root", required=True, help="Dataset root path")
    parser.add_argument("--run-name", default=None, help="Run name under artifacts/")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config)
    data_root = Path(args.data_root).resolve()

    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 42))
    set_seed(seed)

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

    label_map = build_label_map(splits["train"] + splits["val"] + splits["test"])
    index_to_user = {index: user for user, index in label_map.items()}

    config["model"]["num_classes"] = len(label_map)
    config["project"]["seed"] = seed

    run_name = args.run_name or f"gait-mlx-{utc_timestamp()}"
    run_dir = prepare_run_dir(config["project"]["artifacts_root"], run_name)

    save_yaml(run_dir / "resolved_config.yaml", config)
    save_json(run_dir / "label_map.json", label_map)
    save_json(run_dir / "index_to_user.json", {str(k): v for k, v in index_to_user.items()})
    save_json(
        run_dir / "split_summary.json",
        {
            split: summarize_records(split_records)
            for split, split_records in splits.items()
        },
    )

    train_dataset = ClipDataset(
        records=splits["train"],
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=True,
        seed=seed,
    )
    gallery_dataset = ClipDataset(
        records=splits["train"],
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=False,
        seed=seed,
    )
    val_dataset = ClipDataset(
        records=splits["val"],
        label_map=label_map,
        seq_len=int(config["data"]["seq_len"]),
        image_size=config["data"]["image_size"],
        augment=False,
        seed=seed,
    )

    model = create_model(config, num_classes=len(label_map))
    if args.resume:
        load_weights(model, args.resume)

    summary = train_model(
        model=model,
        train_dataset=train_dataset,
        gallery_dataset=gallery_dataset,
        val_dataset=val_dataset,
        config=config,
        run_dir=run_dir,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
