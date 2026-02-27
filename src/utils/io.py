from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    extends = config.pop("extends", None)
    if not extends:
        return config
    parent = (config_path.parent / extends).resolve()
    base_config = load_config(parent)
    return deep_merge(base_config, config)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def prepare_run_dir(artifacts_root: str | Path, run_name: str) -> Path:
    root = Path(artifacts_root)
    candidate = root / run_name
    if candidate.exists():
        candidate = root / f"{run_name}-{utc_timestamp()}"
    (candidate / "checkpoints").mkdir(parents=True, exist_ok=True)
    (candidate / "metrics").mkdir(parents=True, exist_ok=True)
    (candidate / "test").mkdir(parents=True, exist_ok=True)
    return candidate


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), sort_keys=True))
        handle.write("\n")
