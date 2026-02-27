#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract small example NPZ clips and index.csv from sample videos"
    )
    parser.add_argument(
        "--video-dir",
        default="data/examples/raw_videos",
        help="Directory containing source video files",
    )
    parser.add_argument(
        "--output-root",
        default="data/examples",
        help="Root output directory for clips and index.csv",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Target frame count per extracted clip",
    )
    parser.add_argument("--height", type=int, default=128, help="Output frame height")
    parser.add_argument("--width", type=int, default=256, help="Output frame width")
    parser.add_argument("--fps", type=int, default=12, help="Decode FPS for extraction")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ clips and index.csv",
    )
    return parser.parse_args()


def _decode_frames(
    video_path: Path,
    width: int,
    height: int,
    fps: int,
) -> np.ndarray:
    filter_graph = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "format=rgb24"
    )
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filter_graph,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = height * width * 3
    if frame_size <= 0:
        raise ValueError("Invalid frame size computed from width/height")
    if len(raw) == 0 or (len(raw) % frame_size) != 0:
        raise ValueError(f"Failed to decode frames cleanly from: {video_path}")

    num_frames = len(raw) // frame_size
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3).copy()
    if frames.shape[0] == 0:
        raise ValueError(f"No decoded frames from: {video_path}")
    return frames


def _window_with_pad(frames: np.ndarray, start: int, seq_len: int) -> np.ndarray:
    total = int(frames.shape[0])
    start = max(0, min(int(start), max(total - 1, 0)))
    end = min(start + seq_len, total)
    clip = frames[start:end]
    if clip.shape[0] >= seq_len:
        return clip[:seq_len]
    pad_count = seq_len - clip.shape[0]
    pad_frame = clip[-1:] if clip.shape[0] > 0 else frames[:1]
    pad = np.repeat(pad_frame, pad_count, axis=0)
    return np.concatenate([clip, pad], axis=0)


def _session_plan() -> List[tuple[str, float, str]]:
    # Include two validation-session clips per user so default split logic can sample val.
    return [
        ("s1", 0.0, "1"),
        ("s2a", 0.33, "2"),
        ("s2b", 0.66, "2"),
        ("s3", 1.0, "3"),
    ]


def _user_id_from_filename(video_path: Path) -> str:
    stem = video_path.stem.lower()
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return f"user_{sanitized}"


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir).resolve()
    output_root = Path(args.output_root).resolve()
    clips_dir = output_root / "clips"
    index_path = output_root / "index.csv"

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_paths = sorted(
        [
            p
            for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".ogv", ".mov", ".mkv"}
        ]
    )
    if not video_paths:
        raise ValueError(f"No videos found in: {video_dir}")

    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for video_path in video_paths:
        user_id = _user_id_from_filename(video_path)
        frames = _decode_frames(
            video_path=video_path,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
        )
        total_frames = int(frames.shape[0])
        max_start = max(total_frames - int(args.seq_len), 0)

        for tag, fraction, session_id in _session_plan():
            start = int(round(max_start * fraction))
            clip = _window_with_pad(frames, start=start, seq_len=int(args.seq_len))
            clip_name = f"{video_path.stem}_{tag}.npz"
            clip_path = clips_dir / clip_name
            if clip_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Clip already exists: {clip_path}. Use --overwrite to replace."
                )

            np.savez_compressed(clip_path, frames=clip.astype(np.uint8))
            rows.append(
                {
                    "clip_path": str(Path("clips") / clip_name),
                    "user_id": user_id,
                    "session_id": session_id,
                    "camera_id": "example_cam_1",
                    "timestamp": "",
                    "quality_score": "1.0",
                }
            )

    if index_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index already exists: {index_path}. Use --overwrite to replace."
        )

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_path",
                "user_id",
                "session_id",
                "camera_id",
                "timestamp",
                "quality_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to: {clips_dir}")
    print(f"Wrote index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
