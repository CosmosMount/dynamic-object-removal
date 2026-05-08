from __future__ import annotations

from pathlib import Path
from typing import List


def frames_to_mp4(frames_dir: Path, out_mp4: Path, *, fps: float = 24.0) -> None:
    import cv2
    import numpy as np

    files = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        raise ValueError(f"No frames found under: {frames_dir}")

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {files[0]}")
    h, w = first.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_mp4}")
    try:
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read: {p}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            writer.write(img)
    finally:
        writer.release()


def masks_to_mp4(masks_dir: Path, out_mp4: Path, *, fps: float = 24.0) -> None:
    import cv2
    import numpy as np

    files = sorted([p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if not files:
        raise ValueError(f"No masks found under: {masks_dir}")

    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Failed to read: {files[0]}")
    h, w = first.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_mp4}")
    try:
        for p in files:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Failed to read: {p}")
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def mp4_to_frames(video_path: Path, out_dir: Path, *, ext: str = ".png") -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = out_dir / f"{i:05d}{ext}"
            ok2 = cv2.imwrite(str(out), frame)
            if not ok2:
                raise RuntimeError(f"Failed to write frame: {out}")
            i += 1
    finally:
        cap.release()

