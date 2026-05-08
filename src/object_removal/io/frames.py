from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


def list_frame_files(frames_dir: Path) -> List[Path]:
    if not frames_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def assert_contiguous_filenames(files: Iterable[Path], *, digits: int = 5, suffix: str | None = None) -> None:
    files = list(files)
    if not files:
        raise ValueError("No frame files found")
    for i, p in enumerate(files):
        expected = f"{i:0{digits}d}" + (suffix if suffix is not None else p.suffix)
        if p.name != expected:
            raise ValueError(f"Non-contiguous or unexpected filename at index {i}: got {p.name}, expected {expected}")


def infer_size_from_any_image(frames_dir: Path) -> Tuple[int, int]:
    import cv2

    files = list_frame_files(frames_dir)
    if not files:
        raise ValueError(f"No frames found under: {frames_dir}")
    img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {files[0]}")
    h, w = img.shape[:2]
    return int(w), int(h)


def read_bgr_frames(frames_dir: Path, *, max_frames: int | None = None) -> List["np.ndarray"]:
    import cv2
    import numpy as np

    files = list_frame_files(frames_dir)
    if not files:
        raise ValueError(f"No frames found under: {frames_dir}")
    if max_frames is not None:
        files = files[:max_frames]
    frames: List[np.ndarray] = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append(img)
    return frames


def write_bgr_frames(frames: List["np.ndarray"], out_dir: Path, *, ext: str = ".png") -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        path = out_dir / f"{i:05d}{ext}"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write frame: {path}")

