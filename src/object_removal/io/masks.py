from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def list_mask_files(mask_dir: Path) -> List[Path]:
    if not mask_dir.is_dir():
        return []
    files = [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(files)


def read_mask_u8(path: Path) -> np.ndarray:
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return m


def write_mask_u8(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {path}")


def to_binary_255(mask_u8: np.ndarray) -> np.ndarray:
    """Convert any uint8 mask (indexed or binary) into canonical binary {0,255}."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    out = (mask_u8 > 0).astype(np.uint8) * 255
    return out


def sorted_video_frame_stems(frames_dir: Path) -> List[str]:
    """Stem names of RGB frames, ordered like SAM2 VOS (integer stems when possible)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return [p.stem for p in sorted(files, key=sort_key)]


def init_mask_nonempty_on_first_frame(frames_dir: Path, init_masks_dir: Path) -> bool:
    """True iff the first video frame has a PNG init mask with any foreground (value > 0)."""
    stems = sorted_video_frame_stems(frames_dir)
    if not stems:
        return False
    path = init_masks_dir / f"{stems[0]}.png"
    if not path.is_file():
        return False
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return False
    if arr.ndim == 2:
        return bool(np.any(arr > 0))
    return bool(np.any(arr[..., :3] > 0))


def _png_has_foreground(path: Path) -> bool:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return False
    if arr.ndim == 2:
        return bool(np.any(arr > 0))
    return bool(np.any(arr[..., :3] > 0))


def init_masks_dir_has_any_foreground_png(init_masks_dir: Path) -> bool:
    """True if any *.png under init_masks_dir has foreground (e.g. VGGT4d writes a single mask on a non-zero frame)."""
    if not init_masks_dir.is_dir():
        return False
    for path in sorted(init_masks_dir.glob("*.png")):
        if _png_has_foreground(path):
            return True
    return False


def write_eroded_binary_mask_pngs(*, src_dir: Path, dst_dir: Path, iterations: int) -> Path:
    """Copy mask PNGs from ``src_dir`` to ``dst_dir`` with ``iterations`` of 3×3 binary erosion.

    Foreground (value > 0) is treated as the inpaint hole; erosion shrinks that region.
    """
    import cv2

    dst_dir.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for path in list_mask_files(src_dir):
        m = read_mask_u8(path)
        b = to_binary_255(m)
        if iterations > 0:
            b = cv2.erode(b, kernel, iterations=iterations)
        write_mask_u8(dst_dir / path.name, b)
    return dst_dir


def init_masks_sufficient_for_track(frames_dir: Path, init_masks_dir: Path, track_method: str) -> bool:
    """Whether init masks are non-empty for this tracker (SAM2/optflow need frame 0; SAM3/identity allow any frame index)."""
    tm = (track_method or "").strip().lower()
    if tm in ("sam2", "optflow"):
        return init_mask_nonempty_on_first_frame(frames_dir, init_masks_dir)
    if tm in ("sam3", "identity"):
        return init_masks_dir_has_any_foreground_png(init_masks_dir)
    return init_mask_nonempty_on_first_frame(frames_dir, init_masks_dir)

