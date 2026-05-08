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

