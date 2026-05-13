from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from object_removal.io.masks import list_mask_files, read_mask_u8, to_binary_255, write_mask_u8


@dataclass(frozen=True)
class Params:
    mode: str = "intersection"  # "intersection" | "union"
    area_ratio_threshold: float = 2.0  # if xmem/sam3 area ratio > this, trust xmem (sam3 lost object)


def _fuse_mask(mask_a: np.ndarray, mask_b: np.ndarray, mode: str, area_ratio_threshold: float) -> np.ndarray:
    """Per-pixel voting between two binary {0,255} masks (mask_a = xmem, mask_b = sam3).

    When only one side has foreground, keep the non-empty one (no frame drop).
    When both have foreground but area differs by > threshold, trust xmem
    (sam3 tends to lose objects on hard frames).
    Otherwise, apply intersection / union.
    """
    bin_a = to_binary_255(mask_a)
    bin_b = to_binary_255(mask_b)
    has_a = bool((bin_a > 0).any())
    has_b = bool((bin_b > 0).any())

    if not has_a and not has_b:
        return bin_a  # zeros
    if not has_b:
        return bin_a
    if not has_a:
        return bin_b

    area_a = int((bin_a > 0).sum())
    area_b = int((bin_b > 0).sum())
    max_a = max(area_a, area_b)
    min_a = max(min(area_a, area_b), 1)
    if (max_a / min_a) > area_ratio_threshold:
        # large discrepancy → sam3 likely lost the object → trust xmem
        return bin_a

    if mode == "union":
        return np.where((bin_a > 0) | (bin_b > 0), 255, 0).astype(np.uint8)
    else:
        return np.where((bin_a > 0) & (bin_b > 0), 255, 0).astype(np.uint8)


def run(
    *,
    mask_dir_a: Path,
    mask_dir_b: Path,
    out_masks_dir: Path,
    params: Params,
) -> dict:
    files_a = {p.name: p for p in list_mask_files(mask_dir_a)}
    files_b = {p.name: p for p in list_mask_files(mask_dir_b)}
    all_names = sorted(set(files_a.keys()) | set(files_b.keys()))

    if not all_names:
        raise FileNotFoundError(f"No mask PNGs found in {mask_dir_a} or {mask_dir_b}")

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for name in all_names:
        pa = files_a.get(name)
        pb = files_b.get(name)
        if pa is None:
            write_mask_u8(out_masks_dir / name, read_mask_u8(pb))
        elif pb is None:
            write_mask_u8(out_masks_dir / name, read_mask_u8(pa))
        else:
            fused = _fuse_mask(read_mask_u8(pa), read_mask_u8(pb), params.mode, params.area_ratio_threshold)
            write_mask_u8(out_masks_dir / name, fused)

    return {
        "mode": params.mode,
        "area_ratio_threshold": params.area_ratio_threshold,
        "num_frames": len(all_names),
        "dir_a": str(mask_dir_a),
        "dir_b": str(mask_dir_b),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Vote-fusion track method")
    ap.add_argument("--mask_dir_a", required=True)
    ap.add_argument("--mask_dir_b", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--mode", default="intersection", choices=["intersection", "union"])
    ap.add_argument("--area_ratio_threshold", type=float, default=2.0)
    args = ap.parse_args()

    run(
        mask_dir_a=Path(args.mask_dir_a),
        mask_dir_b=Path(args.mask_dir_b),
        out_masks_dir=Path(args.out_masks_dir),
        params=Params(mode=str(args.mode), area_ratio_threshold=float(args.area_ratio_threshold)),
    )


if __name__ == "__main__":
    main()
