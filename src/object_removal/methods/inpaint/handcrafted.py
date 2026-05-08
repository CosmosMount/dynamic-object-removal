from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from object_removal.io.frames import read_bgr_frames, write_bgr_frames
from object_removal.io.masks import list_mask_files, read_mask_u8


@dataclass(frozen=True)
class BaselineInpaintParams:
    temp_bg_window: int = 40
    inpaint_mode: str = "both"  # temporal|spatial|both


def _temporal_fill(frames: List[np.ndarray], masks: List[np.ndarray], p: BaselineInpaintParams, mode: str) -> Tuple[List[np.ndarray], int]:
    import cv2

    n = len(frames)
    results = [frame.copy() for frame in frames]
    px_temporal = 0

    for i in range(n):
        mask = masks[i]
        if mask.max() == 0:
            continue

        filled = np.zeros(mask.shape, dtype=bool)
        neighbors = sorted(
            [j for j in range(max(0, i - p.temp_bg_window), min(n, i + p.temp_bg_window + 1)) if j != i],
            key=lambda j: abs(j - i),
        )

        target_px = int((mask > 0).sum())
        for j in neighbors:
            if int(filled.sum()) == target_px:
                break
            borrow = (~filled) & (mask > 0) & (masks[j] == 0)
            if borrow.any():
                results[i][borrow] = frames[j][borrow]
                filled[borrow] = True

        px_temporal += int(filled.sum())

        if mode == "both":
            residual = ((mask > 0) & ~filled).astype(np.uint8) * 255
            if residual.max() > 0:
                results[i] = cv2.inpaint(results[i], residual, 5, cv2.INPAINT_TELEA)

        if (i + 1) % 30 == 0:
            print(f"   {i + 1}/{n}")

    return results, px_temporal


def _spatial_only(frames: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    import cv2

    results = [frame.copy() for frame in frames]
    px_spatial = 0
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        if mask.max() == 0:
            continue
        results[i] = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
        px_spatial += int(mask.sum() // 255)
        if (i + 1) % 30 == 0:
            print(f"   {i + 1}/{len(frames)}")
    return results, px_spatial


def run(*, frames_dir: Path, masks_dir: Path, out_frames_dir: Path, params: BaselineInpaintParams) -> dict:
    frames = read_bgr_frames(frames_dir)
    mask_files = list_mask_files(masks_dir)
    if not mask_files:
        raise ValueError(f"No masks found under: {masks_dir}")
    masks = [read_mask_u8(p) for p in mask_files[: len(frames)]]

    mode = params.inpaint_mode
    if mode == "spatial":
        print("[Inpaint] Spatial-only inpainting (cv2.inpaint Telea) ...")
        results, px_spatial = _spatial_only(frames, masks)
        px_temporal = 0
    elif mode == "temporal":
        print(f"[Inpaint] Temporal-only propagation (window={params.temp_bg_window}) ...")
        results, px_temporal = _temporal_fill(frames, masks, params, "temporal")
        px_spatial = 0
    else:
        print(f"[Inpaint] Temporal propagation + spatial fallback (window={params.temp_bg_window}) ...")
        results, px_temporal = _temporal_fill(frames, masks, params, "both")
        total_masked = sum(int((m > 0).sum()) for m in masks if m.max() > 0)
        px_spatial = max(0, total_masked - px_temporal)

    write_bgr_frames(results, out_frames_dir, ext=".png")
    return {
        "num_frames": len(results),
        "px_temporal": int(px_temporal),
        "px_spatial": int(px_spatial),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Baseline handcrafted inpaint method")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_frames_dir", required=True)
    ap.add_argument("--mode", default="both", choices=["temporal", "spatial", "both"])
    ap.add_argument("--temp_bg_window", type=int, default=40)
    args = ap.parse_args()

    params = BaselineInpaintParams(temp_bg_window=args.temp_bg_window, inpaint_mode=args.mode)
    run(frames_dir=Path(args.frames_dir), masks_dir=Path(args.masks_dir), out_frames_dir=Path(args.out_frames_dir), params=params)


if __name__ == "__main__":
    main()

