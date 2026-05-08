from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    dyn_threshold_scale: float = 0.7
    threshold: int = 0
    tail_policy: str = "hold_last"  # hold_last|zeros


def _list_frames(frame_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _vggt_mask_count(vggt_scene_dir: Path) -> int:
    mx = -1
    for name in os.listdir(vggt_scene_dir):
        if not (name.startswith("dynamic_mask_") and name.lower().endswith(".png")):
            continue
        m = re.match(r"dynamic_mask_(\d+)\.png$", name, re.IGNORECASE)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 0


def run(*, repo_root: Path, frames_dir: Path, out_masks_dir: Path, params: Params) -> dict:
    """Run VGGT4D and export binary masks for every frame (00000.png ...)."""
    import cv2

    vggt_root = repo_root / "modules" / "VGGT4D"
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    import demo_vggt4d  # type: ignore

    # Prepare input as a single scene folder
    scene_name = frames_dir.name
    tmp_input = out_masks_dir.parent / "vggt_input"
    scene_link = tmp_input / scene_name
    tmp_input.mkdir(parents=True, exist_ok=True)
    if not scene_link.exists():
        os.symlink(str(frames_dir.resolve()), str(scene_link))

    out_root = out_masks_dir.parent / "vggt_output"
    demo_vggt4d.main(str(tmp_input), str(out_root), dyn_threshold_scale=float(params.dyn_threshold_scale))
    scene_out = out_root / scene_name
    if not scene_out.is_dir():
        raise FileNotFoundError(f"VGGT output scene dir missing: {scene_out}")

    frames = _list_frames(frames_dir)
    num_masks = _vggt_mask_count(scene_out)
    thr = int(params.threshold)

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_path in enumerate(frames):
        ref = cv2.imread(str(frame_path))
        if ref is None:
            raise RuntimeError(f"Cannot read frame: {frame_path}")
        h, w = ref.shape[:2]

        if num_masks == 0:
            out = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_masks_dir / f"{i:05d}.png"), out)
            continue

        if i < num_masks:
            vidx = i
        elif params.tail_policy == "hold_last":
            vidx = num_masks - 1
        else:
            out = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_masks_dir / f"{i:05d}.png"), out)
            continue

        vpath = scene_out / f"dynamic_mask_{vidx:04d}.png"
        arr = cv2.imread(str(vpath), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Cannot read VGGT mask: {vpath}")
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = ((arr > thr).astype(np.uint8)) * 255
        cv2.imwrite(str(out_masks_dir / f"{i:05d}.png"), binary)

    return {"scene_out": str(scene_out), "num_frames": len(frames), "num_vggt_masks": num_masks}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VGGT4D framewise binary mask export")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--dyn_threshold_scale", type=float, default=0.7)
    args = ap.parse_args()
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        out_masks_dir=Path(args.out_masks_dir),
        params=Params(dyn_threshold_scale=float(args.dyn_threshold_scale)),
    )


if __name__ == "__main__":
    main()

