from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    dyn_threshold_scale: float = 0.7
    threshold: int = 0
    tail_policy: str = "hold_last"  # hold_last|zeros
    # Same cap as vggt4d: VGGT loads all selected frames on GPU; subsample then map masks to full timeline.
    max_frames_for_vggt: int = 20


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


def _nearest_sample_index(full_frame_idx: int, sample_orig_indices: list[int]) -> int:
    """Index into VGGT mask list (same order as subsampled frames passed to VGGT)."""
    if not sample_orig_indices:
        return 0
    best_k = 0
    best_d = abs(sample_orig_indices[0] - full_frame_idx)
    for k, oi in enumerate(sample_orig_indices):
        d = abs(int(oi) - full_frame_idx)
        if d < best_d:
            best_d = d
            best_k = k
        elif d == best_d and int(oi) > int(sample_orig_indices[best_k]):
            best_k = k
    return best_k


def run(*, repo_root: Path, frames_dir: Path, out_masks_dir: Path, params: Params) -> dict:
    """Run VGGT4D and export binary masks for every frame (00000.png ...)."""
    import cv2

    vggt_root = repo_root / "modules" / "VGGT4D"
    if not vggt_root.is_dir():
        raise FileNotFoundError(f"Missing VGGT4D module dir: {vggt_root}")
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    import demo_vggt4d  # type: ignore

    scene_name = frames_dir.name
    tmp_input = out_masks_dir.parent / "vggt_input"
    tmp_input.mkdir(parents=True, exist_ok=True)
    scene_dir = tmp_input / scene_name
    if scene_dir.is_symlink():
        scene_dir.unlink()
    elif scene_dir.is_dir():
        for p in scene_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
            else:
                p.unlink(missing_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    all_frame_paths = _list_frames(frames_dir)
    if not all_frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    max_n = int(params.max_frames_for_vggt)
    if max_n > 0 and len(all_frame_paths) > max_n:
        idxs = np.linspace(0, len(all_frame_paths) - 1, num=max_n, dtype=int).tolist()
        sample_orig_indices = sorted({int(i) for i in idxs})
        frame_files = [all_frame_paths[i] for i in sample_orig_indices]
    else:
        sample_orig_indices = list(range(len(all_frame_paths)))
        frame_files = all_frame_paths

    for src in frame_files:
        dst = scene_dir / src.name
        if not dst.exists():
            os.symlink(str(src.resolve()), str(dst))

    out_root = out_masks_dir.parent / "vggt_output"
    demo_vggt4d.main(str(tmp_input), str(out_root), dyn_threshold_scale=float(params.dyn_threshold_scale))
    scene_out = out_root / scene_name
    if not scene_out.is_dir():
        raise FileNotFoundError(f"VGGT output scene dir missing: {scene_out}")

    num_masks = _vggt_mask_count(scene_out)
    thr = int(params.threshold)

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_path in enumerate(all_frame_paths):
        ref = cv2.imread(str(frame_path))
        if ref is None:
            raise RuntimeError(f"Cannot read frame: {frame_path}")
        h, w = ref.shape[:2]

        if num_masks == 0:
            out = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_masks_dir / f"{i:05d}.png"), out)
            continue

        v_slot = _nearest_sample_index(i, sample_orig_indices)
        vidx = min(v_slot, num_masks - 1)

        vpath = scene_out / f"dynamic_mask_{vidx:04d}.png"
        arr = cv2.imread(str(vpath), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Cannot read VGGT mask: {vpath}")
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = ((arr > thr).astype(np.uint8)) * 255
        cv2.imwrite(str(out_masks_dir / f"{i:05d}.png"), binary)

    return {
        "scene_out": str(scene_out),
        "num_frames": len(all_frame_paths),
        "num_vggt_masks": num_masks,
        "max_frames_for_vggt": max_n,
        "sample_orig_indices": sample_orig_indices,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VGGT4D framewise binary mask export")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--dyn_threshold_scale", type=float, default=0.7)
    ap.add_argument("--max_frames_for_vggt", type=int, default=20)
    args = ap.parse_args()
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        out_masks_dir=Path(args.out_masks_dir),
        params=Params(
            dyn_threshold_scale=float(args.dyn_threshold_scale),
            max_frames_for_vggt=int(args.max_frames_for_vggt),
        ),
    )


if __name__ == "__main__":
    main()
