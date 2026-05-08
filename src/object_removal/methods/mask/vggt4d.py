from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    dyn_threshold_scale: float = 0.7
    threshold: int = 0
    merge_all: bool = True
    split_cc: bool = True
    cc_min_area: int = 32
    cc_max_objects: int = 64
    cc_close_kernel: int = 0


def _extract_idx(name: str) -> int:
    stem = os.path.splitext(name)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 10**9


def _prep_binary_for_cc(binary: np.ndarray, close_kernel: int) -> np.ndarray:
    import cv2

    bw = (binary > 0).astype(np.uint8)
    k = int(close_kernel)
    if k <= 0:
        return bw
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)


def _count_components_ge_area(binary: np.ndarray, min_area: int, close_kernel: int = 0) -> int:
    import cv2

    bw = _prep_binary_for_cc(binary, close_kernel)
    if int(bw.sum()) == 0:
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    n = 0
    for lab in range(1, num_labels):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area:
            n += 1
    return n


def _split_binary_into_instances(
    binary: np.ndarray,
    min_area: int,
    max_objects: int,
    close_kernel: int = 0,
) -> Tuple[np.ndarray, int]:
    import cv2

    bw = _prep_binary_for_cc(binary, close_kernel)
    if int(bw.sum()) == 0:
        return bw.astype(np.uint8), 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        one = (bw > 0).astype(np.uint8)
        return one, int(one.max())

    cand: List[Tuple[int, int]] = []
    for lab in range(1, num_labels):
        a = int(stats[lab, cv2.CC_STAT_AREA])
        if a >= min_area:
            cand.append((lab, a))
    cand.sort(key=lambda x: -x[1])
    cand = cand[: max(1, max_objects)]

    if not cand:
        one = (bw > 0).astype(np.uint8)
        return one, 1

    out = np.zeros_like(bw, dtype=np.uint8)
    for new_id, (lab, _) in enumerate(cand, start=1):
        out[labels == lab] = new_id
    return out, int(out.max())


def _gen_init_mask_from_scene(scene_output: Path, out_dir: Path, p: Params) -> Path:
    import cv2

    candidates = [n for n in os.listdir(scene_output) if n.startswith("dynamic_mask_") and n.lower().endswith(".png")]
    if not candidates:
        raise RuntimeError(f"No dynamic_mask_*.png found in {scene_output}")

    thr = int(p.threshold)
    min_a = int(p.cc_min_area)
    ck = int(p.cc_close_kernel)

    if p.merge_all:
        merged = None
        per_frame: List[Tuple[str, np.ndarray]] = []
        for name in sorted(candidates, key=_extract_idx):
            arr = cv2.imread(str(scene_output / name), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            binary = (arr > thr).astype(np.uint8)
            per_frame.append((name, binary))
            merged = binary if merged is None else np.maximum(merged, binary)
        if merged is None:
            raise RuntimeError(f"Failed to read any dynamic mask from {scene_output}")

        n_merged = _count_components_ge_area(merged, min_a, ck)
        best_name = "merged_all_frames"
        best_indexed = merged
        best_idx = 0
        best_n = n_merged
        best_fg = int(merged.sum())

        for name, binary in per_frame:
            nf = _count_components_ge_area(binary, min_a, ck)
            fg = int(binary.sum())
            if nf > best_n or (nf == best_n and nf > 0 and fg > best_fg):
                best_n = nf
                best_fg = fg
                best_indexed = binary
                best_name = name
                best_idx = _extract_idx(name)
    else:
        best_name = None
        best_area = -1
        best_indexed = None
        best_n = -1
        for name in sorted(candidates, key=_extract_idx):
            arr = cv2.imread(str(scene_output / name), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            indexed = np.zeros_like(arr, dtype=np.uint8)
            indexed[arr > thr] = 1
            area = int((indexed > 0).sum())
            nf = _count_components_ge_area(indexed, min_a, ck)
            if nf > best_n or (nf == best_n and area > best_area):
                best_n = nf
                best_area = area
                best_name = name
                best_indexed = indexed

        if best_name is None or best_indexed is None:
            raise RuntimeError(f"Failed to read any dynamic mask from {scene_output}")
        best_idx = _extract_idx(best_name)

    if p.split_cc:
        best_indexed, _ = _split_binary_into_instances(
            best_indexed,
            min_area=min_a,
            max_objects=int(p.cc_max_objects),
            close_kernel=ck,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{best_idx:05d}.png"
    ok = cv2.imwrite(str(out_path), best_indexed.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to write init mask: {out_path}")
    return out_path


def run(*, repo_root: Path, frames_dir: Path, out_scene_dir: Path, out_init_dir: Path, params: Params) -> dict:
    """VGGT4D mask stage: run VGGT4D and build SAM3 init indexed mask."""
    # Import VGGT4D as a normal python module from modules/VGGT4D
    vggt_root = repo_root / "modules" / "VGGT4D"
    if not vggt_root.is_dir():
        raise FileNotFoundError(f"Missing VGGT4D module dir: {vggt_root}")
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))

    # VGGT demo expects input_dir containing scene folders.
    # We'll wrap the given frames_dir as a single scene: <tmp_input>/<scene_name> -> frames_dir
    scene_name = frames_dir.name
    tmp_input = out_scene_dir.parent / "vggt_input"
    scene_link = tmp_input / scene_name
    tmp_input.mkdir(parents=True, exist_ok=True)
    if scene_link.exists():
        if scene_link.is_symlink() or scene_link.is_dir():
            scene_link.unlink() if scene_link.is_symlink() else None
    if not scene_link.exists():
        os.symlink(str(frames_dir.resolve()), str(scene_link))

    import demo_vggt4d  # type: ignore

    out_scene_root = out_scene_dir.parent / "vggt_output"
    demo_vggt4d.main(str(tmp_input), str(out_scene_root), dyn_threshold_scale=float(params.dyn_threshold_scale))
    scene_out = out_scene_root / scene_name

    init_mask_path = _gen_init_mask_from_scene(scene_out, out_init_dir, params)
    return {
        "scene_out": str(scene_out),
        "init_mask": str(init_mask_path),
        "dyn_threshold_scale": params.dyn_threshold_scale,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VGGT4D mask method (produce init mask for tracking)")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_scene_dir", required=True)
    ap.add_argument("--out_init_dir", required=True)
    ap.add_argument("--dyn_threshold_scale", type=float, default=0.7)
    args = ap.parse_args()

    p = Params(dyn_threshold_scale=args.dyn_threshold_scale)
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        out_scene_dir=Path(args.out_scene_dir),
        out_init_dir=Path(args.out_init_dir),
        params=p,
    )


if __name__ == "__main__":
    main()

