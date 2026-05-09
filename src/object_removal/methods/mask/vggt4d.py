from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    # VGGT4D loads all frames into GPU memory; cap to the **first** N frames (same as object-removal vggt4dsam3*.sh).
    max_frames_for_vggt: int = 20
    # Full-sequence frame index: pick nearest subsampled VGGT slot, write init mask as NNNNN.png on that frame.
    init_frame: Optional[int] = None


def _slot_from_dynamic_mask_name(name: str) -> int:
    """VGGT writes `dynamic_mask_{slot:04d}.png` where slot is 0..T-1 over subsampled frames (not full-video index)."""
    m = re.match(r"dynamic_mask_(\d+)\.png$", name, re.IGNORECASE)
    if not m:
        return 10**9
    return int(m.group(1))


def _dynamic_mask_png_to_binary(path: Path, thr: int) -> Optional[np.ndarray]:
    """Foreground mask {0,1} from VGGT dynamic_mask PNG.

    IMREAD_GRAYSCALE drops alpha / mis-reads some palette+BGRA exports as all zeros;
    use IMREAD_UNCHANGED and combine channels + alpha.
    """
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 2:
        return ((arr > thr).astype(np.uint8))
    if arr.ndim != 3:
        return None
    ch = arr.shape[2]
    if ch >= 4:
        b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        rgb_max = np.maximum(np.maximum(b.astype(np.int16), g), r)
        fg = (rgb_max > thr) | (a > thr)
    else:
        fg = np.amax(arr[:, :, :ch], axis=2) > thr
    return fg.astype(np.uint8)


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


def _gen_init_mask_from_scene(
    scene_output: Path, out_dir: Path, p: Params, slot_to_orig: List[int]
) -> Path:
    """Build one indexed init mask for SAM3. Filenames use **full-sequence** frame indices (slot_to_orig maps VGGT slots)."""
    import cv2

    candidates = [n for n in os.listdir(scene_output) if n.startswith("dynamic_mask_") and n.lower().endswith(".png")]
    if not candidates:
        raise RuntimeError(f"No dynamic_mask_*.png found in {scene_output}")

    thr = int(p.threshold)
    min_a = int(p.cc_min_area)
    ck = int(p.cc_close_kernel)

    def _orig_for_slot(slot: int) -> int:
        if 0 <= slot < len(slot_to_orig):
            return int(slot_to_orig[slot])
        return int(slot)

    if p.init_frame is not None:
        want = int(p.init_frame)
        if not slot_to_orig:
            raise RuntimeError("vggt4d init_frame requires non-empty subsample mapping")
        best_slot = 0
        best_d = abs(_orig_for_slot(0) - want)
        for s in range(len(slot_to_orig)):
            d = abs(_orig_for_slot(s) - want)
            if d < best_d:
                best_d = d
                best_slot = s
        chosen = f"dynamic_mask_{best_slot:04d}.png"
        if chosen not in candidates:
            avail_orig = [
                _orig_for_slot(_slot_from_dynamic_mask_name(n))
                for n in sorted(candidates, key=_slot_from_dynamic_mask_name)
            ]
            raise RuntimeError(
                f"vggt4d init_frame={want}: expected mask file {chosen!r} missing. "
                f"Available orig-frame indices (from subsample): {avail_orig}"
            )
        bin0 = _dynamic_mask_png_to_binary(scene_output / chosen, thr)
        if bin0 is None or int(bin0.sum()) == 0:
            raise RuntimeError(f"Failed to read or empty dynamic mask: {scene_output / chosen}")
        best_indexed = bin0.astype(np.uint8)
        orig_out = _orig_for_slot(best_slot)
    elif p.merge_all:
        merged = None
        per_frame: List[Tuple[str, int, np.ndarray]] = []
        for name in sorted(candidates, key=_slot_from_dynamic_mask_name):
            slot = _slot_from_dynamic_mask_name(name)
            binary = _dynamic_mask_png_to_binary(scene_output / name, thr)
            if binary is None:
                continue
            per_frame.append((name, slot, binary))
            merged = binary if merged is None else np.maximum(merged, binary)
        if merged is None or not per_frame:
            raise RuntimeError(f"Failed to read any dynamic mask from {scene_output}")

        by_slot = {slot: binary for _name, slot, binary in per_frame}
        best_indexed: np.ndarray
        orig_out: int

        # Prefer VGGT slot 0 when it has foreground (BGRA-safe read); else competition / merged naming.
        b0 = by_slot.get(0)
        if b0 is not None and int((b0 > 0).sum()) > 0:
            best_indexed = b0
            orig_out = _orig_for_slot(0)
        else:
            n_merged = _count_components_ge_area(merged, min_a, ck)
            best_indexed = merged
            best_slot = max(per_frame, key=lambda t: int(t[2].sum()))[1]
            best_n = n_merged
            best_fg = int(merged.sum())

            for _name, slot, binary in per_frame:
                nf = _count_components_ge_area(binary, min_a, ck)
                fg = int(binary.sum())
                if nf > best_n or (nf == best_n and nf > 0 and fg > best_fg):
                    best_n = nf
                    best_fg = fg
                    best_indexed = binary
                    best_slot = slot

            if np.array_equal(best_indexed, merged):
                orig_out = _orig_for_slot(0)
            else:
                orig_out = _orig_for_slot(best_slot)
    else:
        best_area = -1
        best_indexed: Optional[np.ndarray] = None
        best_n = -1
        best_slot = 0
        for name in sorted(candidates, key=_slot_from_dynamic_mask_name):
            slot = _slot_from_dynamic_mask_name(name)
            binary = _dynamic_mask_png_to_binary(scene_output / name, thr)
            if binary is None:
                continue
            indexed = binary.astype(np.uint8)
            area = int((indexed > 0).sum())
            nf = _count_components_ge_area(indexed, min_a, ck)
            if nf > best_n or (nf == best_n and area > best_area):
                best_n = nf
                best_area = area
                best_indexed = indexed
                best_slot = slot

        if best_indexed is None:
            raise RuntimeError(f"Failed to read any dynamic mask from {scene_output}")
        orig_out = _orig_for_slot(best_slot)

    pre_split = best_indexed.copy()
    if p.split_cc:
        best_indexed, _ = _split_binary_into_instances(
            best_indexed,
            min_area=min_a,
            max_objects=int(p.cc_max_objects),
            close_kernel=ck,
        )
    if int((best_indexed > 0).sum()) == 0 and int((pre_split > 0).sum()) > 0:
        best_indexed = (pre_split > 0).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{orig_out:05d}.png"
    ok = cv2.imwrite(str(out_path), best_indexed.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to write init mask: {out_path}")
    return out_path


def run(*, repo_root: Path, frames_dir: Path, out_scene_dir: Path, out_init_dir: Path, params: Params) -> dict:
    """VGGT4D mask stage: run VGGT4D and build SAM3 init indexed mask."""
    vggt_root = repo_root / "modules" / "VGGT4D"
    if not vggt_root.is_dir():
        raise FileNotFoundError(f"Missing VGGT4D module dir: {vggt_root}")
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))

    scene_name = frames_dir.name
    tmp_input = out_scene_dir.parent / "vggt_input"
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

    all_frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not all_frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    max_n = int(params.max_frames_for_vggt)
    # Match object-removal vggt4dsam3*.sh when VGGT_MAX_FRAMES caps input: **first N frames** (not linspace).
    # Note: that shell can also run **chunked** VGGT over all frames when N is large; this path is one GPU batch
    # (OOM risk if you set max_frames_for_vggt very high on long clips).
    if max_n > 0 and len(all_frames) > max_n:
        n_use = max_n
        slot_to_orig = list(range(n_use))
        frame_files = all_frames[:n_use]
    else:
        slot_to_orig = list(range(len(all_frames)))
        frame_files = all_frames

    for src in frame_files:
        dst = scene_dir / src.name
        if not dst.exists():
            os.symlink(str(src.resolve()), str(dst))

    import demo_vggt4d  # type: ignore

    out_scene_root = out_scene_dir.parent / "vggt_output"
    demo_vggt4d.main(str(tmp_input), str(out_scene_root), dyn_threshold_scale=float(params.dyn_threshold_scale))
    scene_out = out_scene_root / scene_name

    init_mask_path = _gen_init_mask_from_scene(scene_out, out_init_dir, params, slot_to_orig)
    return {
        "scene_out": str(scene_out),
        "init_mask": str(init_mask_path),
        "dyn_threshold_scale": params.dyn_threshold_scale,
        "slot_to_orig": slot_to_orig,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="VGGT4D mask method (produce init mask for tracking)")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_scene_dir", required=True)
    ap.add_argument("--out_init_dir", required=True)
    ap.add_argument("--dyn_threshold_scale", type=float, default=0.7)
    ap.add_argument(
        "--init_frame",
        type=int,
        default=None,
        help="Full-sequence frame index: nearest VGGT subsample slot is used; init mask written as NNNNN.png on that frame.",
    )
    ap.add_argument(
        "--max_frames_for_vggt",
        type=int,
        default=20,
        help="Max frames passed to VGGT: first N frames of the clip (0 = all; may OOM).",
    )
    args = ap.parse_args()

    p = Params(
        dyn_threshold_scale=args.dyn_threshold_scale,
        init_frame=args.init_frame,
        max_frames_for_vggt=int(args.max_frames_for_vggt),
    )
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        out_scene_dir=Path(args.out_scene_dir),
        out_init_dir=Path(args.out_init_dir),
        params=p,
    )


if __name__ == "__main__":
    main()
