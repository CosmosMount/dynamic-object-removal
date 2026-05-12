from __future__ import annotations

"""
Vendored XMem batch tracker with optional two-stage merge and bidirectional fusion.

The init mask is always provided from the mask pipeline stage via --init_mask.
Two-stage (`--xmem-two-stage-anchor-idx`): full forward, mirrored full reverse,
pick max-overlap keyframe, re-track prefix from forward-mask seed.
Bidirectional (`--xmem-bidirectional`): full reverse pass from last-frame mask,
then union/intersection fusion with forward result.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


XMEM_URL = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
XMEM_FILENAME = "XMem-s012.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless XMem mask export with mask-pipeline init mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_root", required=True, help="Repo root (to resolve default weights).")
    parser.add_argument("--frame_dir", required=True, help="Input frame directory.")
    parser.add_argument("--raw_mask_dir", required=True, help="Output indexed mask PNG directory.")
    parser.add_argument("--binary_mask_dir", default=None, help="Optional output binary 0/255 PNG directory.")
    parser.add_argument("--vis_dir", default=None, help="Optional output painted tracking frames.")
    parser.add_argument("--xmem_root", required=True, help="Directory that contains the vendored XMem runtime files.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--xmem_checkpoint", default=None)
    parser.add_argument("--init_mask", required=True, help="First-frame mask from mask pipeline (indexed PNG).")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument(
        "--xmem-bidirectional",
        action="store_true",
        help="After two-stage merge, run a full reverse pass and fuse with union/intersection.",
    )
    parser.add_argument(
        "--xmem-bidirectional-merge",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="How to fuse forward vs reverse masks in the bidirectional step.",
    )
    parser.add_argument(
        "--xmem-two-stage-anchor-idx",
        type=str,
        default="-1",
        help="'-1' off (single forward only); 'auto' = max overlap fwd/rev; int = fixed anchor frame.",
    )
    parser.add_argument("--xmem-two-stage-auto-samples", type=int, default=7)
    parser.add_argument("--xmem-two-stage-auto-max-fg-frac", type=float, default=0.92)
    parser.add_argument("--xmem-two-stage-auto-min-fg-frac", type=float, default=0.00008)
    parser.add_argument("--xmem-two-stage-auto-min-fg-pixels", type=int, default=64)
    parser.add_argument(
        "--xmem-two-stage-min-overlap-ratio",
        type=float,
        default=0.25,
        help="Per-object min overlap ratio (intersection/forward_area) to trust a forward object at anchor.",
    )
    return parser.parse_args()


def list_frames(frame_dir: str) -> List[str]:
    names = [
        os.path.join(frame_dir, name)
        for name in os.listdir(frame_dir)
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    names.sort(key=lambda p: os.path.basename(p))
    if not names:
        raise RuntimeError(f"No frames found in {frame_dir}")
    return names


def maybe_download(url: str, path: str, no_download: bool) -> str:
    if os.path.isfile(path):
        return path
    if no_download:
        raise FileNotFoundError(f"Checkpoint not found and --no_download is set: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[xmem] downloading {url} -> {path}")
    import urllib.request

    urllib.request.urlretrieve(url, path)
    return path


def resolve_checkpoints(args: argparse.Namespace) -> str:
    ckpt_dir = os.path.join(args.repo_root, "ckpts", "xmem")
    xmem_checkpoint = args.xmem_checkpoint or os.path.join(ckpt_dir, XMEM_FILENAME)
    return maybe_download(XMEM_URL, xmem_checkpoint, args.no_download)


def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_init_mask(args: argparse.Namespace, h: int, w: int) -> np.ndarray:
    if not args.init_mask:
        raise ValueError("--init_mask is required")
    arr = np.array(Image.open(args.init_mask))
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.shape != (h, w):
        arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((w, h), resample=Image.NEAREST))
    if arr.max() <= 1:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def write_masks(mask_dir: str, binary_dir: Optional[str], frame_paths: List[str], masks: List[np.ndarray]) -> None:
    os.makedirs(mask_dir, exist_ok=True)
    if binary_dir:
        os.makedirs(binary_dir, exist_ok=True)
    for frame_path, mask in zip(frame_paths, masks):
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        raw = mask.astype(np.uint8)
        Image.fromarray(raw, mode="L").save(os.path.join(mask_dir, f"{stem}.png"))
        if binary_dir:
            binary = (raw > 0).astype(np.uint8) * 255
            Image.fromarray(binary, mode="L").save(os.path.join(binary_dir, f"{stem}.png"))


def _parse_two_stage_anchor(raw: str) -> Tuple[str, int]:
    """Same semantics as SAM3: off|-1, auto, or non-negative int."""
    s = str(raw).strip().lower()
    if s in ("", "-1", "none", "off"):
        return "off", -1
    if s == "auto":
        return "auto", -1
    try:
        v = int(s, 10)
    except ValueError as e:
        raise ValueError(
            f"Invalid --xmem-two-stage-anchor-idx: {raw!r} (use -1, auto, or a non-negative int)"
        ) from e
    if v < 0:
        return "off", -1
    return "fixed", v


def _overlap_areas_indexed(masks_fwd: List[np.ndarray], masks_rev: List[np.ndarray]) -> List[int]:
    return [int(np.logical_and(masks_fwd[i] > 0, masks_rev[i] > 0).sum()) for i in range(len(masks_fwd))]


def _pick_anchor_max_overlap(
    masks_fwd: List[np.ndarray],
    masks_rev_global: List[np.ndarray],
    ts_mode: str,
    ts_anchor_req: int,
) -> Tuple[int, List[int]]:
    t = len(masks_fwd)
    if ts_mode == "fixed":
        anchor = min(max(0, ts_anchor_req), t - 1)
        areas = _overlap_areas_indexed(masks_fwd, masks_rev_global)
        return anchor, areas
    areas = _overlap_areas_indexed(masks_fwd, masks_rev_global)
    max_a = max(areas) if areas else 0
    if max_a <= 0:
        return (t // 2) if t > 0 else 0, areas
    best = max(i for i, a in enumerate(areas) if a == max_a)
    return best, areas


def _anchor_seed_validated(
    fwd_i: np.ndarray,
    rev_i: np.ndarray,
    min_overlap_ratio: float,
) -> np.ndarray:
    """Per-object validation: keep forward region only for objects whose
    intersection with reverse covers >= min_overlap_ratio of the forward area.
    Drops objects that only appear in one direction (likely noise)."""
    seed = np.zeros_like(fwd_i, dtype=np.uint8)
    fwd_ids = {int(x) for x in np.unique(fwd_i).tolist() if x > 0}
    for oid in sorted(fwd_ids):
        fwd_region = fwd_i == oid
        fwd_area = int(fwd_region.sum())
        if fwd_area == 0:
            continue
        rev_region = rev_i == oid
        overlap = int((fwd_region & rev_region).sum())
        if overlap >= fwd_area * min_overlap_ratio:
            seed[fwd_region] = np.uint8(oid)
    if int((seed > 0).sum()) == 0:
        return fwd_i.copy()
    return seed


def _fuse_bidirectional_indexed(
    fwd: List[np.ndarray],
    rev_mapped: List[np.ndarray],
    merge: str,
) -> List[np.ndarray]:
    """Union: prefer forward, fill gaps with reverse. Intersection: keep only where both agree."""
    out: List[np.ndarray] = []
    for i in range(len(fwd)):
        a = fwd[i]
        b = rev_mapped[i]
        bp = a > 0
        br = b > 0
        if merge == "intersection":
            out.append(np.where(bp & br, a, 0).astype(np.uint8))
        else:
            u = np.where(bp, a, 0)
            u = np.where(~bp & br, b, u)
            out.append(u.astype(np.uint8))
    return out


def main() -> None:
    args = parse_args()
    args.repo_root = os.path.abspath(args.repo_root)
    args.xmem_root = os.path.abspath(args.xmem_root)
    args.frame_dir = os.path.abspath(args.frame_dir)
    args.raw_mask_dir = os.path.abspath(args.raw_mask_dir)
    if args.binary_mask_dir:
        args.binary_mask_dir = os.path.abspath(args.binary_mask_dir)
    if args.vis_dir:
        args.vis_dir = os.path.abspath(args.vis_dir)
    if args.init_mask:
        args.init_mask = os.path.abspath(args.init_mask)

    frame_paths = list_frames(args.frame_dir)
    first_rgb = read_rgb(frame_paths[0])
    h, w = first_rgb.shape[:2]

    xmem_checkpoint = resolve_checkpoints(args)

    template_mask = load_init_mask(args, h, w)

    if template_mask.shape != (h, w):
        raise RuntimeError(f"Init mask shape {template_mask.shape} does not match first frame {(h, w)}")
    if int((template_mask > 0).sum()) == 0:
        raise RuntimeError("Automatic init mask is empty.")

    ts_mode, ts_anchor_req = _parse_two_stage_anchor(str(args.xmem_two_stage_anchor_idx))
    do_bidir = bool(args.xmem_bidirectional)

    old_cwd = os.getcwd()
    sys.path.insert(0, args.xmem_root)
    os.chdir(args.xmem_root)
    try:
        from base_tracker import BaseTracker

        tracker = BaseTracker(xmem_checkpoint, device=args.device)
        images = [read_rgb(path) for path in frame_paths]
        t = len(images)

        def run_generator(im_list: List[np.ndarray], tmpl: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            masks: List[np.ndarray] = []
            painted: List[np.ndarray] = []
            for idx, image in enumerate(im_list):
                first = tmpl if idx == 0 else None
                mask, _logit, painted_image = tracker.track(image, first)
                masks.append(mask)
                painted.append(painted_image)
            tracker.clear_memory()
            return masks, painted

        # Determine which frame the init mask belongs to (e.g. VGGT4D 00006.png → frame 6).
        init_stem = os.path.splitext(os.path.basename(args.init_mask))[0]
        try:
            init_frame_idx = int(init_stem)
        except ValueError:
            init_frame_idx = 0

        # ---- Stage 1: forward pass from init mask ----
        if init_frame_idx > 0:
            images_from_init = images[init_frame_idx:]
            masks_fwd_suffix, painted_fwd_suffix = run_generator(images_from_init, template_mask)
            masks_fwd = [np.zeros((h, w), dtype=np.uint8) for _ in range(init_frame_idx)] + masks_fwd_suffix
            painted_fwd = [images[i].copy() for i in range(init_frame_idx)] + painted_fwd_suffix
        else:
            masks_fwd, painted_fwd = run_generator(images, template_mask)

        working_masks = [m.copy() for m in masks_fwd]
        working_painted = [p.copy() for p in painted_fwd]

        # ---- Stage 2 (optional): two-stage merge ----
        if ts_mode != "off":
            # Full reverse pass to compute overlap for anchor selection.
            masks_rev_loc, _painted_rev = run_generator(list(reversed(images)), working_masks[-1].copy())
            masks_rev_global = [masks_rev_loc[t - 1 - j] for j in range(t)]

            anchor, overlap_areas = _pick_anchor_max_overlap(masks_fwd, masks_rev_global, ts_mode, ts_anchor_req)
            print(
                f"[xmem] two-stage anchor idx={anchor} (mode={ts_mode}); "
                f"per-frame overlap pixels (first/last/max): "
                f"{overlap_areas[0] if overlap_areas else 0}/"
                f"{overlap_areas[-1] if overlap_areas else 0}/"
                f"{max(overlap_areas) if overlap_areas else 0}"
            )

            if int((masks_fwd[anchor] > 0).sum()) == 0:
                print(f"[xmem] WARN: forward mask empty at anchor {anchor}; keeping forward only.")
            else:
                seed = _anchor_seed_validated(
                    masks_fwd[anchor],
                    masks_rev_global[anchor],
                    float(args.xmem_two_stage_min_overlap_ratio),
                )
                rev_prefix = list(reversed(images[: anchor + 1]))
                masks_left, painted_left = run_generator(rev_prefix, seed)
                # idx < anchor from prefix re-track; idx >= anchor from forward.
                for k in range(len(masks_left)):
                    g = anchor - k
                    if 0 <= g < anchor:
                        working_masks[g] = masks_left[k].copy()
                if args.vis_dir:
                    for k in range(len(painted_left)):
                        g = anchor - k
                        if 0 <= g < anchor:
                            working_painted[g] = painted_left[k].copy()
                print(
                    f"[xmem] two-stage merge: idx<{anchor} from prefix-reverse re-track, "
                    f"idx>={anchor} from forward pass."
                )

        # ---- Stage 3 (optional): bidirectional fusion ----
        if do_bidir:
            rev_full = list(reversed(images))
            masks_rev_loc, _painted_rev = run_generator(rev_full, working_masks[-1].copy())
            masks_rev_global = [masks_rev_loc[t - 1 - j] for j in range(t)]
            merge_mode = str(args.xmem_bidirectional_merge).strip().lower()
            if merge_mode not in ("union", "intersection"):
                raise ValueError(f"Invalid bidirectional merge: {args.xmem_bidirectional_merge!r}")
            working_masks = _fuse_bidirectional_indexed(working_masks, masks_rev_global, merge_mode)
            print(f"[xmem] bidirectional fusion ({merge_mode}) applied.")
    finally:
        os.chdir(old_cwd)

    write_masks(args.raw_mask_dir, args.binary_mask_dir, frame_paths, working_masks)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        for frame_path, image in zip(frame_paths, working_painted):
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(args.vis_dir, f"{stem}.jpg"))


if __name__ == "__main__":
    main()
