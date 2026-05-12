from __future__ import annotations

"""
Vendored XMem batch tracker with optional overlap-anchor bidirectional refinement.

Changes:
- Default YOLO path prefers repo_root/ckpts/yolo/yolov8n-seg.pt.
- Optional overlap-anchor path (`--xmem-*`): full forward, mirrored full reverse from
  last-frame mask, pick max-overlap keyframe, intersect seed, bidirectional re-track from anchor.
- Directly uses XMem `BaseTracker` without any interactive wrapper layer.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_FILENAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
XMEM_URL = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
XMEM_FILENAME = "XMem-s012.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless XMem mask export with automatic init mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_root", required=True, help="Repo root (to resolve default weights).")
    parser.add_argument("--frame_dir", required=True, help="Input frame directory.")
    parser.add_argument("--raw_mask_dir", required=True, help="Output indexed mask PNG directory.")
    parser.add_argument("--binary_mask_dir", default=None, help="Optional output binary 0/255 PNG directory.")
    parser.add_argument("--vis_dir", default=None, help="Optional output painted tracking frames.")
    parser.add_argument("--xmem_root", required=True, help="Directory that contains the vendored XMem runtime files.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam_model_type", default="vit_h", choices=sorted(SAM_FILENAMES.keys()))
    parser.add_argument("--sam_checkpoint", default=None)
    parser.add_argument("--xmem_checkpoint", default=None)
    parser.add_argument("--init_source", default="yolo", choices=["yolo", "sam_auto", "mask"])
    parser.add_argument("--init_mask", default=None, help="Existing first-frame mask for --init_source mask.")
    parser.add_argument("--yolo_model", default=None, help="YOLO segmentation checkpoint or model name.")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--classes", default="all")
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--max_area_ratio", type=float, default=0.80)
    parser.add_argument("--sam_points_per_side", type=int, default=32)
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument(
        "--xmem-bidirectional",
        action="store_true",
        help="Enable overlap-anchor refinement (same gate as non-off two_stage_anchor_idx).",
    )
    parser.add_argument(
        "--xmem-bidirectional-merge",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="Deprecated; kept for CLI compatibility. Overlap-anchor no longer fuses this way.",
    )
    parser.add_argument(
        "--xmem-two-stage-anchor-idx",
        type=str,
        default="-1",
        help="'-1' off (single forward only unless --xmem-bidirectional); 'auto' = max overlap fwd/rev; int = fixed anchor.",
    )
    parser.add_argument("--xmem-two-stage-auto-samples", type=int, default=7)
    parser.add_argument("--xmem-two-stage-auto-max-fg-frac", type=float, default=0.92)
    parser.add_argument("--xmem-two-stage-auto-min-fg-frac", type=float, default=0.00008)
    parser.add_argument("--xmem-two-stage-auto-min-fg-pixels", type=int, default=64)
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


def resolve_checkpoints(args: argparse.Namespace) -> Tuple[Optional[str], str]:
    ckpt_dir = os.path.join(args.repo_root, "ckpts", "xmem")
    xmem_checkpoint = args.xmem_checkpoint or os.path.join(ckpt_dir, XMEM_FILENAME)
    sam_checkpoint: Optional[str] = None
    if args.init_source == "sam_auto":
        sam_checkpoint = args.sam_checkpoint or os.path.join(ckpt_dir, SAM_FILENAMES[args.sam_model_type])
        sam_checkpoint = maybe_download(SAM_URLS[args.sam_model_type], sam_checkpoint, args.no_download)
    xmem_checkpoint = maybe_download(XMEM_URL, xmem_checkpoint, args.no_download)
    return sam_checkpoint, xmem_checkpoint


def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_bool_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def add_component(indexed: np.ndarray, candidate: np.ndarray, obj_id: int, min_area: int, max_area: int) -> bool:
    candidate = candidate.astype(bool)
    area = int(candidate.sum())
    if area < min_area or area > max_area:
        return False
    write_region = candidate & (indexed == 0)
    if int(write_region.sum()) < min_area:
        return False
    indexed[write_region] = obj_id
    return True


def parse_classes(classes: str) -> Optional[List[int]]:
    if classes.strip().lower() in {"", "all", "none"}:
        return None
    return [int(x) for x in classes.split(",") if x.strip()]


def build_yolo_init_mask(args: argparse.Namespace, first_frame_path: str) -> np.ndarray:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for --init_source yolo") from exc

    frame_bgr = cv2.imread(first_frame_path)
    if frame_bgr is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")
    h, w = frame_bgr.shape[:2]
    min_area = int(h * w * args.min_area_ratio)
    max_area = int(h * w * args.max_area_ratio)
    indexed = np.zeros((h, w), dtype=np.uint8)

    default_yolo = os.path.join(args.repo_root, "ckpts", "yolo", "yolov8n-seg.pt")
    model_path = args.yolo_model or (default_yolo if os.path.isfile(default_yolo) else "yolov8n-seg.pt")
    model = YOLO(model_path)
    res = model(frame_bgr, classes=parse_classes(args.classes), conf=args.yolo_conf, verbose=False)[0]

    candidates: List[Tuple[int, np.ndarray]] = []
    if res.masks is not None and len(res.masks.data) > 0:
        masks = res.masks.data.cpu().numpy() > 0.5
        for mask in masks:
            m = resize_bool_mask(mask, h, w)
            candidates.append((int(m.sum()), m))
    elif res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box.tolist()
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            m = np.zeros((h, w), dtype=bool)
            m[y1:y2, x1:x2] = True
            candidates.append((int(m.sum()), m))

    for _, mask in sorted(candidates, key=lambda item: item[0], reverse=True):
        next_id = int(indexed.max()) + 1
        if next_id > max(1, min(args.max_objects, 255)):
            break
        add_component(indexed, mask, next_id, min_area, max_area)
    return indexed


def build_sam_auto_init_mask(args: argparse.Namespace, first_rgb: np.ndarray, sam_checkpoint: str) -> np.ndarray:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import torch

    h, w = first_rgb.shape[:2]
    min_area = int(h * w * args.min_area_ratio)
    max_area = int(h * w * args.max_area_ratio)
    indexed = np.zeros((h, w), dtype=np.uint8)

    sam = sam_model_registry[args.sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.sam_points_per_side,
        min_mask_region_area=max(0, min_area),
    )
    masks = generator.generate(first_rgb)
    masks.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    for item in masks:
        next_id = int(indexed.max()) + 1
        if next_id > max(1, min(args.max_objects, 255)):
            break
        add_component(indexed, item["segmentation"], next_id, min_area, max_area)

    del generator
    del sam
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return indexed


def load_init_mask(args: argparse.Namespace, h: int, w: int) -> np.ndarray:
    if not args.init_mask:
        raise ValueError("--init_mask is required when --init_source mask")
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


def _anchor_seed_indexed(fwd_i: np.ndarray, rev_i: np.ndarray) -> np.ndarray:
    """Per-object-id intersection; empty intersection falls back to forward mask."""
    seed = np.zeros_like(fwd_i, dtype=np.uint8)
    ids = {int(x) for x in np.unique(fwd_i).tolist()} | {int(x) for x in np.unique(rev_i).tolist()}
    for oid in sorted(x for x in ids if x > 0):
        inter = (fwd_i == oid) & (rev_i == oid)
        seed[inter] = np.uint8(oid)
    if int((seed > 0).sum()) == 0:
        return fwd_i.copy()
    return seed


def _merge_bidir_anchor_indexed(
    masks_left_rev_local: List[np.ndarray],
    masks_right_fwd_local: List[np.ndarray],
    anchor: int,
    t: int,
    seed: np.ndarray,
) -> List[np.ndarray]:
    """idx<anchor from prefix-reverse re-track; idx==anchor seed; idx>anchor from suffix-forward re-track."""
    merged: List[np.ndarray] = []
    for i in range(t):
        if i < anchor:
            k = anchor - i
            merged.append(masks_left_rev_local[k].copy())
        elif i > anchor:
            merged.append(masks_right_fwd_local[i - anchor].copy())
        else:
            merged.append(seed.copy())
    return merged


def _stitch_painted_bidir(
    painted_left_rev: List[np.ndarray],
    painted_right_fwd: List[np.ndarray],
    anchor: int,
    t: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(t):
        if i < anchor:
            out.append(painted_left_rev[anchor - i].copy())
        else:
            out.append(painted_right_fwd[i - anchor].copy())
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

    sam_checkpoint, xmem_checkpoint = resolve_checkpoints(args)

    if args.init_source == "yolo":
        template_mask = build_yolo_init_mask(args, frame_paths[0])
    elif args.init_source == "sam_auto":
        if not sam_checkpoint:
            raise RuntimeError("SAM checkpoint was not resolved for --init_source sam_auto")
        template_mask = build_sam_auto_init_mask(args, first_rgb, sam_checkpoint)
    else:
        template_mask = load_init_mask(args, h, w)

    if template_mask.shape != (h, w):
        raise RuntimeError(f"Init mask shape {template_mask.shape} does not match first frame {(h, w)}")
    if int((template_mask > 0).sum()) == 0:
        raise RuntimeError("Automatic init mask is empty.")

    ts_mode, ts_anchor_req = _parse_two_stage_anchor(str(args.xmem_two_stage_anchor_idx))
    use_overlap_anchor = ts_mode != "off" or bool(args.xmem_bidirectional)

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

        if not use_overlap_anchor:
            working_masks, working_painted = run_generator(images, template_mask)
        else:
            masks_fwd, painted_fwd = run_generator(images, template_mask)
            masks_rev_loc, painted_rev_loc = run_generator(list(reversed(images)), masks_fwd[-1].copy())
            masks_rev_global = [masks_rev_loc[t - 1 - j] for j in range(t)]

            anchor, overlap_areas = _pick_anchor_max_overlap(masks_fwd, masks_rev_global, ts_mode, ts_anchor_req)
            print(
                f"[xmem] overlap-anchor idx={anchor} (mode={ts_mode}); "
                f"per-frame overlap pixels (first/last/max): "
                f"{overlap_areas[0] if overlap_areas else 0}/"
                f"{overlap_areas[-1] if overlap_areas else 0}/"
                f"{max(overlap_areas) if overlap_areas else 0}"
            )

            seed = _anchor_seed_indexed(masks_fwd[anchor], masks_rev_global[anchor])
            rev_prefix = list(reversed(images[: anchor + 1]))
            suf = images[anchor:]
            masks_left, painted_left = run_generator(rev_prefix, seed)
            masks_right, painted_right = run_generator(suf, seed)
            working_masks = _merge_bidir_anchor_indexed(masks_left, masks_right, anchor, t, seed)
            if args.vis_dir:
                working_painted = _stitch_painted_bidir(painted_left, painted_right, anchor, t)
            else:
                working_painted = list(painted_fwd)
            print(
                "[xmem] overlap-anchor merge: idx<anchor from prefix-reverse, "
                "idx==anchor seed (intersection or forward fallback), idx>anchor from suffix-forward."
            )
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
