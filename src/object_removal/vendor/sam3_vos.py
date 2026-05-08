from __future__ import annotations

"""
Vendored from legacy: pipelines/vggt4dsam3/sam3_vos_inference.py

Changes:
- Resolve SAM3 path from repo_root/modules/sam3 (instead of external/sam3).
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0"


def _ensure_sam3_on_path(repo_root: str) -> None:
    sam3_dir = os.path.join(repo_root, "modules", "sam3")
    if os.path.isdir(sam3_dir) and sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 VOS inference with init mask prompt.")
    parser.add_argument("--repo_root", required=True, help="Repo root (to resolve modules/sam3).")
    parser.add_argument("--sam3_checkpoint", required=True, help="Local SAM3 checkpoint path.")
    parser.add_argument("--base_video_dir", required=True, help="Root dir containing frame folders for each video.")
    parser.add_argument("--input_mask_dir", required=True, help="Input init indexed masks dir.")
    parser.add_argument("--output_mask_dir", required=True, help="Output propagated indexed masks dir.")
    parser.add_argument("--video_list_file", default=None, help="Optional txt with one video name per line.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Threshold on SAM3 logits.")
    parser.add_argument("--two_stage_anchor_idx", type=str, default="-1")
    parser.add_argument("--two_stage_auto_samples", type=int, default=7)
    parser.add_argument("--two_stage_auto_max_fg_frac", type=float, default=0.92)
    parser.add_argument("--two_stage_auto_min_fg_frac", type=float, default=0.00008)
    parser.add_argument("--two_stage_auto_min_fg_pixels", type=int, default=64)
    parser.add_argument("--instance_stats_csv", type=str, default="")
    return parser.parse_args()


def load_mask(path: str) -> Tuple[np.ndarray, List[int], List[int]]:
    mask_img = Image.open(path)
    arr = np.array(mask_img)
    if arr.ndim > 2:
        arr = arr[..., 0]
    object_ids = [int(v) for v in np.unique(arr) if int(v) > 0]
    palette = mask_img.getpalette()
    return arr.astype(np.uint8), object_ids, palette


def save_indexed_mask(path: str, mask: np.ndarray, palette: List[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = Image.fromarray(mask.astype(np.uint8), mode="P")
    out.putpalette(palette if palette is not None else DAVIS_PALETTE)
    out.save(path)


def list_frame_names(video_dir: str) -> List[str]:
    names = []
    for name in os.listdir(video_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            names.append(os.path.splitext(name)[0])
    names.sort(key=lambda x: int(x) if x.isdigit() else x)
    return names


def resolve_init_mask_path(input_mask_dir: str, video_name: str) -> str:
    video_mask_dir = os.path.join(input_mask_dir, video_name)
    if not os.path.isdir(video_mask_dir):
        raise FileNotFoundError(f"Input mask dir not found: {video_mask_dir}")
    pngs = sorted([p for p in os.listdir(video_mask_dir) if p.lower().endswith(".png")])
    if not pngs:
        raise FileNotFoundError(f"No init masks found in {video_mask_dir}")
    best_path = None
    best_area = -1
    for name in pngs:
        path = os.path.join(video_mask_dir, name)
        arr = np.array(Image.open(path))
        if arr.ndim > 2:
            arr = arr[..., 0]
        area = int((arr > 0).sum())
        if area > best_area:
            best_area = area
            best_path = path
    if best_path is None:
        raise RuntimeError(f"Failed to select an init mask from {video_mask_dir}")
    return best_path


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    _ensure_sam3_on_path(repo_root)

    from sam3.model_builder import build_sam3_video_model  # type: ignore

    base_video_dir = args.base_video_dir
    input_mask_dir = args.input_mask_dir
    output_mask_dir = args.output_mask_dir
    os.makedirs(output_mask_dir, exist_ok=True)

    if args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_names = [x.strip() for x in f.read().splitlines() if x.strip()]
    else:
        video_names = [d for d in os.listdir(base_video_dir) if os.path.isdir(os.path.join(base_video_dir, d))]
        video_names.sort()

    model = build_sam3_video_model(checkpoint=args.sam3_checkpoint)
    model.eval()

    for video_name in video_names:
        video_dir = os.path.join(base_video_dir, video_name)
        frame_names = list_frame_names(video_dir)
        if not frame_names:
            continue

        init_path = resolve_init_mask_path(input_mask_dir, video_name)
        init_mask, obj_ids, palette = load_mask(init_path)
        if len(obj_ids) == 0:
            continue

        # Load frames into a temp dir because upstream script expects JPG list
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in frame_names:
                src = None
                for ext in (".jpg", ".jpeg", ".png"):
                    p = os.path.join(video_dir, name + ext)
                    if os.path.isfile(p):
                        src = p
                        break
                if src is None:
                    continue
                shutil.copy2(src, os.path.join(tmpdir, f"{name}.jpg"))

            outputs1, outputs2 = model.inference(
                video_dir=tmpdir,
                init_mask=init_mask,
                obj_ids=obj_ids,
                score_thresh=float(args.score_thresh),
                two_stage_anchor_idx=args.two_stage_anchor_idx,
                two_stage_auto_samples=int(args.two_stage_auto_samples),
                two_stage_auto_max_fg_frac=float(args.two_stage_auto_max_fg_frac),
                two_stage_auto_min_fg_frac=float(args.two_stage_auto_min_fg_frac),
                two_stage_auto_min_fg_pixels=int(args.two_stage_auto_min_fg_pixels),
            )

        # Compose masks (use stage2 when available)
        out_dir = os.path.join(output_mask_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)
        for t, name in enumerate(frame_names):
            per = outputs2.get(t) or outputs1.get(t) or {}
            canvas = np.zeros(init_mask.shape[:2], dtype=np.uint8)
            for obj_id in sorted(per.keys(), reverse=True):
                m = per[obj_id].reshape(init_mask.shape[0], init_mask.shape[1])
                canvas[m] = np.uint8(obj_id)
            save_indexed_mask(os.path.join(out_dir, f"{name}.png"), canvas, palette)

    if args.instance_stats_csv:
        with open(args.instance_stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["note", "instance_stats_csv_not_implemented_in_vendor"])


if __name__ == "__main__":
    main()

