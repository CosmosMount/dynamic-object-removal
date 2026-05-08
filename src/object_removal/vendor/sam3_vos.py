#!/usr/bin/env python3
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


def _ensure_sam3_on_path(repo_root: str) -> None:
    sam3_dir = os.path.join(repo_root, "modules", "sam3")
    if os.path.isdir(sam3_dir) and sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)


DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 VOS inference with init mask prompt.")
    parser.add_argument("--repo_root", required=True, help="Repo root (for modules/sam3 on sys.path).")
    parser.add_argument("--sam3_checkpoint", required=True, help="Local SAM3 checkpoint path.")
    parser.add_argument("--base_video_dir", required=True, help="Root dir containing frame folders for each video.")
    parser.add_argument("--input_mask_dir", required=True, help="Input init indexed masks dir.")
    parser.add_argument("--output_mask_dir", required=True, help="Output propagated indexed masks dir.")
    parser.add_argument("--video_list_file", default=None, help="Optional txt with one video name per line.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Threshold on SAM3 logits.")
    parser.add_argument(
        "--two_stage_anchor_idx",
        type=str,
        default="-1",
        help="Two-stage SAM3: '-1' off; int = fixed anchor; 'auto' = sample stage1 masks along time, "
        "drop near-full-frame and tiny speckles, then prefer smaller fg among survivors (ties -> later frame). "
        "Merged output: idx<=anchor from stage2, idx>anchor from stage1.",
    )
    parser.add_argument(
        "--two_stage_auto_samples",
        type=int,
        default=7,
        help="With auto anchor: evenly spaced candidates (clamped to 5–7 and T).",
    )
    parser.add_argument(
        "--two_stage_auto_max_fg_frac",
        type=float,
        default=0.92,
        help="Auto anchor: ignore candidates whose stage1 fg covers more than this fraction of the frame (near full-screen).",
    )
    parser.add_argument(
        "--two_stage_auto_min_fg_frac",
        type=float,
        default=0.00008,
        help="Auto anchor: ignore candidates with fg area below max(min_fg_pixels, min_fg_frac * frame_pixels).",
    )
    parser.add_argument(
        "--two_stage_auto_min_fg_pixels",
        type=int,
        default=64,
        help="Auto anchor: minimum fg pixels (used with min_fg_frac).",
    )
    parser.add_argument(
        "--instance_stats_csv",
        type=str,
        default="",
        help="If set, write per-frame SAM3 instance counts (obj_id tracks) to this CSV path.",
    )
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
    if palette is not None:
        out.putpalette(palette)
    else:
        out.putpalette(DAVIS_PALETTE)
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

    pngs = sorted(
        [p for p in os.listdir(video_mask_dir) if p.lower().endswith(".png")],
        key=lambda n: int(os.path.splitext(n)[0]) if os.path.splitext(n)[0].isdigit() else n,
    )
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


def _compose_canvas(per_obj: Dict[int, np.ndarray], video_h: int, video_w: int) -> np.ndarray:
    canvas = np.zeros((video_h, video_w), dtype=np.uint8)
    for obj_id in sorted(per_obj.keys(), reverse=True):
        m = per_obj[obj_id].reshape(video_h, video_w)
        canvas[m] = np.uint8(obj_id)
    return canvas


def _foreground_area(per_obj: Dict[int, np.ndarray], video_h: int, video_w: int) -> int:
    if not per_obj:
        return 0
    return int((_compose_canvas(per_obj, video_h, video_w) > 0).sum())


def _bbox_stripe_and_fill(binary: np.ndarray, video_h: int, video_w: int) -> Tuple[bool, float]:
    """Detect near full-width / full-height thin bands (common wrong background masks). Returns (stripe, fill)."""
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return False, 0.0
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    area = int(binary.sum())
    fr_w = bw / float(max(1, video_w))
    fr_h = bh / float(max(1, video_h))
    stripe_h = fr_w >= 0.86 and fr_h <= 0.34
    stripe_v = fr_h >= 0.86 and fr_w <= 0.34
    stripe = stripe_h or stripe_v
    fill = area / float(max(1, bw * bh))
    return stripe, float(fill)


def _pick_min_area_tie_later(pairs: List[Tuple[int, int]]) -> int:
    best_idx, best_a = pairs[0][0], pairs[0][1]
    for idx, a in pairs:
        if a < best_a or (a == best_a and idx > best_idx):
            best_idx, best_a = idx, a
    return best_idx


def pick_auto_anchor(
    outputs1: Dict[int, Dict[int, np.ndarray]],
    frame_names: List[str],
    video_h: int,
    video_w: int,
    num_samples: int,
    max_fg_frac: float,
    min_fg_frac: float,
    min_fg_pixels: int,
) -> Tuple[int, List[Tuple[int, int, float, bool, float]]]:
    """Evenly spaced candidates; drop huge/tiny fg; drop stripe-like full-frame bands; then min area, tie→later."""
    T = len(frame_names)
    k = min(max(5, min(int(num_samples), 7)), T) if T >= 5 else T
    raw = np.linspace(0, T - 1, num=k, dtype=float)
    cand = sorted({min(T - 1, max(0, int(round(float(x))))) for x in raw})
    pix = int(video_h * video_w)
    max_a = max(1, int(float(max_fg_frac) * pix))
    min_a = max(int(min_fg_pixels), int(float(min_fg_frac) * pix))

    rows: List[Tuple[int, int, float, bool, float]] = []
    for idx in cand:
        per = outputs1.get(idx, {})
        if not per:
            rows.append((idx, 0, 0.0, False, 0.0))
            continue
        cv = _compose_canvas(per, video_h, video_w)
        binm = cv > 0
        a = int(binm.sum())
        stripe, fill = _bbox_stripe_and_fill(binm, video_h, video_w)
        rows.append((idx, a, a / float(pix), stripe, fill))

    fallback_idx = cand[len(cand) // 2]

    def pool(min_on: bool, max_on: bool, allow_stripe: bool) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for idx, a, _, stripe, _fill in rows:
            if a <= 0:
                continue
            if max_on and a > max_a:
                continue
            if min_on and a < min_a:
                continue
            if not allow_stripe and stripe:
                continue
            out.append((idx, a))
        return out

    # Prefer non-stripe when possible, then same min/max relax chain as before.
    for allow_stripe in (False, True):
        for min_on, max_on in ((True, True), (False, True), (True, False)):
            p = pool(min_on=min_on, max_on=max_on, allow_stripe=allow_stripe)
            if p:
                return _pick_min_area_tie_later(p), rows

    pos = [(idx, a) for idx, a, _, _, _ in rows if a > 0]
    if pos:
        return _pick_min_area_tie_later(pos), rows
    return fallback_idx, rows


def merge_prefix_stage2(
    outputs1: Dict[int, Dict[int, np.ndarray]],
    outputs2: Dict[int, Dict[int, np.ndarray]],
    anchor: int,
    num_frames: int,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Frames idx<=anchor from stage2; idx>anchor from stage1 (avoids stage2 hurting later frames)."""
    merged: Dict[int, Dict[int, np.ndarray]] = {}
    for idx in range(num_frames):
        if idx <= anchor:
            src = outputs2 if idx in outputs2 else outputs1
            merged[idx] = dict(src.get(idx, {}))
        else:
            src = outputs1 if idx in outputs1 else outputs2
            merged[idx] = dict(src.get(idx, {}))
    return merged


def propagate_video(
    predictor,
    video_dir: str,
    frame_names: List[str],
    init_mask_path: str,
    score_thresh: float,
    video_name: str,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], int, int, List[int]]:
    first_mask, object_ids, palette = load_mask(init_mask_path)
    if len(object_ids) == 0:
        raise RuntimeError(f"No foreground object in init mask: {init_mask_path}")

    init_frame_name = os.path.splitext(os.path.basename(init_mask_path))[0]
    if init_frame_name not in frame_names:
        raise RuntimeError(
            f"Init mask frame {init_frame_name}.png not found in video frames for {video_name}"
        )
    init_frame_idx = frame_names.index(init_frame_name)

    print(f"[sam3] {video_name}: init {init_frame_name} (idx={init_frame_idx}) path={init_mask_path}")

    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    video_h = int(inference_state["video_height"])
    video_w = int(inference_state["video_width"])
    if first_mask.shape != (video_h, video_w):
        resized = Image.fromarray(first_mask, mode="L").resize((video_w, video_h), resample=Image.NEAREST)
        first_mask = np.array(resized).astype(np.uint8)

    for obj_id in object_ids:
        obj_mask = torch.from_numpy(first_mask == obj_id)
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=init_frame_idx,
            obj_id=int(obj_id),
            mask=obj_mask,
        )

    outputs: Dict[int, Dict[int, np.ndarray]] = {}
    for reverse in (False, True):
        for out_frame_idx, out_obj_ids, _, out_video_res_masks, _ in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=init_frame_idx,
            max_frame_num_to_track=len(frame_names),
            reverse=reverse,
            propagate_preflight=(not reverse),
        ):
            per_obj: Dict[int, np.ndarray] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                per_obj[int(out_obj_id)] = (out_video_res_masks[i] > score_thresh).cpu().numpy()
            outputs[int(out_frame_idx)] = per_obj

    return outputs, video_h, video_w, palette


def save_outputs_to_dir(
    outputs: Dict[int, Dict[int, np.ndarray]],
    frame_names: List[str],
    video_h: int,
    video_w: int,
    save_dir: str,
    video_name: str,
    palette: List[int],
) -> None:
    out_root = os.path.join(save_dir, video_name)
    os.makedirs(out_root, exist_ok=True)
    for frame_idx, per_obj in outputs.items():
        canvas = _compose_canvas(per_obj, video_h, video_w)
        out_name = frame_names[frame_idx]
        save_indexed_mask(os.path.join(out_root, f"{out_name}.png"), canvas, palette)


def write_instance_stats_csv(
    outputs: Dict[int, Dict[int, np.ndarray]],
    frame_names: List[str],
    video_h: int,
    video_w: int,
    csv_path: str,
    video_name: str,
) -> None:
    """One row per frame: SAM3 track count (len(per_obj)), not semantic object count."""
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "frame_stem", "n_instances", "obj_ids", "fg_pixels"])
        for idx in range(len(frame_names)):
            per = outputs.get(idx, {})
            ids = sorted(per.keys())
            fg = _foreground_area(per, video_h, video_w) if per else 0
            w.writerow([video_name, idx, frame_names[idx], len(ids), ";".join(str(i) for i in ids), fg])
    print(f"[sam3] {video_name}: instance stats -> {csv_path}")


def _parse_two_stage_anchor(raw: str) -> Tuple[str, int]:
    """Returns (mode, anchor) where mode is 'off'|'fixed'|'auto'. anchor only for fixed."""
    s = raw.strip().lower()
    if s in ("", "-1", "none", "off"):
        return "off", -1
    if s == "auto":
        return "auto", -1
    try:
        v = int(s, 10)
    except ValueError as e:
        raise ValueError(f"Invalid --two_stage_anchor_idx: {raw!r} (use -1, auto, or a non-negative int)") from e
    if v < 0:
        return "off", -1
    return "fixed", v


def run_one_video(
    predictor,
    base_video_dir: str,
    input_mask_dir: str,
    output_mask_dir: str,
    video_name: str,
    score_thresh: float,
    two_stage_anchor_spec: str,
    two_stage_auto_samples: int,
    two_stage_auto_max_fg_frac: float,
    two_stage_auto_min_fg_frac: float,
    two_stage_auto_min_fg_pixels: int,
    instance_stats_csv: str,
) -> None:
    video_dir = os.path.join(base_video_dir, video_name)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video frame dir not found: {video_dir}")

    frame_names = list_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"No frames found in {video_dir}")

    mode, anchor_req = _parse_two_stage_anchor(two_stage_anchor_spec)
    init_mask_path = resolve_init_mask_path(input_mask_dir, video_name)
    outputs1, video_h, video_w, palette = propagate_video(
        predictor, video_dir, frame_names, init_mask_path, score_thresh, video_name
    )

    if mode == "off":
        save_outputs_to_dir(outputs1, frame_names, video_h, video_w, output_mask_dir, video_name, palette)
        if instance_stats_csv:
            write_instance_stats_csv(
                outputs1, frame_names, video_h, video_w, instance_stats_csv, video_name
            )
        return

    if mode == "auto":
        anchor, cand_rows = pick_auto_anchor(
            outputs1,
            frame_names,
            video_h,
            video_w,
            two_stage_auto_samples,
            two_stage_auto_max_fg_frac,
            two_stage_auto_min_fg_frac,
            two_stage_auto_min_fg_pixels,
        )
        print(
            f"[sam3] {video_name}: auto anchor idx={anchor} "
            f"(candidates idx,pixels,frac,stripe,fill: "
            f"{[(i, int(a), round(f, 4), st, round(fl, 3)) for i, a, f, st, fl in cand_rows]})"
        )
    else:
        anchor = min(max(0, anchor_req), len(frame_names) - 1)
        print(f"[sam3] {video_name}: two-stage fixed anchor idx={anchor} (requested={anchor_req})")

    per_anchor = outputs1.get(anchor)
    if not per_anchor:
        print(f"[sam3] WARN: stage1 missing frame {anchor}; using stage1 only.")
        save_outputs_to_dir(outputs1, frame_names, video_h, video_w, output_mask_dir, video_name, palette)
        if instance_stats_csv:
            write_instance_stats_csv(
                outputs1, frame_names, video_h, video_w, instance_stats_csv, video_name
            )
        return

    canvas_anchor = _compose_canvas(per_anchor, video_h, video_w)
    if int((canvas_anchor > 0).sum()) == 0:
        print(f"[sam3] WARN: stage1 mask empty at anchor {anchor}; using stage1 only.")
        save_outputs_to_dir(outputs1, frame_names, video_h, video_w, output_mask_dir, video_name, palette)
        if instance_stats_csv:
            write_instance_stats_csv(
                outputs1, frame_names, video_h, video_w, instance_stats_csv, video_name
            )
        return

    tmp_root = tempfile.mkdtemp(prefix=f"sam3_two_stage_{video_name}_", dir=os.path.dirname(output_mask_dir))
    try:
        s2_input = os.path.join(tmp_root, video_name)
        os.makedirs(s2_input, exist_ok=True)
        anchor_name = frame_names[anchor]
        stage2_init_path = os.path.join(s2_input, f"{anchor_name}.png")
        save_indexed_mask(stage2_init_path, canvas_anchor, palette)

        outputs2, h2, w2, _pal2 = propagate_video(
            predictor, video_dir, frame_names, stage2_init_path, score_thresh, video_name
        )
        merged = merge_prefix_stage2(outputs1, outputs2, anchor, len(frame_names))
        save_outputs_to_dir(merged, frame_names, h2, w2, output_mask_dir, video_name, palette)
        if instance_stats_csv:
            write_instance_stats_csv(merged, frame_names, h2, w2, instance_stats_csv, video_name)
        print(
            f"[sam3] {video_name}: stage2 done (re-init {anchor_name}); "
            f"saved merge: idx<={anchor} from stage2, idx>{anchor} from stage1."
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    _ensure_sam3_on_path(repo_root)

    from sam3.model_builder import build_sam3_video_model  # type: ignore

    if not os.path.isfile(args.sam3_checkpoint):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")

    if args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        video_names = sorted(os.listdir(args.base_video_dir))
        video_names = [n for n in video_names if os.path.isdir(os.path.join(args.base_video_dir, n))]

    if not video_names:
        raise RuntimeError("No videos to process")

    model = build_sam3_video_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    for video_name in video_names:
        print(f"[sam3] processing {video_name}")
        run_one_video(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=float(args.score_thresh),
            two_stage_anchor_spec=str(args.two_stage_anchor_idx),
            two_stage_auto_samples=int(args.two_stage_auto_samples),
            two_stage_auto_max_fg_frac=float(args.two_stage_auto_max_fg_frac),
            two_stage_auto_min_fg_frac=float(args.two_stage_auto_min_fg_frac),
            two_stage_auto_min_fg_pixels=int(args.two_stage_auto_min_fg_pixels),
            instance_stats_csv=str(args.instance_stats_csv or "").strip(),
        )

    print(f"Done. SAM3 masks saved to: {args.output_mask_dir}")


if __name__ == "__main__":
    main()
