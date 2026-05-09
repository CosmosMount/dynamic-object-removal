from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.inpaint_stage import run_inpaint_stage


def _opt_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inpaint stage CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--masks_dir", required=True)
    p.add_argument("--method", required=True, choices=["baseline_handcrafted", "propainter", "diffueraser"])
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--diffueraser-video-length", type=int, default=None)
    p.add_argument("--diffueraser-mask-dilation-iter", type=int, default=None)
    p.add_argument("--diffueraser-max-img-size", type=int, default=None)
    p.add_argument("--diffueraser-ref-stride", type=int, default=None)
    p.add_argument("--diffueraser-neighbor-length", type=int, default=None)
    p.add_argument("--diffueraser-subvideo-length", type=int, default=None)
    p.add_argument("--propainter-resize-ratio", type=float, default=None)
    p.add_argument("--propainter-subvideo-length", type=int, default=None)
    p.add_argument("--propainter-neighbor-length", type=int, default=None)
    p.add_argument("--propainter-raft-iter", type=int, default=None)
    p.add_argument("--propainter-fp16", type=_opt_bool, default=None)
    p.add_argument("--propainter-save-frames", type=_opt_bool, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    diffueraser_options = None
    if args.method == "diffueraser":
        diffueraser_options = {}
        if args.diffueraser_video_length is not None:
            diffueraser_options["video_length"] = int(args.diffueraser_video_length)
        if args.diffueraser_mask_dilation_iter is not None:
            diffueraser_options["mask_dilation_iter"] = int(args.diffueraser_mask_dilation_iter)
        if args.diffueraser_max_img_size is not None:
            diffueraser_options["max_img_size"] = int(args.diffueraser_max_img_size)
        if args.diffueraser_ref_stride is not None:
            diffueraser_options["ref_stride"] = int(args.diffueraser_ref_stride)
        if args.diffueraser_neighbor_length is not None:
            diffueraser_options["neighbor_length"] = int(args.diffueraser_neighbor_length)
        if args.diffueraser_subvideo_length is not None:
            diffueraser_options["subvideo_length"] = int(args.diffueraser_subvideo_length)
        if not diffueraser_options:
            diffueraser_options = None
    propainter_options = None
    if args.method == "propainter":
        propainter_options = {}
        if args.propainter_resize_ratio is not None:
            propainter_options["resize_ratio"] = float(args.propainter_resize_ratio)
        if args.propainter_subvideo_length is not None:
            propainter_options["subvideo_length"] = int(args.propainter_subvideo_length)
        if args.propainter_neighbor_length is not None:
            propainter_options["neighbor_length"] = int(args.propainter_neighbor_length)
        if args.propainter_raft_iter is not None:
            propainter_options["raft_iter"] = int(args.propainter_raft_iter)
        if args.propainter_fp16 is not None:
            propainter_options["fp16"] = bool(args.propainter_fp16)
        if args.propainter_save_frames is not None:
            propainter_options["save_frames"] = bool(args.propainter_save_frames)
        if not propainter_options:
            propainter_options = None
    out_dir = run_inpaint_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        masks_dir=Path(args.masks_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
        diffueraser_options=diffueraser_options,
        propainter_options=propainter_options,
    )
    print(f"inpaint_frames_dir={out_dir}")


if __name__ == "__main__":
    main()
