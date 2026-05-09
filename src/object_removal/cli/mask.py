from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.mask_stage import run_mask_stage


def _vggt_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mask stage CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--method", required=True, choices=["baseline_yolo_motion", "vggt4d", "yolo_first", "vggt_framewise"])
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--repo_root", default=".", help="Repo root (for ckpts/modules resolution).")
    p.add_argument(
        "--vggt4d-init-frame",
        type=int,
        default=None,
        help="When --method vggt4d: full-sequence frame index (nearest VGGT slot); init written as NNNNN.png.",
    )
    p.add_argument(
        "--vggt4d-max-frames",
        type=int,
        default=None,
        help="When --method vggt4d: first N frames fed to VGGT (object-removal style). vggt_framewise: uniform subsample cap.",
    )
    p.add_argument(
        "--vggt4d-dyn-threshold-scale",
        type=float,
        default=None,
        help="VGGT4D demo_vggt4d dyn threshold scale (<1 → more motion). Same as object-removal --dyn_threshold_scale.",
    )
    p.add_argument(
        "--vggt4d-threshold",
        type=int,
        default=None,
        help="Binarize dynamic_mask PNG with arr > threshold (default in Params if omitted).",
    )
    p.add_argument(
        "--vggt4d-merge-all",
        type=_vggt_bool,
        default=None,
        help="vggt4d init: merge_all mode (true/false).",
    )
    p.add_argument(
        "--vggt4d-split-cc",
        type=_vggt_bool,
        default=None,
        help="Split foreground into instance ids by CC (true/false).",
    )
    p.add_argument("--vggt4d-cc-min-area", type=int, default=None, help="CC min area (pixels).")
    p.add_argument("--vggt4d-cc-max-objects", type=int, default=None, help="Max CC instances kept.")
    p.add_argument("--vggt4d-cc-close-kernel", type=int, default=None, help="Morph close kernel (0=off).")
    p.add_argument(
        "--vggt4d-tail-policy",
        type=str,
        default=None,
        choices=["hold_last", "zeros"],
        help="vggt_framewise only: how to fill frames without a VGGT slot.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    vggt4d_options = None
    if args.method in ("vggt4d", "vggt_framewise"):
        vggt4d_options = {}
        if args.method == "vggt4d" and args.vggt4d_init_frame is not None:
            vggt4d_options["init_frame"] = int(args.vggt4d_init_frame)
        if args.vggt4d_max_frames is not None:
            vggt4d_options["max_frames_for_vggt"] = int(args.vggt4d_max_frames)
        if args.vggt4d_dyn_threshold_scale is not None:
            vggt4d_options["dyn_threshold_scale"] = float(args.vggt4d_dyn_threshold_scale)
        if args.vggt4d_threshold is not None:
            vggt4d_options["threshold"] = int(args.vggt4d_threshold)
        if args.vggt4d_merge_all is not None:
            vggt4d_options["merge_all"] = bool(args.vggt4d_merge_all)
        if args.vggt4d_split_cc is not None:
            vggt4d_options["split_cc"] = bool(args.vggt4d_split_cc)
        if args.vggt4d_cc_min_area is not None:
            vggt4d_options["cc_min_area"] = int(args.vggt4d_cc_min_area)
        if args.vggt4d_cc_max_objects is not None:
            vggt4d_options["cc_max_objects"] = int(args.vggt4d_cc_max_objects)
        if args.vggt4d_cc_close_kernel is not None:
            vggt4d_options["cc_close_kernel"] = int(args.vggt4d_cc_close_kernel)
        if args.vggt4d_tail_policy is not None:
            vggt4d_options["tail_policy"] = str(args.vggt4d_tail_policy)
        if not vggt4d_options:
            vggt4d_options = None
    out_dir = run_mask_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
        max_frames=args.max_frames,
        vggt4d_options=vggt4d_options,
        repo_root=Path(args.repo_root).resolve(),
    )
    print(f"mask_dir={out_dir}")


if __name__ == "__main__":
    main()
