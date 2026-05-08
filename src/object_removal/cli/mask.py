from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.mask_stage import run_mask_stage


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
        help="When --method vggt4d: first N frames fed to VGGT (object-removal style). vggt_framewise: still uniform subsample.",
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
