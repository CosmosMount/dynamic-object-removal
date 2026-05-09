"""CLI: export mask overlay video from frames + per-frame PNG masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.io.mask_vis import export_mask_overlay_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encode a mask overlay video (same idea as object-removal postprocess_sam3 export_mask_video).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames_dir", required=True, type=Path, help="Directory of JPEG/PNG video frames")
    p.add_argument("--masks_dir", required=True, type=Path, help="Directory of PNG masks (same stem as frames)")
    p.add_argument("--out", required=True, type=Path, help="Output .mp4 path")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.5, help="Blend strength on masked pixels")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ok = export_mask_overlay_video(
        args.frames_dir.resolve(),
        args.masks_dir.resolve(),
        args.out.resolve(),
        fps=float(args.fps),
        alpha=float(args.alpha),
    )
    if not ok:
        raise SystemExit("mask_vis: no frames written (empty frames_dir or unreadable images)")


if __name__ == "__main__":
    main()
