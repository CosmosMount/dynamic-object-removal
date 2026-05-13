from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.track_stage import run_track_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track stage CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--in_masks_dir", required=True)
    p.add_argument("--method", required=True, choices=["identity", "sam3", "optflow", "xmem"])
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--repo_root", default=".", help="Repo root (required layout for sam3 ckpt paths).")
    p.add_argument("--sam3-checkpoint", default="ckpts/sam3/sam3.pt", help="Used when --method sam3")
    p.add_argument(
        "--sam3-two-stage-anchor-idx",
        default="auto",
        help="Used when --method sam3: overlap-anchor (-1=off, auto=max overlap, int=fixed anchor).",
    )
    p.add_argument(
        "--sam3-two-stage-auto-samples",
        type=int,
        default=7,
        help="Used when --method sam3 (deprecated; kept for CLI compatibility).",
    )
    p.add_argument(
        "--sam3-two-stage-auto-max-fg-frac",
        type=float,
        default=0.92,
        help="Used when --method sam3 (deprecated).",
    )
    p.add_argument(
        "--sam3-two-stage-auto-min-fg-frac",
        type=float,
        default=0.00008,
        help="Used when --method sam3 (deprecated).",
    )
    p.add_argument(
        "--sam3-two-stage-auto-min-fg-pixels",
        type=int,
        default=64,
        help="Used when --method sam3 (deprecated).",
    )
    p.add_argument("--sam3-score-thresh", type=float, default=0.0, help="Used when --method sam3")
    p.add_argument(
        "--xmem-bidirectional",
        action="store_true",
        help="Used when --method xmem: enable overlap-anchor (same gate as non-off two_stage_anchor_idx).",
    )
    p.add_argument(
        "--xmem-bidirectional-merge",
        type=str,
        default="union",
        choices=["union", "intersection", "vote"],
        help="Used when --method xmem (deprecated; ignored).",
    )
    p.add_argument(
        "--xmem-two-stage-anchor-idx",
        type=str,
        default="-1",
        help="Used when --method xmem: -1 off unless --xmem-bidirectional; auto=max overlap; int=fixed anchor.",
    )
    p.add_argument(
        "--xmem-two-stage-auto-samples",
        type=int,
        default=7,
        help="Used when --method xmem (deprecated).",
    )
    p.add_argument(
        "--xmem-two-stage-auto-max-fg-frac",
        type=float,
        default=0.92,
        help="Used when --method xmem (deprecated).",
    )
    p.add_argument(
        "--xmem-two-stage-auto-min-fg-frac",
        type=float,
        default=0.00008,
        help="Used when --method xmem (deprecated).",
    )
    p.add_argument(
        "--xmem-two-stage-auto-min-fg-pixels",
        type=int,
        default=64,
        help="Used when --method xmem (deprecated).",
    )
    p.add_argument(
        "--xmem-two-stage-min-overlap-ratio",
        type=float,
        default=0.25,
        help="Per-object min overlap ratio for two-stage anchor seed validation.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    sam3_options = None
    if args.method == "sam3":
        sam3_options = {
            "checkpoint": args.sam3_checkpoint,
            "two_stage_anchor_idx": args.sam3_two_stage_anchor_idx,
            "two_stage_auto_samples": args.sam3_two_stage_auto_samples,
            "two_stage_auto_max_fg_frac": args.sam3_two_stage_auto_max_fg_frac,
            "two_stage_auto_min_fg_frac": args.sam3_two_stage_auto_min_fg_frac,
            "two_stage_auto_min_fg_pixels": args.sam3_two_stage_auto_min_fg_pixels,
            "score_thresh": args.sam3_score_thresh,
        }
    xmem_options = None
    if args.method == "xmem":
        xmem_options = {
            "bidirectional": bool(args.xmem_bidirectional),
            "bidirectional_merge": str(args.xmem_bidirectional_merge),
            "two_stage_anchor_idx": str(args.xmem_two_stage_anchor_idx),
            "two_stage_auto_samples": int(args.xmem_two_stage_auto_samples),
            "two_stage_auto_max_fg_frac": float(args.xmem_two_stage_auto_max_fg_frac),
            "two_stage_auto_min_fg_frac": float(args.xmem_two_stage_auto_min_fg_frac),
            "two_stage_auto_min_fg_pixels": int(args.xmem_two_stage_auto_min_fg_pixels),
            "two_stage_min_overlap_ratio": float(args.xmem_two_stage_min_overlap_ratio),
        }
    out_dir = run_track_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        in_masks_dir=Path(args.in_masks_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
        sam3_options=sam3_options,
        xmem_options=xmem_options,
        repo_root=Path(args.repo_root).resolve(),
    )
    print(f"track_masks_dir={out_dir}")


if __name__ == "__main__":
    main()
