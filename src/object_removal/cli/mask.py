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
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run_mask_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
        max_frames=args.max_frames,
    )
    print(f"mask_dir={out_dir}")


if __name__ == "__main__":
    main()

