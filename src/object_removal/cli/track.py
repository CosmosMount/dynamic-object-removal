from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.track_stage import run_track_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track stage CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--in_masks_dir", required=True)
    p.add_argument("--method", required=True, choices=["identity", "sam2", "sam3", "optflow", "trackanything"])
    p.add_argument("--overwrite", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run_track_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        in_masks_dir=Path(args.in_masks_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
    )
    print(f"track_masks_dir={out_dir}")


if __name__ == "__main__":
    main()

