from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.stages.inpaint_stage import run_inpaint_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inpaint stage CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--masks_dir", required=True)
    p.add_argument("--method", required=True, choices=["baseline_handcrafted", "propainter", "diffueraser"])
    p.add_argument("--overwrite", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run_inpaint_stage(
        run_dir=Path(args.run_dir),
        frames_dir=Path(args.frames_dir),
        masks_dir=Path(args.masks_dir),
        method=args.method,
        overwrite=bool(args.overwrite),
    )
    print(f"inpaint_frames_dir={out_dir}")


if __name__ == "__main__":
    main()

