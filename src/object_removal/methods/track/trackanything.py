from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Params:
    device: str = "cuda:0"
    sam_model_type: str = "vit_h"
    init_source: str = "yolo"  # yolo|sam_auto|mask
    init_mask: Path | None = None
    yolo_model: str | None = None
    yolo_conf: float = 0.25
    classes: str = "all"
    max_objects: int = 4
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.80
    sam_points_per_side: int = 32
    no_download: bool = False


def run(*, repo_root: Path, frames_dir: Path, out_binary_dir: Path, params: Params) -> dict:
    """Track-Anything tracker producing binary masks.

    Runs in the `trackanything` env. Uses vendored headless implementation.
    """
    import sys

    from object_removal.vendor import trackanything_masks as mod

    track_dir = repo_root / "modules" / "Track-Anything"
    out_raw = out_binary_dir.parent / "masks_indexed_raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_binary_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "trackanything_masks.py",
        "--repo_root",
        str(repo_root),
        "--frame_dir",
        str(frames_dir),
        "--raw_mask_dir",
        str(out_raw),
        "--binary_mask_dir",
        str(out_binary_dir),
        "--trackanything_dir",
        str(track_dir),
        "--device",
        params.device,
        "--sam_model_type",
        params.sam_model_type,
        "--init_source",
        params.init_source,
        "--yolo_conf",
        str(params.yolo_conf),
        "--classes",
        params.classes,
        "--max_objects",
        str(params.max_objects),
        "--min_area_ratio",
        str(params.min_area_ratio),
        "--max_area_ratio",
        str(params.max_area_ratio),
        "--sam_points_per_side",
        str(params.sam_points_per_side),
    ]
    if params.no_download:
        argv.append("--no_download")
    if params.init_mask is not None:
        argv += ["--init_mask", str(params.init_mask)]
    if params.yolo_model is not None:
        argv += ["--yolo_model", params.yolo_model]

    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv

    return {"out_binary_dir": str(out_binary_dir)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Track-Anything tracker (binary masks)")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_binary_dir", required=True)
    args = ap.parse_args()
    run(repo_root=Path(args.repo_root), frames_dir=Path(args.frames_dir), out_binary_dir=Path(args.out_binary_dir), params=Params())


if __name__ == "__main__":
    main()

