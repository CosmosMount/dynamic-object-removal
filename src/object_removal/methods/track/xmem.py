from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional


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
    # Overlap-anchor refinement (see vendor `xmem_masks.py`).
    bidirectional: bool = False
    bidirectional_merge: str = "union"  # Deprecated; kept for YAML/CLI compatibility.
    two_stage_anchor_idx: str = "-1"  # "-1"|"auto"|non-negative int
    two_stage_auto_samples: int = 7  # Deprecated for overlap-anchor auto (unused).
    two_stage_auto_max_fg_frac: float = 0.92
    two_stage_auto_min_fg_frac: float = 0.00008
    two_stage_auto_min_fg_pixels: int = 64


def params_with_overrides(base: Params, overrides: Optional[Dict[str, Any]]) -> Params:
    if not overrides:
        return base
    names = {f.name for f in fields(Params)}
    kw = {k: v for k, v in overrides.items() if k in names}
    return replace(base, **kw)


def run(
    *,
    repo_root: Path,
    frames_dir: Path,
    out_binary_dir: Path,
    params: Params,
    options: Optional[Dict[str, Any]] = None,
) -> dict:
    """XMem tracker producing binary masks."""
    import sys

    from object_removal.vendor import xmem_masks as mod

    params = params_with_overrides(params, options)
    xmem_root = repo_root / "modules" / "xmem_tracker"
    out_raw = out_binary_dir.parent / "masks_indexed_raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_binary_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "xmem_masks.py",
        "--repo_root",
        str(repo_root),
        "--frame_dir",
        str(frames_dir),
        "--raw_mask_dir",
        str(out_raw),
        "--binary_mask_dir",
        str(out_binary_dir),
        "--xmem_root",
        str(xmem_root),
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
        argv += ["--yolo_model", str(params.yolo_model)]
    if params.bidirectional:
        argv.append("--xmem-bidirectional")
    argv += ["--xmem-bidirectional-merge", str(params.bidirectional_merge)]
    argv += [
        "--xmem-two-stage-anchor-idx",
        str(params.two_stage_anchor_idx),
        "--xmem-two-stage-auto-samples",
        str(int(params.two_stage_auto_samples)),
        "--xmem-two-stage-auto-max-fg-frac",
        str(float(params.two_stage_auto_max_fg_frac)),
        "--xmem-two-stage-auto-min-fg-frac",
        str(float(params.two_stage_auto_min_fg_frac)),
        "--xmem-two-stage-auto-min-fg-pixels",
        str(int(params.two_stage_auto_min_fg_pixels)),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv

    return {"out_binary_dir": str(out_binary_dir)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="XMem tracker (binary masks)")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_binary_dir", required=True)
    args = ap.parse_args()
    run(repo_root=Path(args.repo_root), frames_dir=Path(args.frames_dir), out_binary_dir=Path(args.out_binary_dir), params=Params())


if __name__ == "__main__":
    main()
