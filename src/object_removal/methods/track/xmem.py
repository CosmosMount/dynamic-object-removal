from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Params:
    device: str = "cuda:0"
    no_download: bool = False
    # Two-stage merge + bidirectional fusion.
    bidirectional: bool = False
    bidirectional_merge: str = "union"  # union|intersection|vote (bidirectional fusion mode)
    bidirectional_vote_threshold: float = 0.3  # min overlap with intersection for union components
    two_stage_anchor_idx: str = "-1"  # "-1"|"auto"|non-negative int
    two_stage_auto_samples: int = 7  # Deprecated (max-overlap anchor selection ignores this).
    two_stage_auto_max_fg_frac: float = 0.92
    two_stage_auto_min_fg_frac: float = 0.00008
    two_stage_auto_min_fg_pixels: int = 64
    two_stage_min_overlap_ratio: float = 0.25


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
    in_masks_dir: Path,
    out_binary_dir: Path,
    params: Params,
    options: Optional[Dict[str, Any]] = None,
) -> dict:
    """XMem tracker producing binary masks from mask-pipeline init."""
    import sys

    from object_removal.vendor import xmem_masks as mod

    params = params_with_overrides(params, options)
    xmem_root = repo_root / "modules" / "xmem_tracker"
    out_raw = out_binary_dir.parent / "masks_indexed_raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_binary_dir.mkdir(parents=True, exist_ok=True)

    # VGGT4D may only write masks for frames where motion is detected (e.g. 00006.png).
    init_files = sorted(in_masks_dir.glob("*.png"))
    if not init_files:
        raise FileNotFoundError(f"XMem requires at least one init mask in {in_masks_dir}")
    init_mask = init_files[0]

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
        "--init_mask",
        str(init_mask),
    ]
    if params.no_download:
        argv.append("--no_download")
    if params.bidirectional:
        argv.append("--xmem-bidirectional")
    argv += ["--xmem-bidirectional-merge", str(params.bidirectional_merge)]
    argv += ["--xmem-bidirectional-vote-threshold", str(float(params.bidirectional_vote_threshold))]
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
        "--xmem-two-stage-min-overlap-ratio",
        str(float(params.two_stage_min_overlap_ratio)),
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
    ap.add_argument("--in_masks_dir", required=True)
    ap.add_argument("--out_binary_dir", required=True)
    args = ap.parse_args()
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        in_masks_dir=Path(args.in_masks_dir),
        out_binary_dir=Path(args.out_binary_dir),
        params=Params(),
    )


if __name__ == "__main__":
    main()
