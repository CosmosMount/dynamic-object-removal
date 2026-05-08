from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Params:
    checkpoint: Path
    two_stage_anchor_idx: str = "auto"  # "-1" | int | "auto"
    two_stage_auto_samples: int = 7
    score_thresh: float = 0.0


def run(
    *,
    repo_root: Path,
    frames_dir: Path,
    init_mask_dir: Path,
    out_raw_indexed_dir: Path,
    out_binary_dir: Path,
    params: Params,
) -> dict:
    """Run SAM3 VOS based on the existing inference implementation.

    This keeps the core logic identical by reusing the vendored implementation in
    `src/object_removal/vendor/sam3_vos.py`. You are expected to run this inside a python env
    where SAM3 + torch are installed and usable.
    """
    from object_removal.vendor import sam3_vos as mod

    # Prepare minimal folder layout expected by legacy script:
    # - base_video_dir contains <video_name>/00000.jpg ...
    # - input_mask_dir contains <video_name>/00000.png (indexed init mask)
    video_name = frames_dir.name
    base_video_dir = frames_dir.parent
    if (base_video_dir / video_name) != frames_dir:
        raise ValueError("frames_dir must be a leaf directory named as the video/sequence (e.g. .../bmx-trees)")

    video_list = out_raw_indexed_dir.parent / "video_list.txt"
    video_list.parent.mkdir(parents=True, exist_ok=True)
    video_list.write_text(video_name + "\n", encoding="utf-8")

    argv = [
        "sam3_vos.py",
        "--repo_root",
        str(repo_root),
        "--sam3_checkpoint",
        str(params.checkpoint),
        "--base_video_dir",
        str(base_video_dir),
        "--input_mask_dir",
        str(init_mask_dir.parent),
        "--video_list_file",
        str(video_list),
        "--output_mask_dir",
        str(out_raw_indexed_dir.parent),
        "--score_thresh",
        str(params.score_thresh),
        "--two_stage_anchor_idx",
        str(params.two_stage_anchor_idx),
        "--two_stage_auto_samples",
        str(params.two_stage_auto_samples),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv

    # Convert indexed masks to canonical binary
    from object_removal.io.masks import list_mask_files, read_mask_u8, to_binary_255, write_mask_u8

    raw_seq_dir = out_raw_indexed_dir.parent / video_name
    if not raw_seq_dir.is_dir():
        raise FileNotFoundError(f"SAM3 output dir missing: {raw_seq_dir}")

    out_binary_dir.mkdir(parents=True, exist_ok=True)
    files = list_mask_files(raw_seq_dir)
    for p in files:
        write_mask_u8(out_binary_dir / p.name, to_binary_255(read_mask_u8(p)))

    return {"video_name": video_name, "num_frames": len(files)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="SAM3 track method")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--init_mask_dir", required=True)
    ap.add_argument("--out_raw_indexed_dir", required=True)
    ap.add_argument("--out_binary_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--two_stage_anchor_idx", default="auto")
    ap.add_argument("--two_stage_auto_samples", type=int, default=7)
    args = ap.parse_args()

    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        init_mask_dir=Path(args.init_mask_dir),
        out_raw_indexed_dir=Path(args.out_raw_indexed_dir),
        out_binary_dir=Path(args.out_binary_dir),
        params=Params(
            checkpoint=Path(args.checkpoint),
            two_stage_anchor_idx=str(args.two_stage_anchor_idx),
            two_stage_auto_samples=int(args.two_stage_auto_samples),
        ),
    )


if __name__ == "__main__":
    main()

