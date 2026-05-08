from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from object_removal.utils.video import reencode_mp4_h264_inplace


@dataclass(frozen=True)
class Params:
    video_length: int = 10
    mask_dilation_iter: int = 8
    max_img_size: int = 960
    ref_stride: int = 10
    neighbor_length: int = 10
    subvideo_length: int = 50
    base_model_path: Path | None = None
    vae_path: Path | None = None
    diffueraser_path: Path | None = None
    propainter_model_dir: Path | None = None


def run(
    *,
    repo_root: Path,
    input_video: Path,
    input_mask: Path,
    out_dir: Path,
    params: Params,
) -> dict:
    """Run DiffuEraser by importing its official entrypoint in-process.

    Assumes you run inside an env with diffueraser dependencies installed.
    """
    diff_root = repo_root / "modules" / "DiffuEraser"
    if not diff_root.is_dir():
        raise FileNotFoundError(f"Missing DiffuEraser module dir: {diff_root}")
    if str(diff_root) not in sys.path:
        sys.path.insert(0, str(diff_root))

    script = diff_root / "run_diffueraser.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing DiffuEraser entrypoint: {script}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("diffueraser_entry", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "run_diffueraser.py",
        "--input_video",
        str(input_video),
        "--input_mask",
        str(input_mask),
        "--video_length",
        str(params.video_length),
        "--mask_dilation_iter",
        str(params.mask_dilation_iter),
        "--max_img_size",
        str(params.max_img_size),
        "--save_path",
        str(out_dir),
        "--ref_stride",
        str(params.ref_stride),
        "--neighbor_length",
        str(params.neighbor_length),
        "--subvideo_length",
        str(params.subvideo_length),
    ]

    if params.base_model_path:
        argv += ["--base_model_path", str(params.base_model_path)]
    if params.vae_path:
        argv += ["--vae_path", str(params.vae_path)]
    if params.diffueraser_path:
        argv += ["--diffueraser_path", str(params.diffueraser_path)]
    if params.propainter_model_dir:
        argv += ["--propainter_model_dir", str(params.propainter_model_dir)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = old_argv

    out_video = out_dir / "diffueraser_result.mp4"
    if not out_video.is_file():
        raise FileNotFoundError(f"DiffuEraser output video not found: {out_video}")
    # OpenCV mp4v is poorly supported in VS Code / Chromium; re-encode when ffmpeg is available.
    priori = out_dir / "priori.mp4"
    reencode_mp4_h264_inplace(out_video)
    if priori.is_file():
        reencode_mp4_h264_inplace(priori)
    return {"output_video": str(out_video)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="DiffuEraser inpaint method")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--input_video", required=True)
    ap.add_argument("--input_mask", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run(
        repo_root=Path(args.repo_root),
        input_video=Path(args.input_video),
        input_mask=Path(args.input_mask),
        out_dir=Path(args.out_dir),
        params=Params(),
    )


if __name__ == "__main__":
    main()

