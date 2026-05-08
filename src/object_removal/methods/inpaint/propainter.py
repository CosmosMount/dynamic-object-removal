from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Params:
    resize_ratio: float = 0.75
    subvideo_length: int = 40
    neighbor_length: int = 8
    raft_iter: int = 12
    fp16: bool = True
    save_frames: bool = True


def run(*, repo_root: Path, frames_dir: Path, masks_dir: Path, out_root: Path, params: Params) -> dict:
    """Run ProPainter via its `inference_propainter.py`.

    You must run this inside the `propainter` conda env.
    """
    import os
    import subprocess

    pp_root = repo_root / "modules" / "ProPainter"
    script = pp_root / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing ProPainter inference script: {script}")

    # We run ProPainter with cwd=repo_root/ckpts/propainter to centralize downloaded weights.
    # Therefore, all user-provided I/O paths must be absolute to avoid being interpreted under that cwd.
    frames_dir = frames_dir.resolve()
    masks_dir = masks_dir.resolve()
    out_root = out_root.resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    # ProPainter downloads weights into `model_dir='weights'` (relative path).
    # We pin that relative directory under repo_root/ckpts/propainter so models are centralized.
    ckpt_cwd = repo_root / "ckpts" / "propainter"
    ckpt_cwd.mkdir(parents=True, exist_ok=True)

    argv = [
        "inference_propainter.py",
        "--video",
        str(frames_dir),
        "--mask",
        str(masks_dir),
        "--output",
        str(out_root),
        "--resize_ratio",
        str(params.resize_ratio),
        "--subvideo_length",
        str(params.subvideo_length),
        "--neighbor_length",
        str(params.neighbor_length),
        "--raft_iter",
        str(params.raft_iter),
    ]
    if params.fp16:
        argv.append("--fp16")
    if params.save_frames:
        argv.append("--save_frames")

    cmd = [sys.executable, str(script), *argv[1:]]
    subprocess.run(cmd, cwd=str(ckpt_cwd), check=True, env={**os.environ, "PYTHONPATH": str(pp_root)})

    # ProPainter writes <out_root>/<video_name>/frames and inpaint_out.mp4
    video_name = frames_dir.name
    frames_out = out_root / video_name / "frames"
    video_out = out_root / video_name / "inpaint_out.mp4"
    return {"frames_out": str(frames_out), "video_out": str(video_out)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ProPainter inpaint method")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    run(repo_root=Path(args.repo_root), frames_dir=Path(args.frames_dir), masks_dir=Path(args.masks_dir), out_root=Path(args.out_root), params=Params())


if __name__ == "__main__":
    main()

