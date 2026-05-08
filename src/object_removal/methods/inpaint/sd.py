from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Params:
    keyframe_stride: int = 10
    min_mask_area: int = 128
    sd_model: str = "runwayml/stable-diffusion-inpainting"
    controlnet_canny: str = "lllyasviel/sd-controlnet-canny"
    controlnet_depth: str = "lllyasviel/sd-controlnet-depth"
    prompt: str = "clean natural background, remove target object, photorealistic"
    negative_prompt: str = "artifacts, blurry, distorted, watermark, text, logo"
    seed: int = 1234
    steps: int = 28
    guidance_scale: float = 7.0
    strength: float = 0.95
    controlnet_canny_scale: float = 0.8
    controlnet_depth_scale: float = 0.7
    canny_low: int = 100
    canny_high: int = 200
    device: str = "cuda"


def run(*, repo_root: Path, frames_dir: Path, masks_dir: Path, out_frames_dir: Path, params: Params) -> dict:
    """SD keyframe inpainting (sparse keyframes, merged into full frames dir).

    Notes:
    - This only generates keyframes every N frames and writes merged frames.
    - It does NOT do temporal propagation between keyframes (kept consistent with legacy pipeline).
    - Must be run in an env with diffusers + controlnet_aux + torch + opencv.
    """
    import sys

    from object_removal.vendor import sd_keyframe_inpaint as mod

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = out_frames_dir.parent / "sd_keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "sd_keyframe_inpaint.py",
        "--frame_dir",
        str(frames_dir),
        "--mask_dir",
        str(masks_dir),
        "--output_keyframe_dir",
        str(keyframes_dir),
        "--output_merged_dir",
        str(out_frames_dir),
        "--keyframe_stride",
        str(params.keyframe_stride),
        "--min_mask_area",
        str(params.min_mask_area),
        "--sd_model",
        params.sd_model,
        "--controlnet_canny",
        params.controlnet_canny,
        "--controlnet_depth",
        params.controlnet_depth,
        "--prompt",
        params.prompt,
        "--negative_prompt",
        params.negative_prompt,
        "--seed",
        str(params.seed),
        "--steps",
        str(params.steps),
        "--guidance_scale",
        str(params.guidance_scale),
        "--strength",
        str(params.strength),
        "--controlnet_canny_scale",
        str(params.controlnet_canny_scale),
        "--controlnet_depth_scale",
        str(params.controlnet_depth_scale),
        "--canny_low",
        str(params.canny_low),
        "--canny_high",
        str(params.canny_high),
        "--device",
        params.device,
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv

    return {
        "frames_out": str(out_frames_dir),
        "keyframes_dir": str(keyframes_dir),
        "keyframe_stride": params.keyframe_stride,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="SD keyframe inpaint (merged frames)")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_frames_dir", required=True)
    args = ap.parse_args()
    run(
        repo_root=Path(args.repo_root),
        frames_dir=Path(args.frames_dir),
        masks_dir=Path(args.masks_dir),
        out_frames_dir=Path(args.out_frames_dir),
        params=Params(),
    )


if __name__ == "__main__":
    main()

