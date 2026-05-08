from __future__ import annotations

"""
Vendored from legacy: pipelines/vggt4dsam3sd/sd_keyframe_inpaint.py

No functional changes intended.
"""

#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def _require_pkg(name: str, install_hint: str):
    try:
        return __import__(name)
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(f"Missing dependency '{name}'. Install with: {install_hint}") from exc


@dataclass
class KeyframeResult:
    index: int
    frame_name: str
    mask_area: int
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SD1.5 + ControlNet keyframe inpainting.")
    parser.add_argument("--frame_dir", required=True, help="Input frame directory.")
    parser.add_argument("--mask_dir", required=True, help="Binary mask directory (white=inpaint).")
    parser.add_argument("--output_keyframe_dir", required=True, help="Output directory for generated keyframes.")
    parser.add_argument("--output_merged_dir", required=True, help="Output full frame directory with generated keyframes merged in.")
    parser.add_argument("--keyframe_stride", type=int, default=10, help="Sample one keyframe every N frames.")
    parser.add_argument("--min_mask_area", type=int, default=128, help="Skip keyframes with tiny masks.")

    parser.add_argument(
        "--sd_model",
        default="runwayml/stable-diffusion-inpainting",
        help="HF model id or local path for SD inpainting model.",
    )
    parser.add_argument(
        "--controlnet_canny",
        default="lllyasviel/sd-controlnet-canny",
        help="HF model id or local path for canny ControlNet.",
    )
    parser.add_argument(
        "--controlnet_depth",
        default="lllyasviel/sd-controlnet-depth",
        help="HF model id or local path for depth ControlNet.",
    )
    parser.add_argument("--prompt", default="clean natural background, remove target object, photorealistic")
    parser.add_argument("--negative_prompt", default="artifacts, blurry, distorted, watermark, text, logo")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--controlnet_canny_scale", type=float, default=0.8)
    parser.add_argument("--controlnet_depth_scale", type=float, default=0.7)
    parser.add_argument("--canny_low", type=int, default=100)
    parser.add_argument("--canny_high", type=int, default=200)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def sorted_frame_names(frame_dir: str) -> List[str]:
    names = [n for n in os.listdir(frame_dir) if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}]
    names.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    return names


def load_mask(mask_path: str, size: Tuple[int, int]) -> np.ndarray:
    if not os.path.isfile(mask_path):
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if (m.shape[1], m.shape[0]) != size:
        m = np.array(Image.fromarray(m, mode="L").resize(size, resample=Image.NEAREST), dtype=np.uint8)
    return (m > 0).astype(np.uint8) * 255


def make_canny_image(image_rgb: np.ndarray, low: int, high: int) -> Image.Image:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_3c = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_3c)


def ensure_merged_dir(frame_dir: str, output_merged_dir: str, frame_names: List[str]) -> None:
    os.makedirs(output_merged_dir, exist_ok=True)
    for name in frame_names:
        src = os.path.join(frame_dir, name)
        dst = os.path.join(output_merged_dir, name)
        if os.path.isfile(dst):
            continue
        Image.open(src).save(dst)


def get_8_aligned_size(size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = size
    aligned_w = max(64, (width // 8) * 8)
    aligned_h = max(64, (height // 8) * 8)
    return aligned_w, aligned_h


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available; fallback to CPU")
        args.device = "cpu"

    frame_names = sorted_frame_names(args.frame_dir)
    if not frame_names:
        raise RuntimeError(f"No frames found in: {args.frame_dir}")

    os.makedirs(args.output_keyframe_dir, exist_ok=True)
    ensure_merged_dir(args.frame_dir, args.output_merged_dir, frame_names)

    _require_pkg("diffusers", "pip install diffusers transformers accelerate")
    _require_pkg("controlnet_aux", "pip install controlnet_aux")

    from controlnet_aux import MidasDetector
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    controlnets = [
        ControlNetModel.from_pretrained(args.controlnet_canny, torch_dtype=dtype),
        ControlNetModel.from_pretrained(args.controlnet_depth, torch_dtype=dtype),
    ]

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.sd_model,
        controlnet=controlnets,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    if args.device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    depth_estimator = MidasDetector.from_pretrained("lllyasviel/Annotators")

    keyframe_results: List[KeyframeResult] = []
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for idx, frame_name in enumerate(frame_names):
        if idx % args.keyframe_stride != 0:
            continue

        frame_path = os.path.join(args.frame_dir, frame_name)
        frame_img = Image.open(frame_path).convert("RGB")
        frame_np = np.array(frame_img)

        mask_name = os.path.splitext(frame_name)[0] + ".png"
        mask_path = os.path.join(args.mask_dir, mask_name)
        mask_np = load_mask(mask_path, frame_img.size)
        mask_area = int((mask_np > 0).sum())
        if mask_area < args.min_mask_area:
            continue

        canny_img = make_canny_image(frame_np, args.canny_low, args.canny_high)
        depth_img = depth_estimator(frame_img)
        if not isinstance(depth_img, Image.Image):
            depth_img = Image.fromarray(np.array(depth_img))
        depth_img = depth_img.convert("RGB")

        target_w, target_h = get_8_aligned_size(frame_img.size)
        frame_in = frame_img.resize((target_w, target_h), resample=Image.BILINEAR)
        mask_in = Image.fromarray(mask_np, mode="L").resize((target_w, target_h), resample=Image.NEAREST)
        canny_in = canny_img.resize((target_w, target_h), resample=Image.BILINEAR)
        depth_in = depth_img.resize((target_w, target_h), resample=Image.BILINEAR)

        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=frame_in,
            mask_image=mask_in,
            control_image=[canny_in, depth_in],
            controlnet_conditioning_scale=[args.controlnet_canny_scale, args.controlnet_depth_scale],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            generator=generator,
        ).images[0]
        out = out.resize(frame_img.size, resample=Image.BILINEAR)

        keyframe_out_path = os.path.join(args.output_keyframe_dir, frame_name)
        merged_out_path = os.path.join(args.output_merged_dir, frame_name)
        out.save(keyframe_out_path)
        out.save(merged_out_path)
        keyframe_results.append(
            KeyframeResult(index=idx, frame_name=frame_name, mask_area=mask_area, output_path=keyframe_out_path)
        )
        print(f"[sd] keyframe {idx:05d} -> {keyframe_out_path} (mask_area={mask_area})")

    summary_path = os.path.join(args.output_keyframe_dir, "keyframes.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for item in keyframe_results:
            f.write(f"{item.index}\t{item.frame_name}\t{item.mask_area}\t{item.output_path}\n")

    print(f"Done. keyframes={len(keyframe_results)}")
    print(f"- keyframe dir: {args.output_keyframe_dir}")
    print(f"- merged dir:   {args.output_merged_dir}")
    print(f"- summary:      {summary_path}")


if __name__ == "__main__":
    main()

