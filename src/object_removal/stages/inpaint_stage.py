from __future__ import annotations

import math
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.io.masks import write_eroded_binary_mask_pngs
from object_removal.io.manifest import write_method_manifest
from object_removal.methods.inpaint import handcrafted, propainter, diffueraser
from object_removal.methods.inpaint.handcrafted import BaselineInpaintParams
from object_removal.utils.video import frames_to_mp4, masks_to_mp4, mp4_to_frames, reencode_mp4_h264_inplace


def run_inpaint_stage(
    *,
    run_dir: Path,
    frames_dir: Path,
    masks_dir: Path,
    method: str,
    overwrite: bool = False,
    diffueraser_options: Optional[Dict[str, Any]] = None,
    propainter_options: Optional[Dict[str, Any]] = None,
) -> Path:
    layout = RunLayout(run_dir)
    ensure_layout_dirs(layout)

    out_dir = layout.inpaint_frames_dir
    # Stale PNGs from a longer previous run break frame–GT alignment and metrics; clear before re-inpaint.
    if overwrite and out_dir.is_dir():
        for p in list(out_dir.glob("*.png")):
            p.unlink(missing_ok=True)
    existing = list(out_dir.glob("*.png"))
    if existing and not overwrite:
        return out_dir

    if method == "baseline_handcrafted":
        params = BaselineInpaintParams()
        meta = handcrafted.run(frames_dir=frames_dir, masks_dir=masks_dir, out_frames_dir=out_dir, params=params)
        write_method_manifest(layout.inpaint_dir, stage="inpaint", method=method, params=meta)
        return out_dir

    if method == "propainter":
        tmp_root = layout.inpaint_dir / "propainter"
        base_pp = propainter.Params()
        po = propainter_options or {}
        allowed_pp = {f.name for f in fields(propainter.Params)}
        params_pp = replace(base_pp, **{k: v for k, v in po.items() if k in allowed_pp})
        meta = propainter.run(
            repo_root=Path.cwd(),
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            out_root=tmp_root,
            params=params_pp,
        )
        # normalize to canonical run_dir/inpaint/frames
        pp_frames = Path(meta.get("frames_out", ""))
        if pp_frames.is_dir():
            # copy as %05d.png to out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            srcs = sorted([p for p in pp_frames.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            for i, p in enumerate(srcs):
                shutil.copy2(p, out_dir / f"{i:05d}.png")
        write_method_manifest(layout.inpaint_dir, stage="inpaint", method=method, params=meta)
        return out_dir

    if method == "diffueraser":
        # prepare inputs for diffueraser
        tmp = layout.inpaint_dir / "diffueraser"
        if overwrite and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        input_video = tmp / "input_video.mp4"
        input_mask = tmp / "input_mask.mp4"
        _de_def = {
            "mask_dilation_iter": 0,
            "mask_hole_shrink_iters": 2,
            "max_img_size": 960,
            "ref_stride": 8,
            "neighbor_length": 12,
            "subvideo_length": 50,
        }
        raw_do = dict(diffueraser_options or {})
        do = {**_de_def, **{k: v for k, v in raw_do.items() if v is not None}}
        hole_shrink = int(do.pop("mask_hole_shrink_iters", 0) or 0)
        masks_in = masks_dir
        if hole_shrink > 0:
            tight = tmp / "masks_hole_shrink"
            write_eroded_binary_mask_pngs(src_dir=masks_dir, dst_dir=tight, iterations=hole_shrink)
            masks_in = tight
        frames_to_mp4(frames_dir, input_video, fps=24.0)
        masks_to_mp4(masks_in, input_mask, fps=24.0)
        reencode_mp4_h264_inplace(input_video)
        reencode_mp4_h264_inplace(input_mask)
        repo_root = Path.cwd()
        # Keep all DiffuEraser weights under repo_root/ckpts/diffueraser/weights/...
        w = repo_root / "ckpts" / "diffueraser" / "weights"
        base_de = diffueraser.Params(
            base_model_path=w / "stable-diffusion-v1-5",
            vae_path=w / "sd-vae-ft-mse",
            diffueraser_path=w / "diffuEraser",
            propainter_model_dir=w / "propainter",
        )
        # object-removal vggt4dsam3_diffueraser.sh: video_length = ceil(n_frames / fps) when unset (fps=24).
        fps_de = 24.0
        vid_n = len(
            sorted(
                p
                for p in frames_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        )
        mask_n = len(sorted(masks_in.glob("*.png")))
        n_pair = min(vid_n, mask_n)
        if "video_length" not in do and n_pair > 0:
            do["video_length"] = max(1, int(math.ceil(n_pair / fps_de)))
        skip_paths = {"base_model_path", "vae_path", "diffueraser_path", "propainter_model_dir"}
        allowed_de = {f.name for f in fields(diffueraser.Params)} - skip_paths
        params = replace(base_de, **{k: v for k, v in do.items() if k in allowed_de})
        print(
            f"[inpaint] diffueraser: mask_dilation_iter={params.mask_dilation_iter} "
            f"mask_hole_shrink_iters={hole_shrink} ref_stride={params.ref_stride} "
            f"neighbor_length={params.neighbor_length} max_img_size={params.max_img_size} "
            f"video_length={params.video_length}"
        )
        meta = diffueraser.run(repo_root=repo_root, input_video=input_video, input_mask=input_mask, out_dir=tmp, params=params)
        # extract frames to canonical
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_to_frames(Path(meta["output_video"]), out_dir, ext=".png")
        write_method_manifest(layout.inpaint_dir, stage="inpaint", method=method, params=meta)
        return out_dir

    raise ValueError(f"Unknown inpaint method: {method}")

