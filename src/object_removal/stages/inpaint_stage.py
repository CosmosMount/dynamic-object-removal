from __future__ import annotations

from pathlib import Path

from object_removal.io.layout import RunLayout, ensure_layout_dirs
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
        meta = propainter.run(repo_root=Path.cwd(), frames_dir=frames_dir, masks_dir=masks_dir, out_root=tmp_root, params=propainter.Params())
        # normalize to canonical run_dir/inpaint/frames
        pp_frames = Path(meta.get("frames_out", ""))
        if pp_frames.is_dir():
            # copy as %05d.png to out_dir
            import shutil

            out_dir.mkdir(parents=True, exist_ok=True)
            srcs = sorted([p for p in pp_frames.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            for i, p in enumerate(srcs):
                shutil.copy2(p, out_dir / f"{i:05d}.png")
        write_method_manifest(layout.inpaint_dir, stage="inpaint", method=method, params=meta)
        return out_dir

    if method == "diffueraser":
        # prepare inputs for diffueraser
        tmp = layout.inpaint_dir / "diffueraser"
        tmp.mkdir(parents=True, exist_ok=True)
        input_video = tmp / "input_video.mp4"
        input_mask = tmp / "input_mask.mp4"
        frames_to_mp4(frames_dir, input_video, fps=24.0)
        masks_to_mp4(masks_dir, input_mask, fps=24.0)
        reencode_mp4_h264_inplace(input_video)
        reencode_mp4_h264_inplace(input_mask)
        repo_root = Path.cwd()
        # Keep all DiffuEraser weights under repo_root/ckpts/diffueraser/weights/...
        w = repo_root / "ckpts" / "diffueraser" / "weights"
        params = diffueraser.Params(
            base_model_path=w / "stable-diffusion-v1-5",
            vae_path=w / "sd-vae-ft-mse",
            diffueraser_path=w / "diffuEraser",
            propainter_model_dir=w / "propainter",
        )
        meta = diffueraser.run(repo_root=repo_root, input_video=input_video, input_mask=input_mask, out_dir=tmp, params=params)
        # extract frames to canonical
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_to_frames(Path(meta["output_video"]), out_dir, ext=".png")
        write_method_manifest(layout.inpaint_dir, stage="inpaint", method=method, params=meta)
        return out_dir

    raise ValueError(f"Unknown inpaint method: {method}")

