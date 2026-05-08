from __future__ import annotations

from pathlib import Path

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.io.manifest import write_method_manifest
from object_removal.methods.mask import baseline, vggt4d, vggt_framewise, yolo


def run_mask_stage(
    *,
    run_dir: Path,
    frames_dir: Path,
    method: str,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> Path:
    layout = RunLayout(run_dir)
    ensure_layout_dirs(layout)

    out_dir = layout.mask_init_masks_dir
    existing = list(out_dir.glob("*.png"))
    if existing and not overwrite:
        return out_dir

    if method == "baseline_yolo_motion":
        meta = baseline.run(frames_dir=frames_dir, out_masks_dir=out_dir, max_frames=max_frames)
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params={**meta, "max_frames": max_frames})
        return out_dir

    if method == "vggt4d":
        scene_dir = layout.mask_init_dir / "vggt4d_scene"
        init_dir = out_dir
        params = vggt4d.Params()
        meta = vggt4d.run(
            repo_root=layout.run_dir.parent if (layout.run_dir / "modules").exists() else Path.cwd(),
            frames_dir=frames_dir,
            out_scene_dir=scene_dir,
            out_init_dir=init_dir,
            params=params,
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    if method == "yolo_first":
        # write first-frame indexed init mask as 00000.png
        first = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[0]
        out_path = out_dir / "00000.png"
        meta = yolo.run(
            first_frame=first,
            out_mask=out_path,
            params=yolo.Params(model_path=Path("ckpts/yolo/yolov8n-seg.pt")),
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    if method == "vggt_framewise":
        meta = vggt_framewise.run(
            repo_root=Path.cwd(),
            frames_dir=frames_dir,
            out_masks_dir=out_dir,
            params=vggt_framewise.Params(),
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    raise ValueError(f"Unknown mask method: {method}")

