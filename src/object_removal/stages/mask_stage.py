from __future__ import annotations

import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

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
    vggt4d_options: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    layout = RunLayout(run_dir)
    ensure_layout_dirs(layout)

    out_dir = layout.mask_init_masks_dir
    if overwrite and out_dir.is_dir():
        for p in list(out_dir.glob("*.png")):
            p.unlink(missing_ok=True)
    existing = list(out_dir.glob("*.png"))
    if existing and not overwrite:
        return out_dir

    root = (repo_root or Path.cwd()).resolve()
    if (run_dir / "modules").is_dir():
        inferred = run_dir
    else:
        inferred = root

    if method == "baseline_yolo_motion":
        meta = baseline.run(frames_dir=frames_dir, out_masks_dir=out_dir, max_frames=max_frames)
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params={**meta, "max_frames": max_frames})
        return out_dir

    if method == "vggt4d":
        # Overwrite must drop VGGT scene outputs; otherwise stale dynamic_mask_*.png can remain
        # (e.g. after max_frames / code change) and glob() picks wrong files — full run_dir delete avoided here.
        if overwrite:
            for rel in ("vggt_output", "vggt_chunks", "vggt_input"):
                d = layout.mask_init_dir / rel
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
        scene_dir = layout.mask_init_dir / "vggt4d_scene"
        init_dir = out_dir
        base = vggt4d.Params()
        vo = vggt4d_options or {}
        allowed = {f.name for f in fields(vggt4d.Params)}
        params = replace(base, **{k: v for k, v in vo.items() if k in allowed})
        meta = vggt4d.run(
            repo_root=inferred if (inferred / "modules").exists() else root,
            frames_dir=frames_dir,
            out_scene_dir=scene_dir,
            out_init_dir=init_dir,
            params=params,
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    if method == "yolo_first":
        first = sorted(
            [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )[0]
        out_path = out_dir / "00000.png"
        meta = yolo.run(
            first_frame=first,
            out_mask=out_path,
            params=yolo.Params(model_path=root / "ckpts/yolo/yolov8n-seg.pt"),
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    if method == "vggt_framewise":
        base_fw = vggt_framewise.Params()
        vo = vggt4d_options or {}
        allowed_fw = {f.name for f in fields(vggt_framewise.Params)}
        params_fw = replace(base_fw, **{k: v for k, v in vo.items() if k in allowed_fw})
        meta = vggt_framewise.run(
            repo_root=inferred if (inferred / "modules").exists() else root,
            frames_dir=frames_dir,
            out_masks_dir=out_dir,
            params=params_fw,
        )
        write_method_manifest(layout.mask_init_dir, stage="mask", method=method, params=meta)
        return out_dir

    raise ValueError(f"Unknown mask method: {method}")
