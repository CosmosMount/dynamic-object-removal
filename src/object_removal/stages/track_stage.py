from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.io.manifest import write_method_manifest
from object_removal.methods.track import identity
from object_removal.methods.track import optflow, sam2, sam3, xmem


def run_track_stage(
    *,
    run_dir: Path,
    frames_dir: Path,
    in_masks_dir: Path,
    method: str,
    overwrite: bool = False,
    sam3_options: Optional[Dict[str, Any]] = None,
    xmem_options: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """@brief Run the track stage and normalize outputs into binary mask PNGs.

    @param run_dir Canonical per-pipeline run directory.
    @param frames_dir Input RGB frame directory.
    @param in_masks_dir Init-mask directory produced by the mask stage.
    @param method Track method id from the pipeline registry.
    @param overwrite Whether existing tracked masks should be cleared before rerunning.
    @param sam3_options Optional YAML-derived overrides for the SAM3 tracker.
    @param xmem_options Optional YAML-derived overrides for the XMem tracker.
    @param repo_root Optional repository root used to resolve checkpoints and modules.
    @return The canonical binary-mask directory under `run_dir/track`.
    @raises FileNotFoundError If a required init mask is missing.
    @raises ValueError If `method` is unknown.
    """
    layout = RunLayout(run_dir)
    root = (repo_root or Path.cwd()).resolve()
    ensure_layout_dirs(layout)

    out_dir = layout.track_masks_binary_dir
    if overwrite and out_dir.is_dir():
        for p in list(out_dir.glob("*.png")):
            p.unlink(missing_ok=True)
    # SAM3 writes indexed masks under track/<video_name>/; clearing only masks_binary/*.png leaves stale frames.
    if overwrite and method == "sam3":
        vn = frames_dir.name
        for rel in (vn, "tmp_sam3_init_masks", "masks_indexed_raw"):
            p = layout.track_dir / rel
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    if overwrite and method == "xmem":
        raw_xmem = layout.track_dir / "masks_indexed_raw"
        if raw_xmem.is_dir():
            shutil.rmtree(raw_xmem, ignore_errors=True)
    existing = list(out_dir.glob("*.png"))
    if existing and not overwrite:
        return out_dir

    if method == "identity":
        meta = identity.run(in_masks_dir=in_masks_dir, out_masks_dir=out_dir)
        write_method_manifest(layout.track_dir, stage="track", method=method, params=meta)
        return out_dir

    if method == "sam3":
        raw_root = layout.track_dir / "masks_indexed_raw"
        s3 = sam3_options or {}
        ckpt_raw = str(s3.get("checkpoint", "ckpts/sam3/sam3.pt"))
        ckpt_path = Path(ckpt_raw)
        if not ckpt_path.is_absolute():
            ckpt_path = (root / ckpt_path).resolve()
        meta = sam3.run(
            repo_root=root,
            frames_dir=frames_dir,
            init_mask_dir=in_masks_dir,
            out_raw_indexed_dir=raw_root,
            out_binary_dir=out_dir,
            params=sam3.Params(
                checkpoint=ckpt_path,
                two_stage_anchor_idx=str(s3.get("two_stage_anchor_idx", "auto")),
                two_stage_auto_samples=int(s3.get("two_stage_auto_samples", 7)),
                two_stage_auto_max_fg_frac=float(s3.get("two_stage_auto_max_fg_frac", 0.92)),
                two_stage_auto_min_fg_frac=float(s3.get("two_stage_auto_min_fg_frac", 0.00008)),
                two_stage_auto_min_fg_pixels=int(s3.get("two_stage_auto_min_fg_pixels", 64)),
                score_thresh=float(s3.get("score_thresh", 0.0)),
            ),
        )
        write_method_manifest(layout.track_dir, stage="track", method=method, params=meta)
        return out_dir

    if method == "sam2":
        # SAM2 expects a base dir containing <video_name>/frames
        video_name = frames_dir.name
        tmp_base = layout.track_dir / "tmp_sam2_base"
        tmp_in_masks = layout.track_dir / "tmp_sam2_init_masks" / video_name
        tmp_out = layout.track_dir / "tmp_sam2_masks_raw"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)
        tmp_in_masks.mkdir(parents=True, exist_ok=True)

        # symlink frames
        link = tmp_base / video_name
        if not link.exists():
            link.symlink_to(frames_dir.resolve(), target_is_directory=True)
        # copy init mask(s)
        for p in sorted(in_masks_dir.glob("*.png")):
            (tmp_in_masks / p.name).write_bytes(p.read_bytes())

        video_list = layout.track_dir / "video_list.txt"
        video_list.write_text(video_name + "\n", encoding="utf-8")

        meta = sam2.run(
            repo_root=root,
            base_video_dir=tmp_base,
            input_mask_dir=tmp_in_masks.parent,
            video_list_file=video_list,
            output_mask_dir=tmp_out,
            params=sam2.Params(
                cfg=Path("modules/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
                checkpoint=Path("ckpts/sam2/sam2.1_hiera_large.pt"),
                score_thresh=0.0,
            ),
        )

        # convert indexed masks to canonical binary
        from object_removal.io.masks import list_mask_files, read_mask_u8, to_binary_255, write_mask_u8

        raw_seq = tmp_out / video_name
        files = list_mask_files(raw_seq)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            write_mask_u8(out_dir / p.name, to_binary_255(read_mask_u8(p)))
        write_method_manifest(layout.track_dir, stage="track", method=method, params=meta)
        return out_dir

    if method == "optflow":
        # expects a first-frame init mask under in_masks_dir/00000.png (indexed or binary)
        init_mask = in_masks_dir / "00000.png"
        if not init_mask.is_file():
            raise FileNotFoundError(f"optflow requires init mask: {init_mask}")
        meta = optflow.run(
            frames_dir=frames_dir,
            init_mask_path=init_mask,
            out_masks_dir=out_dir,
            params=optflow.Params(yolo_model=Path("ckpts/yolo/yolov8n-seg.pt")),
        )
        write_method_manifest(layout.track_dir, stage="track", method=method, params=meta)
        return out_dir

    if method == "xmem":
        meta = xmem.run(
            repo_root=root,
            frames_dir=frames_dir,
            in_masks_dir=in_masks_dir,
            out_binary_dir=out_dir,
            params=xmem.Params(),
            options=xmem_options,
        )
        write_method_manifest(layout.track_dir, stage="track", method=method, params=meta)
        return out_dir

    raise ValueError(f"Unknown track method: {method}")

