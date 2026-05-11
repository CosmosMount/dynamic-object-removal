"""Mask overlay video export (aligned with object-removal pipelines/vggt4dsam3/postprocess_sam3.py)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from object_removal.utils.video import resolve_ffmpeg


def _list_frame_files(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=sort_key)


def export_mask_overlay_video(
    frames_dir: Path,
    masks_dir: Path,
    out_mp4: Path,
    *,
    fps: float = 10.0,
    alpha: float = 0.5,
    mask_color_bgr: tuple[int, int, int] = (0, 0, 255),
) -> bool:
    """Blend mask foreground onto RGB frames and encode H.264 with ffmpeg (fallback: OpenCV mp4v).

    Returns True if a non-empty video was written.
    """
    import cv2

    frame_files = _list_frame_files(frames_dir)
    if not frame_files:
        return False

    mask_names = {p.name for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}

    video_frames: List[np.ndarray] = []
    for fp in frame_files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = fp.stem
        mp = masks_dir / f"{stem}.png"
        if mp.name in mask_names:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                mask_bool = np.zeros((h, w), dtype=bool)
            else:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = m > 0
        else:
            mask_bool = np.zeros((h, w), dtype=bool)

        out = img.astype(np.float32)
        b0, g0, r0 = (float(mask_color_bgr[0]), float(mask_color_bgr[1]), float(mask_color_bgr[2]))
        for c, col in enumerate((b0, g0, r0)):
            ch = out[..., c]
            ch[mask_bool] = (1.0 - alpha) * ch[mask_bool] + alpha * col
        video_frames.append(out.astype(np.uint8))

    if not video_frames:
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mask_vis_") as tmpdir:
        tmp = Path(tmpdir)
        for i, frame in enumerate(video_frames):
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), frame)

        ffmpeg = resolve_ffmpeg()
        if ffmpeg is None:
            ffmpeg = "ffmpeg"
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp / "%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (w, h))
            if not vw.isOpened():
                return False
            for frame in video_frames:
                vw.write(frame)
            vw.release()
            return True
