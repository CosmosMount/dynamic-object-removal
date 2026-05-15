from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def resolve_ffmpeg() -> Optional[str]:
    """Find ffmpeg: PATH first, then common install paths (conda run often omits /usr/bin)."""
    w = shutil.which("ffmpeg")
    if w and Path(w).is_file():
        return w
    for p in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"):
        if Path(p).is_file():
            return p
    return None


def _ffmpeg_mux_images_to_h264(
    ffmpeg: str,
    image_paths: List[Path],
    out_mp4: Path,
    *,
    fps: float,
    crf: int = 20,
    preset: str = "medium",
) -> None:
    """Mux sorted images to one H.264 MP4 (yuv420p + faststart). Uses a temp %06d symlink chain."""
    if not image_paths:
        raise ValueError("empty image_paths")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    exts = {p.suffix.lower() for p in image_paths}
    if len(exts) != 1:
        raise ValueError(f"frames must share one extension for ffmpeg mux, got: {sorted(exts)}")
    ext = image_paths[0].suffix.lower()
    td = Path(tempfile.mkdtemp(prefix="objrem_frames_"))
    try:
        for i, src in enumerate(image_paths):
            dst = td / f"{i:06d}{ext}"
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        inp = str(td / f"%06d{ext}")
        # Even dimensions: match object-removal vggt4dsam3_diffueraser.sh (ceil(iw/2)*2), not trunc.
        vf = "format=yuv420p,scale=ceil(iw/2)*2:ceil(ih/2)*2"
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            inp,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            str(int(crf)),
            "-preset",
            str(preset),
            str(out_mp4),
        ]
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip()
            raise RuntimeError(f"ffmpeg mux failed ({p.returncode}): {err}")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def frames_to_mp4(frames_dir: Path, out_mp4: Path, *, fps: float = 24.0) -> None:
    files = sorted(
        [
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not files:
        raise ValueError(f"No frames found under: {frames_dir}")

    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        try:
            _ffmpeg_mux_images_to_h264(ffmpeg, files, out_mp4, fps=fps, crf=20, preset="medium")
            return
        except Exception as exc:
            print(f"[object_removal] ffmpeg mux frames failed, falling back to OpenCV: {exc}", file=sys.stderr)

    import cv2
    import numpy as np

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {files[0]}")
    h, w = first.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_mp4}")
    try:
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read: {p}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            writer.write(img)
    finally:
        writer.release()


def masks_to_mp4(masks_dir: Path, out_mp4: Path, *, fps: float = 24.0) -> None:
    files = sorted([p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if not files:
        raise ValueError(f"No masks found under: {masks_dir}")

    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        try:
            # object-removal DiffuEraser mask mp4: lossless H.264 (crf 0) + veryfast preset
            _ffmpeg_mux_images_to_h264(ffmpeg, files, out_mp4, fps=fps, crf=0, preset="veryfast")
            return
        except Exception as exc:
            print(f"[object_removal] ffmpeg mux masks failed, falling back to OpenCV: {exc}", file=sys.stderr)

    import cv2
    import numpy as np

    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Failed to read: {files[0]}")
    h, w = first.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_mp4}")
    try:
        for p in files:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Failed to read: {p}")
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def mp4_to_frames(video_path: Path, out_dir: Path, *, ext: str = ".png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        pattern = str(out_dir / f"%05d{ext}")
        p = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vsync",
                "0",
                "-start_number",
                "0",
                pattern,
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            produced = list(out_dir.glob(f"*{ext}"))
            if produced:
                return
        err = (p.stderr or p.stdout or "").strip()
        print(f"[object_removal] ffmpeg extract frames failed, using OpenCV: {err}", file=sys.stderr)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = out_dir / f"{i:05d}{ext}"
            ok2 = cv2.imwrite(str(out), frame)
            if not ok2:
                raise RuntimeError(f"Failed to write frame: {out}")
            i += 1
    finally:
        cap.release()


def reencode_mp4_h264_inplace(path: Path, *, crf: int = 20) -> bool:
    """Re-encode MP4 to H.264 yuv420p + faststart for players (e.g. VS Code) that choke on OpenCV mp4v."""
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        print(
            "[object_removal] ffmpeg not found; MP4 may stay mpeg4 (mp4v) and not play in VS Code. "
            "Install ffmpeg or ensure /usr/bin is on PATH.",
            file=sys.stderr,
        )
        return False
    tmp = path.with_name(path.stem + "._reencode_.mp4")
    vf = "scale=ceil(iw/2)*2:ceil(ih/2)*2"
    try:
        p = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                str(int(crf)),
                "-an",
                str(tmp),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip()
            print(f"[object_removal] ffmpeg reencode failed for {path}: {err}", file=sys.stderr)
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
            return False
    except OSError as exc:
        print(f"[object_removal] ffmpeg reencode OSError for {path}: {exc}", file=sys.stderr)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False
    try:
        os.replace(str(tmp), str(path))
    except OSError as exc:
        print(f"[object_removal] could not replace {path} with re-encoded file: {exc}", file=sys.stderr)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False
    return True
