from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalInputs:
    output_dir: Path
    part_label: str = ""
    experiment_name: str = ""
    davis_csv: Path | None = None
    pred_mask_dir: Path | None = None
    gt_mask_dir: Path | None = None
    merge_gt_objects: bool = True
    pred_video: Path | None = None
    gt_video: Path | None = None
    pred_frames_dir: Path | None = None
    gt_frames_dir: Path | None = None
    video_metric_impl: str = "internal"  # "internal" or "propainter"


def _float_or_none(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _resolve_csv_value(row: dict, keys: List[str]) -> Optional[float]:
    normalized = {k.strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = normalized.get(key.strip().lower())
        out = _float_or_none(value)
        if out is not None:
            return out
    return None


def load_davis_jm_jr(davis_csv: Path) -> Tuple[Optional[float], Optional[float]]:
    import csv

    if not davis_csv.exists():
        return None, None
    with davis_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, None
    row = rows[0]
    jm = _resolve_csv_value(row, ["J-Mean", "JMean", "J Mean", "JM"])
    jr = _resolve_csv_value(row, ["J-Recall", "JRecall", "J Recall", "JR"])
    return jm, jr


def _list_image_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _load_mask(path: Path, merge_objects: bool = True) -> np.ndarray:
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if merge_objects:
        return mask > 0
    return mask


def compute_mask_jm_jr(pred_dir: Path, gt_dir: Path, merge_gt_objects: bool = True) -> Tuple[Optional[float], Optional[float], int]:
    pred_files = _list_image_files(pred_dir)
    gt_files = _list_image_files(gt_dir)
    if not pred_files or not gt_files:
        return None, None, 0

    pred_map = {p.name: p for p in pred_files}
    gt_map = {p.name: p for p in gt_files}
    common_names = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    if not common_names:
        n = min(len(pred_files), len(gt_files))
        pred_files = pred_files[:n]
        gt_files = gt_files[:n]
        pairs = list(zip(pred_files, gt_files))
    else:
        pairs = [(pred_map[n], gt_map[n]) for n in common_names]

    ious: List[float] = []
    for pred_path, gt_path in pairs:
        pred = _load_mask(pred_path, merge_objects=False)
        gt = _load_mask(gt_path, merge_objects=merge_gt_objects)
        if pred.shape != gt.shape:
            pred_u8 = (pred.astype(np.uint8) * 255)
            pred_u8 = cv2.resize(pred_u8, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = pred_u8 > 0

        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = 1.0 if union == 0 else float(inter) / float(union)
        ious.append(iou)

    if not ious:
        return None, None, 0

    arr = np.asarray(ious, dtype=np.float64)
    jm = float(arr.mean())
    jr = float((arr > 0.5).mean())
    return jm, jr, int(arr.size)


def _load_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    import cv2

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def _ssim_color(img1: np.ndarray, img2: np.ndarray) -> float:
    scores = [_ssim_single_channel(img1[..., c], img2[..., c]) for c in range(3)]
    return float(np.mean(scores))


def _internal_calc_psnr_and_ssim(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
    import cv2

    return float(cv2.PSNR(img1, img2)), _ssim_color(img1, img2)


def resolve_video_metric_fn(impl: str, *, propainter_root: Path | None = None) -> Callable[[np.ndarray, np.ndarray], Tuple[float, float]]:
    if impl == "internal":
        return _internal_calc_psnr_and_ssim

    # Optional: ProPainter metrics (kept for parity with original script)
    import importlib.util
    import sys

    if propainter_root is None:
        raise RuntimeError("propainter_root must be provided when video_metric_impl='propainter'")
    metrics_file = propainter_root / "core" / "metrics.py"
    if not metrics_file.is_file():
        raise RuntimeError(f"ProPainter metrics file not found: {metrics_file}")

    if str(propainter_root) not in sys.path:
        sys.path.insert(0, str(propainter_root))

    spec = importlib.util.spec_from_file_location("propainter_core_metrics", metrics_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ProPainter metrics module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "calc_psnr_and_ssim"):
        raise RuntimeError("ProPainter calc_psnr_and_ssim not found")
    return module.calc_psnr_and_ssim


def compute_video_metrics_from_frame_dirs(
    pred_dir: Path,
    gt_dir: Path,
    metric_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float], int]:
    pred_files = _list_image_files(pred_dir)
    gt_files = _list_image_files(gt_dir)
    if not pred_files or not gt_files:
        return None, None, 0

    pred_map = {p.name: p for p in pred_files}
    gt_map = {p.name: p for p in gt_files}
    common_names = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    if common_names:
        pairs = [(pred_map[n], gt_map[n]) for n in common_names]
    else:
        n = min(len(pred_files), len(gt_files))
        pairs = list(zip(pred_files[:n], gt_files[:n]))

    psnrs: List[float] = []
    ssims: List[float] = []
    for pred_path, gt_path in pairs:
        pred = _load_bgr(pred_path)
        gt = _load_bgr(gt_path)
        if pred.shape != gt.shape:
            import cv2

            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr_i, ssim_i = metric_fn(gt, pred)
        psnrs.append(float(psnr_i))
        ssims.append(float(ssim_i))

    if not psnrs:
        return None, None, 0
    return float(np.mean(psnrs)), float(np.mean(ssims)), len(psnrs)


def _iter_video_frames(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def compute_video_metrics_from_videos(
    pred_video: Path,
    gt_video: Path,
    metric_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float], int]:
    if not pred_video.exists() or not gt_video.exists():
        return None, None, 0
    psnrs: List[float] = []
    ssims: List[float] = []
    for pred, gt in zip(_iter_video_frames(pred_video), _iter_video_frames(gt_video)):
        if pred.shape != gt.shape:
            import cv2

            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr_i, ssim_i = metric_fn(gt, pred)
        psnrs.append(float(psnr_i))
        ssims.append(float(ssim_i))
    if not psnrs:
        return None, None, 0
    return float(np.mean(psnrs)), float(np.mean(ssims)), len(psnrs)


def write_summary(output_dir: Path, summary: dict) -> Tuple[Path, Path]:
    import csv
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics_summary.json"
    csv_path = output_dir / "metrics_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = list(summary.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)
    return json_path, csv_path


def run_eval(inputs: EvalInputs, *, propainter_root: Path | None = None) -> dict:
    metric_fn = resolve_video_metric_fn(inputs.video_metric_impl, propainter_root=propainter_root)

    mask_jm = None
    mask_jr = None
    mask_frames = 0
    mask_source = "none"

    if inputs.davis_csv and inputs.davis_csv.exists() and not inputs.merge_gt_objects:
        jm, jr = load_davis_jm_jr(inputs.davis_csv)
        if jm is not None and jr is not None:
            mask_jm, mask_jr = jm, jr
            mask_source = "davis_csv"

    if mask_jm is None and inputs.pred_mask_dir and inputs.gt_mask_dir:
        jm, jr, n = compute_mask_jm_jr(inputs.pred_mask_dir, inputs.gt_mask_dir, merge_gt_objects=inputs.merge_gt_objects)
        if jm is not None and jr is not None:
            mask_jm, mask_jr, mask_frames = jm, jr, n
            mask_source = "mask_dir"

    psnr = None
    ssim = None
    video_frames = 0
    video_source = "none"

    if inputs.pred_frames_dir and inputs.gt_frames_dir:
        psnr, ssim, video_frames = compute_video_metrics_from_frame_dirs(inputs.pred_frames_dir, inputs.gt_frames_dir, metric_fn)
        if psnr is not None and ssim is not None:
            video_source = "frame_dir"

    if psnr is None and inputs.pred_video and inputs.gt_video:
        psnr, ssim, video_frames = compute_video_metrics_from_videos(inputs.pred_video, inputs.gt_video, metric_fn)
        if psnr is not None and ssim is not None:
            video_source = "video"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "part_label": inputs.part_label,
        "experiment_name": inputs.experiment_name,
        "mask_jm": mask_jm,
        "mask_jr": mask_jr,
        "mask_frames": mask_frames,
        "mask_source": mask_source,
        "video_psnr": psnr,
        "video_ssim": ssim,
        "video_frames": video_frames,
        "video_source": video_source,
        "video_metric_impl": inputs.video_metric_impl,
        "davis_csv": str(inputs.davis_csv) if inputs.davis_csv else "",
        "pred_mask_dir": str(inputs.pred_mask_dir) if inputs.pred_mask_dir else "",
        "gt_mask_dir": str(inputs.gt_mask_dir) if inputs.gt_mask_dir else "",
        "pred_video": str(inputs.pred_video) if inputs.pred_video else "",
        "gt_video": str(inputs.gt_video) if inputs.gt_video else "",
        "pred_frames_dir": str(inputs.pred_frames_dir) if inputs.pred_frames_dir else "",
        "gt_frames_dir": str(inputs.gt_frames_dir) if inputs.gt_frames_dir else "",
    }

    write_summary(inputs.output_dir, summary)
    return summary

