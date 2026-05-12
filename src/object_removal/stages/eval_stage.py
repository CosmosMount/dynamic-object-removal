from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalInputs:
    """@brief Inputs required to compute evaluation summaries for one pipeline run.

    Paths are optional because compare may evaluate from either per-frame masks,
    a DAVIS CSV summary, generated inpaint frames, or a subset of those artifacts.
    """

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
    source_frames_dir: Path | None = None  # Original RGB frames used by bg-L1 and temporal metrics.
    video_metric_impl: str = "internal"  # retained for API compat; GT PSNR/SSIM no longer used
    # Optional no-reference inpaint quality (FAST-VQA / FasterVQA via external repo clone).
    fast_vqa: bool = False
    fast_vqa_root: Path | None = None
    fast_vqa_model: str = "FasterVQA"
    fast_vqa_device: str = "cuda"
    fast_vqa_fps: float = 24.0
    # If set, runs upstream vqa.py with this interpreter (else FAST_VQA_PYTHON env, else sys.executable).
    fast_vqa_python: str | None = None


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


def _mask_score_from_components(
    jm: Optional[float],
    jr: Optional[float],
    fm: Optional[float],
    fr: Optional[float],
) -> Optional[float]:
    """Blend mask metrics into one summary score.

    When per-frame masks are available, the score is the mean of JM, FM, and FR.
    When only a DAVIS CSV summary is available, the fallback score is `(JM + JR) / 2`.
    """
    if jm is None:
        return None
    if fm is not None and fr is not None:
        return float((jm + fm + fr) / 3.0)
    if fm is not None:
        return float((jm + fm) / 2.0)
    if fr is not None:
        return float((jm + fr) / 2.0)
    if jr is not None:
        return float(0.5 * jm + 0.5 * jr)
    return float(jm)


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


def _boundary_f_pr(
    pred: np.ndarray, gt: np.ndarray, dilate_px: int = 2
) -> Tuple[float, float, float]:
    """Boundary precision, recall, F on dilated contour bands (pred/gt bool HxW)."""
    import cv2

    pu = pred.astype(np.uint8) * 255
    gu = gt.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    bd_p = (cv2.subtract(pu, cv2.erode(pu, k)) > 0) | (cv2.subtract(cv2.dilate(pu, k), pu) > 0)
    bd_g = (cv2.subtract(gu, cv2.erode(gu, k)) > 0) | (cv2.subtract(cv2.dilate(gu, k), gu) > 0)
    if dilate_px > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        band_g = cv2.dilate((bd_g.astype(np.uint8) * 255), dk) > 0
        band_p = cv2.dilate((bd_p.astype(np.uint8) * 255), dk) > 0
    else:
        band_g = bd_g
        band_p = bd_p
    np_bd = int(bd_p.sum())
    ng_bd = int(bd_g.sum())
    if np_bd == 0 and ng_bd == 0:
        return 1.0, 1.0, 1.0
    prec = float(np.logical_and(bd_p, band_g).sum()) / max(np_bd, 1)
    rec = float(np.logical_and(bd_g, band_p).sum()) / max(ng_bd, 1)
    if prec + rec < 1e-8:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def compute_mask_metrics_bundle(
    pred_dir: Path, gt_dir: Path, merge_gt_objects: bool = True
) -> Dict[str, Any]:
    """Compute the full mask metric bundle from predicted and ground-truth mask folders."""
    import cv2

    pred_files = _list_image_files(pred_dir)
    gt_files = _list_image_files(gt_dir)
    out: Dict[str, Any] = {
        "mask_jm": None,
        "mask_jr": None,
        "mask_mean_recall": None,
        "mask_mean_precision": None,
        "mask_mean_dice": None,
        "mask_fm": None,
        "mask_fr": None,
        "mask_score": None,
        "mask_frames": 0,
        "mask_frames_gt_empty": 0,
    }
    if not pred_files or not gt_files:
        return out

    pred_map = {p.name: p for p in pred_files}
    gt_map = {p.name: p for p in gt_files}
    common_names = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common_names:
        n = min(len(pred_files), len(gt_files))
        pairs = list(zip(pred_files[:n], gt_files[:n]))
    else:
        pairs = [(pred_map[n], gt_map[n]) for n in common_names]

    ious: List[float] = []
    recalls: List[float] = []
    precs: List[float] = []
    dices: List[float] = []
    f1s: List[float] = []
    brs: List[float] = []
    gt_empty = 0

    for pred_path, gt_path in pairs:
        pred = _load_mask(pred_path, merge_objects=False)
        gt = _load_mask(gt_path, merge_objects=merge_gt_objects)
        if pred.shape != gt.shape:
            pred_u8 = (pred.astype(np.uint8) * 255)
            pred_u8 = cv2.resize(pred_u8, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = pred_u8 > 0

        g_count = int(gt.sum())
        p_count = int(pred.sum())
        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        iou = 1.0 if union == 0 else float(inter) / float(union)
        ious.append(iou)

        if g_count == 0:
            gt_empty += 1
            recalls.append(1.0 if p_count == 0 else 0.0)
            precs.append(1.0 if p_count == 0 else 0.0)
            dices.append(1.0 if p_count == 0 else 0.0)
        else:
            recalls.append(float(inter) / float(g_count))
            precs.append(float(inter) / float(max(p_count, 1)))
            dices.append(2.0 * float(inter) / float(max(p_count + g_count, 1)))

        _, br, f1 = _boundary_f_pr(pred, gt)
        f1s.append(f1)
        brs.append(br)

    if not ious:
        return out

    arr_iou = np.asarray(ious, dtype=np.float64)
    out["mask_jm"] = float(arr_iou.mean())
    out["mask_jr"] = float((arr_iou > 0.5).mean())
    out["mask_mean_recall"] = float(np.mean(recalls))
    out["mask_mean_precision"] = float(np.mean(precs))
    out["mask_mean_dice"] = float(np.mean(dices))
    out["mask_fm"] = float(np.mean(f1s))
    out["mask_fr"] = float(np.mean(brs))
    out["mask_frames"] = int(len(ious))
    out["mask_frames_gt_empty"] = int(gt_empty)
    out["mask_score"] = _mask_score_from_components(
        out["mask_jm"], out["mask_jr"], out["mask_fm"], out["mask_fr"]
    )
    return out


def _load_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _align_pairs(
    pred_dir: Path, src_dir: Path, mask_dir: Path
) -> List[Tuple[Path, Path, Path]]:
    pred_files = _list_image_files(pred_dir)
    src_files = _list_image_files(src_dir)
    mask_files = _list_image_files(mask_dir)
    if not pred_files or not src_files or not mask_files:
        return []
    pm, sm, mm = {p.name: p for p in pred_files}, {p.name: p for p in src_files}, {p.name: p for p in mask_files}
    names = sorted(set(pm.keys()) & set(sm.keys()) & set(mm.keys()))
    if not names:
        n = min(len(pred_files), len(src_files), len(mask_files))
        return list(zip(pred_files[:n], src_files[:n], mask_files[:n]))
    return [(pm[n], sm[n], mm[n]) for n in names]


def compute_bg_l1_mean(
    pred_dir: Path, source_dir: Path, mask_dir: Path, merge_mask: bool = True
) -> Tuple[Optional[float], int, str]:
    """Mean L1 on pixels outside (dilated) foreground mask."""
    import cv2

    pairs = _align_pairs(pred_dir, source_dir, mask_dir)
    if len(pairs) < 1:
        return None, 0, "no_pairs"
    vals: List[float] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for pp, sp, mp in pairs:
        pred = _load_bgr(pp)
        src = _load_bgr(sp)
        m = _load_mask(mp, merge_objects=merge_mask)
        if pred.shape != src.shape:
            pred = cv2.resize(pred, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        if m.shape[:2] != src.shape[:2]:
            m_u8 = (m.astype(np.uint8) * 255)
            m = cv2.resize(m_u8, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        region = cv2.dilate(m.astype(np.uint8), kernel, iterations=1) == 0
        if not np.any(region):
            continue
        diff = np.abs(pred.astype(np.float64) - src.astype(np.float64))
        vals.append(float(diff[region].mean()))
    if not vals:
        return None, len(pairs), "no_valid_region"
    return float(np.mean(vals)), len(pairs), "ok"


def _warp_flow_bgr(prev_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    import cv2

    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(prev_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compute_temporal_warping_and_flow_consistency(
    pred_dir: Path, source_dir: Path, mask_dir: Path, merge_mask: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[float], int, str]:
    """Compute temporal warp error and flow consistency from consecutive frames.

    Optical flow is estimated on grayscale source frames. The function reports mean
    warp L1 on background pixels, mean warp L1 inside the inpaint hole, and the
    frame-to-frame flow difference on background pixels.
    """
    import cv2

    pairs = _align_pairs(pred_dir, source_dir, mask_dir)
    if len(pairs) < 2:
        return None, None, None, len(pairs), "need_at_least_2_frames"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    warp_bg: List[float] = []
    warp_hole: List[float] = []
    flow_diffs: List[float] = []
    flows: List[np.ndarray] = []

    for i in range(1, len(pairs)):
        pp_prev, sp_prev, mp_prev = pairs[i - 1]
        pp_curr, sp_curr, mp_curr = pairs[i]
        prev_src = cv2.cvtColor(_load_bgr(sp_prev), cv2.COLOR_BGR2GRAY)
        curr_src = cv2.cvtColor(_load_bgr(sp_curr), cv2.COLOR_BGR2GRAY)
        prev_pred = _load_bgr(pp_prev)
        curr_pred = _load_bgr(pp_curr)
        m_curr = _load_mask(mp_curr, merge_objects=merge_mask)
        # Normalize everything to the current source-frame resolution before flow and warping.
        h, w = int(curr_src.shape[0]), int(curr_src.shape[1])
        if prev_src.shape[:2] != (h, w):
            prev_src = cv2.resize(prev_src, (w, h), interpolation=cv2.INTER_LINEAR)
        if curr_src.shape[:2] != (h, w):
            curr_src = cv2.resize(curr_src, (w, h), interpolation=cv2.INTER_LINEAR)
        prev_pred = cv2.resize(prev_pred, (w, h), interpolation=cv2.INTER_LINEAR)
        curr_pred = cv2.resize(curr_pred, (w, h), interpolation=cv2.INTER_LINEAR)
        if m_curr.shape[:2] != (h, w):
            m_u8 = (m_curr.astype(np.uint8) * 255)
            m_curr = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        flow = cv2.calcOpticalFlowFarneback(
            prev_src, curr_src, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
        warped = _warp_flow_bgr(prev_pred, flow)
        m_d = cv2.dilate(m_curr.astype(np.uint8), kernel, iterations=1) > 0
        region_bg = ~m_d
        region_hole = m_d
        diff = np.abs(warped.astype(np.float64) - curr_pred.astype(np.float64))
        if np.any(region_bg):
            warp_bg.append(float(diff[region_bg].mean()))
        if np.any(region_hole):
            warp_hole.append(float(diff[region_hole].mean()))
        if len(flows) >= 2 and flows[-1].shape == flows[-2].shape:
            d = flows[-1].astype(np.float64) - flows[-2].astype(np.float64)
            if np.any(region_bg) and region_bg.shape == d.shape[:2]:
                fd = float(np.sqrt((d * d).sum(axis=-1))[region_bg].mean())
                flow_diffs.append(fd)

    if not warp_bg and not warp_hole:
        return None, None, None, len(pairs), "no_warp_samples"
    bg_mean = float(np.mean(warp_bg)) if warp_bg else None
    hole_mean = float(np.mean(warp_hole)) if warp_hole else None
    f_mean = float(np.mean(flow_diffs)) if flow_diffs else None
    return bg_mean, hole_mean, f_mean, len(pairs), "ok"


def compute_laplacian_var_mean(frames_dir: Path) -> Tuple[Optional[float], int, str]:
    """Compute the mean grayscale Laplacian variance across all frames."""
    import cv2

    files = _list_image_files(frames_dir)
    if not files:
        return None, 0, "no_frames"
    vals: List[float] = []
    # OpenCV Python exposes `CV_64F` / `CV_32F`; there is no `CV64F` alias.
    lap_depth = getattr(cv2, "CV_64F", None) or getattr(cv2, "CV_32F", 5)
    for p in files:
        gray = cv2.cvtColor(_load_bgr(p), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, lap_depth)
        vals.append(float(np.asarray(lap, dtype=np.float64).var()))
    return float(np.mean(vals)), len(vals), "ok"


def compute_brisque_mean_optional(frames_dir: Path) -> Tuple[Optional[float], int]:
    """Return the mean BRISQUE score when OpenCV contrib quality is available."""
    files = _list_image_files(frames_dir)
    if not files:
        return None, 0
    br_vals: List[float] = []
    for p in files:
        b = _try_brisque(_load_bgr(p))
        if b is not None:
            br_vals.append(b)
    if not br_vals:
        return None, len(files)
    return float(np.mean(br_vals)), len(files)


def _try_brisque(bgr: np.ndarray) -> Optional[float]:
    import cv2

    qm = getattr(cv2, "quality", None)
    if qm is None:
        return None
    br = getattr(qm, "QualityBRISQUE_compute", None)
    if br is None:
        return None
    try:
        r = br(bgr, None)
        return float(r[0])
    except Exception:
        return None


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


def write_zero_eval_summary(
    output_dir: Path,
    *,
    experiment_name: str,
    part_label: str = "",
    skip_reason: str = "skipped_no_init_mask",
) -> dict:
    """Minimal summary for skipped runs (numeric placeholders where needed)."""
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "part_label": part_label,
        "experiment_name": experiment_name,
        "mask_jm": 0.0,
        "mask_jr": 0.0,
        "mask_mean_recall": None,
        "mask_mean_precision": None,
        "mask_mean_dice": None,
        "mask_fm": None,
        "mask_fr": None,
        "mask_score": None,
        "mask_frames": 0,
        "mask_frames_gt_empty": 0,
        "mask_source": skip_reason,
        "bg_l1_mean": None,
        "bg_l1_frames": 0,
        "bg_l1_source": skip_reason,
        "temporal_warp_error_mean": None,
        "temporal_warp_error_hole_mean": None,
        "flow_consistency_mean": None,
        "temporal_frames": 0,
        "temporal_source": skip_reason,
        "laplacian_var_mean": None,
        "laplacian_var_frames": 0,
        "laplacian_var_source": skip_reason,
        "brisque_mean": None,
        "brisque_frames": 0,
        "quality_score": None,
        "quality_score_source": "disabled_no_single_reference_metric",
        "quality_norm": {},
        "davis_csv": "",
        "pred_mask_dir": "",
        "gt_mask_dir": "",
        "pred_video": "",
        "gt_video": "",
        "pred_frames_dir": "",
        "gt_frames_dir": "",
        "source_frames_dir": "",
        "video_metric_impl": "deprecated_no_gt_video_metrics",
        "fast_vqa_score": None,
        "fast_vqa_source": skip_reason,
        "fast_vqa_model": "",
        "fast_vqa_fps": 0.0,
    }
    write_summary(output_dir, summary)
    return summary


def run_eval(inputs: EvalInputs, *, propainter_root: Path | None = None) -> dict:
    """@brief Evaluate one pipeline run and write a metrics summary to disk.

    @param inputs Resolved evaluation inputs and optional artifact paths.
    @param propainter_root Deprecated compatibility parameter retained for old call sites.
    @return A summary dictionary identical to the JSON/CSV payload written to `inputs.output_dir`.
    """
    _ = propainter_root  # GT video metrics removed; kept for call-site compatibility

    mask_source = "none"
    mask_block: Dict[str, Any] = {
        "mask_jm": None,
        "mask_jr": None,
        "mask_mean_recall": None,
        "mask_mean_precision": None,
        "mask_mean_dice": None,
        "mask_fm": None,
        "mask_fr": None,
        "mask_score": None,
        "mask_frames": 0,
        "mask_frames_gt_empty": 0,
    }

    if inputs.davis_csv and inputs.davis_csv.exists() and not inputs.merge_gt_objects:
        jm, jr = load_davis_jm_jr(inputs.davis_csv)
        if jm is not None and jr is not None:
            mask_block["mask_jm"] = jm
            mask_block["mask_jr"] = jr
            mask_block["mask_score"] = _mask_score_from_components(jm, jr, None, None)
            mask_source = "davis_csv"

    if mask_block["mask_jm"] is None and inputs.pred_mask_dir and inputs.gt_mask_dir:
        bundle = compute_mask_metrics_bundle(
            inputs.pred_mask_dir, inputs.gt_mask_dir, merge_gt_objects=inputs.merge_gt_objects
        )
        if bundle["mask_jm"] is not None:
            mask_block.update({k: bundle[k] for k in bundle})
            mask_source = "mask_dir"

    src_dir = inputs.source_frames_dir
    pred_dir = inputs.pred_frames_dir
    mask_dir = inputs.pred_mask_dir

    bg_l1, bg_n, bg_src = None, 0, "none"
    warp_bg, warp_hole, flow_m, temp_n, temp_src = None, None, None, 0, "none"
    lap_m, lap_n, lap_src = None, 0, "none"
    br_m, br_n = None, 0

    if pred_dir and pred_dir.is_dir() and src_dir and src_dir.is_dir() and mask_dir and mask_dir.is_dir():
        bg_l1, bg_n, bg_src = compute_bg_l1_mean(pred_dir, src_dir, mask_dir, merge_mask=True)
        warp_bg, warp_hole, flow_m, temp_n, temp_src = compute_temporal_warping_and_flow_consistency(
            pred_dir, src_dir, mask_dir, merge_mask=True
        )
    if pred_dir and pred_dir.is_dir():
        lap_m, lap_n, lap_src = compute_laplacian_var_mean(pred_dir)
        br_m, br_n = compute_brisque_mean_optional(pred_dir)

    fast_vqa_score: Optional[float] = None
    fast_vqa_source = "disabled"
    fast_vqa_model_used = ""
    if inputs.fast_vqa and inputs.fast_vqa_root and pred_dir and pred_dir.is_dir():
        from object_removal.metrics.fast_vqa import run_fast_vqa_on_frames_dir

        fast_vqa_model_used = str(inputs.fast_vqa_model).strip() or "FasterVQA"
        fast_vqa_score, fast_vqa_source = run_fast_vqa_on_frames_dir(
            pred_dir,
            fast_vqa_root=inputs.fast_vqa_root,
            model=fast_vqa_model_used,
            device=str(inputs.fast_vqa_device).strip() or "cuda",
            fps=float(inputs.fast_vqa_fps),
            python_exe=inputs.fast_vqa_python,
        )
        if fast_vqa_score is None:
            print(f"[eval] fast_vqa skipped/failed: {fast_vqa_source}", flush=True)
        else:
            print(f"[eval] fast_vqa_score={fast_vqa_score:.5f} ({fast_vqa_model_used})", flush=True)
    elif inputs.fast_vqa and not inputs.fast_vqa_root:
        fast_vqa_source = "disabled_no_fast_vqa_root"

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "part_label": inputs.part_label,
        "experiment_name": inputs.experiment_name,
        "mask_jm": mask_block["mask_jm"],
        "mask_jr": mask_block["mask_jr"],
        "mask_mean_recall": mask_block["mask_mean_recall"],
        "mask_mean_precision": mask_block["mask_mean_precision"],
        "mask_mean_dice": mask_block["mask_mean_dice"],
        "mask_fm": mask_block["mask_fm"],
        "mask_fr": mask_block["mask_fr"],
        "mask_score": mask_block["mask_score"],
        "mask_frames": mask_block["mask_frames"],
        "mask_frames_gt_empty": mask_block["mask_frames_gt_empty"],
        "mask_source": mask_source,
        "bg_l1_mean": bg_l1,
        "bg_l1_frames": bg_n,
        "bg_l1_source": bg_src,
        "temporal_warp_error_mean": warp_bg,
        "temporal_warp_error_hole_mean": warp_hole,
        "flow_consistency_mean": flow_m,
        "temporal_frames": temp_n,
        "temporal_source": temp_src,
        "laplacian_var_mean": lap_m,
        "laplacian_var_frames": lap_n,
        "laplacian_var_source": lap_src,
        "brisque_mean": br_m,
        "brisque_frames": br_n,
        "quality_score": None,
        "quality_score_source": "disabled_no_single_reference_metric",
        "quality_norm": {},
        "davis_csv": str(inputs.davis_csv) if inputs.davis_csv else "",
        "pred_mask_dir": str(inputs.pred_mask_dir) if inputs.pred_mask_dir else "",
        "gt_mask_dir": str(inputs.gt_mask_dir) if inputs.gt_mask_dir else "",
        "pred_video": str(inputs.pred_video) if inputs.pred_video else "",
        "gt_video": str(inputs.gt_video) if inputs.gt_video else "",
        "pred_frames_dir": str(inputs.pred_frames_dir) if inputs.pred_frames_dir else "",
        "gt_frames_dir": str(inputs.gt_frames_dir) if inputs.gt_frames_dir else "",
        "source_frames_dir": str(src_dir) if src_dir else "",
        "video_metric_impl": inputs.video_metric_impl,
        "fast_vqa_score": fast_vqa_score,
        "fast_vqa_source": fast_vqa_source,
        "fast_vqa_model": fast_vqa_model_used,
        "fast_vqa_fps": float(inputs.fast_vqa_fps) if inputs.fast_vqa else 0.0,
    }

    write_summary(inputs.output_dir, summary)
    return summary
