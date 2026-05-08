from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from object_removal.io.frames import read_bgr_frames
from object_removal.io.masks import to_binary_255, write_mask_u8


@dataclass(frozen=True)
class Params:
    dynamic_classes: List[int] = (0, 1, 2, 3, 5, 7)
    motion_threshold: float = 1.5
    lk_max_corners: int = 60
    mog2_history: int = 200
    mog2_threshold: int = 40
    min_blob_area: int = 800
    dilation_kernel: int = 15
    adaptive_dilation: bool = True


def _try_yolo(frames: List[np.ndarray], dynamic_classes: List[int]) -> Tuple[List[np.ndarray] | None, str | None]:
    import cv2

    try:
        from ultralytics import YOLO

        ckpt = Path("ckpts/yolo/yolov8n-seg.pt")
        model_path = str(ckpt) if ckpt.is_file() else "yolov8n-seg.pt"
        model = YOLO(model_path)
        height, width = frames[0].shape[:2]
        masks: List[np.ndarray] = []

        for i, frame in enumerate(frames):
            result = model(frame, classes=dynamic_classes, verbose=False)[0]
            mask = np.zeros((height, width), dtype=np.uint8)
            if result.masks is not None:
                for seg in result.masks.data:
                    seg_map = seg.cpu().numpy()
                    seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, (seg_map > 0.5).astype(np.uint8) * 255)
            masks.append(mask)
            if (i + 1) % 30 == 0:
                print(f"   YOLO: {i + 1}/{len(frames)}")

        return masks, "YOLOv8-Seg"
    except Exception as exc:
        print(f"   YOLO unavailable ({exc}) - using MOG2 fallback")
        return None, None


def _mog2(frames: List[np.ndarray], history: int, threshold: int, min_blob_area: int) -> Tuple[List[np.ndarray], str]:
    import cv2

    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=True)
    for frame in frames:
        fgbg.apply(frame)

    fgbg2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=True)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    masks: List[np.ndarray] = []

    for frame in frames:
        fg = fgbg2.apply(frame)
        fg[fg == 127] = 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(fg)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_blob_area:
                cv2.drawContours(clean, [cnt], -1, 255, -1)
        masks.append(clean)

    return masks, "MOG2 background subtraction"


def _extract_masks(frames: List[np.ndarray], p: Params) -> Tuple[List[np.ndarray], str]:
    print("[Mask] Extracting masks ...")
    masks, method = _try_yolo(frames, list(p.dynamic_classes))
    if masks is None:
        masks, method = _mog2(frames, p.mog2_history, p.mog2_threshold, p.min_blob_area)
    detected = sum(1 for m in masks if m.max() > 0)
    print(f"[Mask] {method} - detections in {detected}/{len(frames)} frames")
    return masks, method


def _filter_dynamic(frames: List[np.ndarray], masks: List[np.ndarray], p: Params) -> List[np.ndarray]:
    import cv2

    print("[Mask] Optical flow dynamic filter ...")
    lk = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    fp = dict(maxCorners=p.lk_max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)

    out: List[np.ndarray] = []
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        if mask.max() == 0 or i == 0:
            out.append(mask)
            continue

        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **fp)
        if pts is None or len(pts) < 3:
            out.append(mask)
            continue

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk)
        good_o = pts[st == 1]
        good_n = nxt[st == 1]
        mag = float(np.mean(np.linalg.norm(good_n - good_o, axis=1))) if len(good_o) else 0.0
        if mag >= p.motion_threshold:
            out.append(mask)
        else:
            out.append(np.zeros_like(mask))

    kept = sum(1 for m in out if m.max() > 0)
    print(f"[Mask] Dynamic frames: {kept}/{len(frames)}  (thr={p.motion_threshold}px)")
    return out


def _dilate_masks(masks: List[np.ndarray], p: Params) -> List[np.ndarray]:
    import cv2

    print(f"[Mask] Dilation kernel={p.dilation_kernel}px adaptive={p.adaptive_dilation} ...")
    k_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.dilation_kernel, p.dilation_kernel))
    k_ext = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.dilation_kernel * 2, p.dilation_kernel * 2))

    result: List[np.ndarray] = []
    for mask in masks:
        if mask.max() == 0:
            result.append(mask)
            continue

        if not p.adaptive_dilation:
            result.append(cv2.dilate(mask, k_body, iterations=2))
            continue

        ys = np.where(mask > 0)[0]
        if len(ys) == 0:
            result.append(mask)
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        margin = max(int((y_max - y_min) * 0.22), 20)

        top = np.zeros_like(mask)
        top[: y_min + margin, :] = mask[: y_min + margin, :]

        bottom = np.zeros_like(mask)
        bottom[y_max - margin :, :] = mask[y_max - margin :, :]

        mid = mask.copy()
        mid[: y_min + margin, :] = 0
        mid[y_max - margin :, :] = 0

        dilated = cv2.dilate(mid, k_body, iterations=2)
        dilated = np.maximum(dilated, cv2.dilate(top, k_ext, iterations=2))
        dilated = np.maximum(dilated, cv2.dilate(bottom, k_ext, iterations=2))
        result.append(dilated)

    print("[Mask] Done")
    return result


def run(*, frames_dir: Path, out_masks_dir: Path, params: Params | None = None, max_frames: int | None = None) -> dict:
    p = params or Params()
    frames = read_bgr_frames(frames_dir, max_frames=max_frames)
    raw_masks, seg_method = _extract_masks(frames, p)
    dynamic_masks = _filter_dynamic(frames, raw_masks, p)
    final_masks = _dilate_masks(dynamic_masks, p)

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(final_masks):
        write_mask_u8(out_masks_dir / f"{i:05d}.png", to_binary_255(m))

    return {"segmentation_method": seg_method, "num_frames": len(frames)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Baseline mask method (YOLO/MOG2 + flow filter + dilation)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()
    run(frames_dir=Path(args.frames_dir), out_masks_dir=Path(args.out_masks_dir), max_frames=args.max_frames)


if __name__ == "__main__":
    main()

