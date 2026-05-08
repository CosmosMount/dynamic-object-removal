from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from object_removal.io.frames import read_bgr_frames
from object_removal.io.masks import read_mask_u8, to_binary_255, write_mask_u8


@dataclass(frozen=True)
class Params:
    yolo_model: Path
    yolo_conf: float = 0.25
    dynamic_classes: List[int] = (0, 1, 2, 3, 5, 7)
    motion_threshold: float = 1.5
    dilation_kernel: int = 15


def _compute_flow(prev_gray, curr_gray, mask):
    import cv2

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(
        maxCorners=60,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
    if pts is None or len(pts) < 3:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk_params)
    good_prev = pts[status == 1]
    good_next = next_pts[status == 1]
    if len(good_prev) < 3:
        return None
    return good_next - good_prev


def _dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    import cv2

    if mask.max() == 0:
        return mask
    k_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, k_body, iterations=2)


def run(*, frames_dir: Path, init_mask_path: Path, out_masks_dir: Path, params: Params) -> dict:
    import cv2
    from ultralytics import YOLO

    frames = read_bgr_frames(frames_dir)
    if not frames:
        raise ValueError(f"No frames under: {frames_dir}")
    h, w = frames[0].shape[:2]

    init = read_mask_u8(init_mask_path)
    if init.shape != (h, w):
        init = cv2.resize(init, (w, h), interpolation=cv2.INTER_NEAREST)

    model = YOLO(str(params.yolo_model))

    raw_masks: List[np.ndarray] = []
    for frame in frames:
        res = model(frame, classes=list(params.dynamic_classes), conf=float(params.yolo_conf), verbose=False)[0]
        mask = np.zeros((h, w), dtype=np.uint8)
        if res.masks is not None:
            for seg in res.masks.data:
                seg_map = seg.cpu().numpy()
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, (seg_map > 0.5).astype(np.uint8) * 255)
        raw_masks.append(mask)

    out_masks: List[np.ndarray] = []
    for i, mask in enumerate(raw_masks):
        if mask.max() == 0 or i == 0:
            out_masks.append(_dilate(mask, params.dilation_kernel))
            continue

        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = _compute_flow(prev_gray, curr_gray, mask)
        if flow is None:
            out_masks.append(_dilate(mask, params.dilation_kernel))
            continue
        mag = float(np.mean(np.linalg.norm(flow, axis=1)))
        dynamic = mask if mag >= float(params.motion_threshold) else np.zeros_like(mask)
        out_masks.append(_dilate(dynamic, params.dilation_kernel))

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(out_masks):
        write_mask_u8(out_masks_dir / f"{i:05d}.png", to_binary_255(m))
    return {"num_frames": len(out_masks)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="YOLO + optical-flow mask propagation (binary masks)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--init_mask", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--yolo_model", required=True)
    args = ap.parse_args()
    run(
        frames_dir=Path(args.frames_dir),
        init_mask_path=Path(args.init_mask),
        out_masks_dir=Path(args.out_masks_dir),
        params=Params(yolo_model=Path(args.yolo_model)),
    )


if __name__ == "__main__":
    main()

