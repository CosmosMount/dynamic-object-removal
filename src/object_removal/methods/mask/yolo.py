from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class Params:
    model_path: Path
    conf: float = 0.25
    max_init_objects: int = 4
    classes: List[int] = (0, 1, 2, 3, 5, 7)


def _safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _align_mask_to_frame(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    if mask.shape == (h, w):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def run(*, first_frame: Path, out_mask: Path, params: Params) -> dict:
    import cv2
    from ultralytics import YOLO

    frame = cv2.imread(str(first_frame))
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {first_frame}")

    h, w = frame.shape[:2]
    model = YOLO(str(params.model_path))
    res = model(frame, classes=list(params.classes), conf=float(params.conf), verbose=False)[0]

    indexed_mask = np.zeros((h, w), dtype=np.uint8)
    max_init_objects = max(1, min(int(params.max_init_objects), 255))

    if res.masks is not None and len(res.masks.data) > 0:
        masks = res.masks.data.cpu().numpy() > 0.5
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        order = np.argsort(-areas)
        kept = 0
        for idx in order:
            if kept >= max_init_objects:
                break
            m = _align_mask_to_frame(masks[idx], h, w)
            if m.sum() == 0:
                continue
            write_region = np.logical_and(m, indexed_mask == 0)
            if write_region.sum() == 0:
                continue
            indexed_mask[write_region] = kept + 1
            kept += 1
    elif res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order = np.argsort(-areas)
        kept = 0
        for idx in order:
            if kept >= max_init_objects:
                break
            x1, y1, x2, y2 = _safe_box(*boxes_xyxy[idx], w, h)
            region = np.zeros((h, w), dtype=bool)
            region[y1:y2, x1:x2] = True
            write_region = np.logical_and(region, indexed_mask == 0)
            if write_region.sum() == 0:
                continue
            indexed_mask[write_region] = kept + 1
            kept += 1

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_mask), indexed_mask.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to write mask: {out_mask}")
    return {"out_mask": str(out_mask), "num_objects": int(indexed_mask.max())}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="YOLO first-frame init mask (indexed)")
    ap.add_argument("--first_frame", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_init_objects", type=int, default=4)
    ap.add_argument("--classes", default="0,1,2,3,5,7")
    args = ap.parse_args()

    params = Params(
        model_path=Path(args.model_path),
        conf=float(args.conf),
        max_init_objects=int(args.max_init_objects),
        classes=[int(x) for x in str(args.classes).split(",") if x.strip() != ""],
    )
    run(first_frame=Path(args.first_frame), out_mask=Path(args.out_mask), params=params)


if __name__ == "__main__":
    main()

