from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path

    @property
    def meta_dir(self) -> Path:
        return self.run_dir / "meta"

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "input"

    @property
    def input_frames_dir(self) -> Path:
        return self.input_dir / "frames"

    @property
    def input_video_path(self) -> Path:
        return self.input_dir / "video.mp4"

    @property
    def mask_dir(self) -> Path:
        return self.run_dir / "mask"

    @property
    def mask_init_dir(self) -> Path:
        return self.mask_dir / "init"

    @property
    def mask_init_masks_dir(self) -> Path:
        return self.mask_init_dir / "masks"

    @property
    def track_dir(self) -> Path:
        return self.run_dir / "track"

    @property
    def track_masks_binary_dir(self) -> Path:
        return self.track_dir / "masks_binary"

    @property
    def track_mask_vis_mp4(self) -> Path:
        return self.track_dir / "mask_vis.mp4"

    @property
    def track_masks_indexed_dir(self) -> Path:
        return self.track_dir / "masks_indexed"

    @property
    def inpaint_dir(self) -> Path:
        return self.run_dir / "inpaint"

    @property
    def inpaint_frames_dir(self) -> Path:
        return self.inpaint_dir / "frames"

    @property
    def inpaint_video_path(self) -> Path:
        return self.inpaint_dir / "video.mp4"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def eval_metrics_json(self) -> Path:
        return self.eval_dir / "metrics_summary.json"

    @property
    def eval_metrics_csv(self) -> Path:
        return self.eval_dir / "metrics_summary.csv"


def ensure_layout_dirs(layout: RunLayout) -> None:
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.meta_dir.mkdir(parents=True, exist_ok=True)
    layout.input_dir.mkdir(parents=True, exist_ok=True)
    layout.mask_init_dir.mkdir(parents=True, exist_ok=True)
    layout.mask_init_masks_dir.mkdir(parents=True, exist_ok=True)
    layout.track_dir.mkdir(parents=True, exist_ok=True)
    layout.inpaint_dir.mkdir(parents=True, exist_ok=True)
    layout.eval_dir.mkdir(parents=True, exist_ok=True)

