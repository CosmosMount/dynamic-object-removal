from __future__ import annotations

import argparse
from pathlib import Path

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.io.manifest import write_method_manifest
from object_removal.stages.eval_stage import EvalInputs, run_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eval stage: compute mask/video metrics and write metrics_summary.{json,csv}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", required=True, help="Run directory (will write to run_dir/eval/)")
    p.add_argument("--part_label", default="")
    p.add_argument("--experiment_name", default="")
    p.add_argument("--davis_csv", default="")
    p.add_argument("--pred_mask_dir", default="")
    p.add_argument("--gt_mask_dir", default="")
    p.add_argument("--merge_gt_objects", action="store_true", default=True)
    p.add_argument("--no_merge_gt_objects", action="store_false", dest="merge_gt_objects")
    p.add_argument("--pred_video", default="")
    p.add_argument("--gt_video", default="")
    p.add_argument("--pred_frames_dir", default="")
    p.add_argument(
        "--source_frames_dir",
        default="",
        help="Original RGB frames (e.g. DAVIS JPEGImages); required for bg L1 and temporal metrics.",
    )
    p.add_argument(
        "--gt_frames_dir",
        default="",
        help="Deprecated: GT inpaint frames are no longer used for PSNR/SSIM.",
    )
    p.add_argument("--video_metric_impl", choices=["internal", "propainter"], default="internal")
    p.add_argument("--propainter_root", default="", help="Required if video_metric_impl=propainter")
    return p


def main() -> None:
    args = build_parser().parse_args()
    layout = RunLayout(Path(args.run_dir))
    ensure_layout_dirs(layout)

    inputs = EvalInputs(
        output_dir=layout.eval_dir,
        part_label=args.part_label,
        experiment_name=args.experiment_name,
        davis_csv=Path(args.davis_csv) if args.davis_csv else None,
        pred_mask_dir=Path(args.pred_mask_dir) if args.pred_mask_dir else None,
        gt_mask_dir=Path(args.gt_mask_dir) if args.gt_mask_dir else None,
        merge_gt_objects=bool(args.merge_gt_objects),
        pred_video=Path(args.pred_video) if args.pred_video else None,
        gt_video=Path(args.gt_video) if args.gt_video else None,
        pred_frames_dir=Path(args.pred_frames_dir) if args.pred_frames_dir else None,
        gt_frames_dir=Path(args.gt_frames_dir) if args.gt_frames_dir else None,
        source_frames_dir=Path(args.source_frames_dir) if args.source_frames_dir else None,
        video_metric_impl=args.video_metric_impl,
    )

    propainter_root = Path(args.propainter_root) if args.propainter_root else None
    summary = run_eval(inputs, propainter_root=propainter_root)

    write_method_manifest(layout.eval_dir, stage="eval", method="eval", params=vars(args))
    print(f"Wrote: {layout.eval_metrics_json}")
    print(f"Wrote: {layout.eval_metrics_csv}")
    print(
        f"mask_jm={summary.get('mask_jm')} mask_jr={summary.get('mask_jr')} "
        f"mask_score={summary.get('mask_score')} quality_score={summary.get('quality_score')}"
    )


if __name__ == "__main__":
    main()

