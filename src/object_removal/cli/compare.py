from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.io.mask_vis import export_mask_overlay_video
from object_removal.io.masks import init_masks_sufficient_for_track
from object_removal.io.pipeline_argv import (
    inpaint_argv_from_diffueraser_opts,
    inpaint_argv_from_propainter_opts,
    mask_argv_from_vggt_opts,
    xmem_argv_from_opts,
)
from object_removal.stages.inpaint_stage import run_inpaint_stage
from object_removal.stages.mask_stage import run_mask_stage
from object_removal.stages.track_stage import run_track_stage
from object_removal.stages.eval_stage import EvalInputs, run_eval, write_zero_eval_summary
from object_removal.io.compare_run import CompareRunContext, env_for, load_pipelines_yaml, resolve_compare_context


def _safe_float(v: Any):
    if v is None:
        return ""
    try:
        return float(v)
    except Exception:
        return ""


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _track_uses_init_masks_dir(track_method: str) -> bool:
    """Track methods that read init masks from mask/init/masks (not e.g. xmem's own init)."""
    return track_method in ("sam2", "sam3", "optflow", "identity")


def _load_metrics_row(path: Path, method_name: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": method_name,
        "mask_jm": _safe_float(data.get("mask_jm")),
        "mask_jr": _safe_float(data.get("mask_jr")),
        "mask_fm": _safe_float(data.get("mask_fm")),
        "mask_fr": _safe_float(data.get("mask_fr")),
        "mask_score": _safe_float(data.get("mask_score")),
        "bg_l1_mean": _safe_float(data.get("bg_l1_mean")),
        "temporal_warp_error_mean": _safe_float(data.get("temporal_warp_error_mean")),
        "temporal_warp_error_hole_mean": _safe_float(data.get("temporal_warp_error_hole_mean")),
        "laplacian_var_mean": _safe_float(data.get("laplacian_var_mean")),
    }


def _write_combined(rows: List[Dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    import csv

    columns = [
        "method",
        "mask_jm",
        "mask_jr",
        "mask_fm",
        "mask_fr",
        "mask_score",
        "bg_l1_mean",
        "temporal_warp_error_mean",
        "temporal_warp_error_hole_mean",
        "laplacian_var_mean",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pipeline_registry(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load pipeline registry only from YAML `pipelines:`."""
    return load_pipelines_yaml(config_path)


def _run_stage_via_conda_run(*, conda_exe: str, env_name: str, module: str, argv: List[str], cwd: Path) -> None:
    cmd = [conda_exe, "run", "-n", env_name, "python", "-m", module, *argv]
    env = os.environ.copy()
    src_root = (cwd / "src").resolve()
    if src_root.is_dir():
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src_root) if not old else f"{src_root}{os.pathsep}{old}"
    # conda run often drops /usr/bin from PATH; keep system tools (e.g. ffmpeg) visible for post-steps.
    if os.name == "posix":
        prefix = "/usr/bin:/usr/local/bin:/bin"
        old_path = env.get("PATH", "")
        env["PATH"] = f"{prefix}{os.pathsep}{old_path}" if old_path else prefix
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def _list_conda_env_names(conda_exe: str) -> List[str]:
    cmd = [conda_exe, "env", "list", "--json"]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(p.stdout)
    envs = data.get("envs", [])
    names: List[str] = []
    for e in envs:
        try:
            names.append(Path(str(e)).name)
        except Exception:
            continue
    return sorted(set(names))


def _require_envs(conda_exe: str, required: List[str]) -> None:
    required = sorted({e for e in required if str(e).strip() != ""})
    if not required:
        return
    available = set(_list_conda_env_names(conda_exe))
    missing = [e for e in required if e not in available]
    if not missing:
        return
    msg = [
        "Missing conda env(s) required by selected pipelines:",
        "  - " + "\n  - ".join(missing),
        "",
        "Create them and install this repo inside each env:",
        "  conda create -n <env> python=3.10",
        "  conda activate <env>",
        "  pip install -e .",
    ]
    raise RuntimeError("\n".join(msg))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run multiple pipelines on the same task and aggregate metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Default: read run settings from configs/compare.yaml (task, pipelines, out_root, …).\n"
            "Omit out_root in YAML to use outputs/compare/<SEQ> from task davis:SEQ.\n"
            "CLI options override YAML when provided.\n"
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --config configs/my_compare.yaml\n"
            "  %(prog)s --task davis:blackswan   # override task only"
        ),
    )
    p.add_argument(
        "--config",
        default="configs/compare.yaml",
        help="Run config YAML (task, davis_root, pipelines, out_root, overwrite, …). File may be absent if you pass all required CLI flags.",
    )
    p.add_argument("--task", default=None, help="Task spec: davis:SEQ. Overrides YAML.")
    p.add_argument("--davis_root", default=None, help="Overrides YAML (default in YAML or data/DAVIS).")
    p.add_argument(
        "--pipelines_config",
        default=None,
        help="Pipeline definitions YAML (pipelines: …). Overrides YAML.",
    )
    mx = p.add_mutually_exclusive_group()
    mx.add_argument(
        "--all",
        action="store_true",
        help="Run every pipeline id in the registry (overrides YAML pipelines list).",
    )
    mx.add_argument(
        "--pipelines",
        nargs="*",
        metavar="ID",
        default=None,
        help="Pipeline id(s); overrides YAML when given (use one or more ids).",
    )
    p.add_argument("--out_root", default=None, help="Output root; overrides YAML. If neither set, outputs/compare/<seq>.")
    p.add_argument("--part_label", default=None, help="Overrides YAML (default empty).")
    p.add_argument("--overwrite", action="store_true", default=False, help="Force overwrite (OR with YAML overwrite).")
    p.add_argument("--conda_exe", default=None, help="Overrides YAML (default conda).")
    p.add_argument("--env_map", default=None, help="Overrides YAML path to env map JSON.")
    p.add_argument(
        "--env_policy",
        default=None,
        choices=["auto", "force_multi", "force_single"],
        help="Overrides YAML env_policy.",
    )
    p.add_argument(
        "--export-mask-vis",
        action="store_true",
        default=False,
        help="After track, write track/mask_vis.mp4 (also set export_mask_vis in compare YAML).",
    )
    p.add_argument(
        "--only-stage",
        default=None,
        choices=["mask", "track", "inpaint", "eval"],
        help="Run only this stage (requires exactly one pipeline). Overrides YAML only_stage when set.",
    )
    return p


def _collect_required_envs(ctx: CompareRunContext) -> List[str]:
    """Conda env names required for the selected only_stage (or full pipeline)."""
    req: List[str] = []
    st = ctx.only_stage
    for name in ctx.pipeline_ids:
        spec = ctx.registry[name]
        if st is None or st == "mask":
            req.append(env_for(ctx.env_map, stage="mask", method=str(spec["mask"])))
        if st is None or st == "track":
            req.append(env_for(ctx.env_map, stage="track", method=str(spec["track"])))
        if st is None or st == "inpaint":
            req.append(env_for(ctx.env_map, stage="inpaint", method=str(spec["inpaint"])))
        if st is None or st == "eval":
            req.append(env_for(ctx.env_map, stage="eval", method="internal"))
    return req


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(os.getcwd()).resolve()

    compare_cfg_path = Path(args.config)
    if not compare_cfg_path.is_absolute():
        compare_cfg_path = (repo_root / compare_cfg_path).resolve()
    try:
        ctx = resolve_compare_context(
            repo_root=repo_root,
            compare_cfg_path=compare_cfg_path,
            task=args.task,
            davis_root=args.davis_root,
            pipelines_config=args.pipelines_config,
            pipelines_arg=args.pipelines,
            run_all_pipelines=bool(args.all),
            out_root=args.out_root,
            part_label=args.part_label,
            overwrite_cli=bool(args.overwrite),
            conda_exe=args.conda_exe,
            env_map_path=args.env_map,
            env_policy=args.env_policy,
            export_mask_vis_cli=bool(args.export_mask_vis),
            only_stage_cli=args.only_stage,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = ctx.cfg
    task = ctx.task
    davis_root = ctx.davis_root
    pipelines_config_path = ctx.pipelines_config_path
    env_map = ctx.env_map
    conda_exe = ctx.conda_exe
    env_policy = ctx.env_policy
    overwrite = ctx.overwrite
    part_label = ctx.part_label
    export_mask_vis = ctx.export_mask_vis
    mask_vis_fps = ctx.mask_vis_fps
    mask_vis_alpha = ctx.mask_vis_alpha
    registry = ctx.registry
    pipeline_ids = ctx.pipeline_ids
    out_root = ctx.out_root
    runs_root = ctx.runs_root
    summary_root = ctx.summary_root
    meta_root = ctx.meta_root
    frames_dir = ctx.frames_dir
    gt_mask_dir = ctx.gt_mask_dir
    gt_frames_dir = ctx.gt_frames_dir
    only = ctx.only_stage

    if env_policy != "force_single":
        _require_envs(conda_exe, _collect_required_envs(ctx))

    meta_root.mkdir(parents=True, exist_ok=True)
    seq = task.split(":", 1)[1]
    task_meta = {"type": "davis", "seq": seq, "davis_root": str(davis_root)}
    (meta_root / "task.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")
    (meta_root / "compare_run.json").write_text(
        json.dumps(
            {
                "compare_config": str(compare_cfg_path),
                "compare_config_loaded": compare_cfg_path.is_file(),
                "task": task,
                "davis_root": str(davis_root),
                "pipelines_config": str(pipelines_config_path),
                "pipeline_ids": pipeline_ids,
                "out_root": str(out_root),
                "overwrite": overwrite,
                "part_label": part_label,
                "conda_exe": conda_exe,
                "env_map": str(ctx.env_map_path),
                "env_policy": env_policy,
                "only_stage": only,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_root / "pipelines.json").write_text(
        json.dumps(
            {
                "pipelines_config": str(pipelines_config_path),
                "pipeline_ids": pipeline_ids,
                "selected_specs": {k: registry[k] for k in pipeline_ids},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = []

    for name in pipeline_ids:
        spec = registry[name]

        run_dir = runs_root / name
        layout = RunLayout(run_dir)
        ensure_layout_dirs(layout)

        # Stages
        mask_method = str(spec["mask"])
        track_method = str(spec["track"])
        inpaint_method = str(spec["inpaint"])
        vggt4d_opts = spec.get("vggt4d") if mask_method == "vggt4d" and isinstance(spec.get("vggt4d"), dict) else None
        if mask_method == "vggt_framewise" and isinstance(spec.get("vggt4d"), dict):
            vggt4d_opts = spec.get("vggt4d")
        if inpaint_method == "diffueraser":
            raw_de = spec.get("diffueraser")
            diffueraser_opts = dict(raw_de) if isinstance(raw_de, dict) else {}
        else:
            diffueraser_opts = None
        propainter_opts = (
            spec.get("propainter") if inpaint_method == "propainter" and isinstance(spec.get("propainter"), dict) else None
        )
        sam3_opts: Optional[Dict[str, Any]] = None
        if track_method == "sam3":
            base_s3 = {
                "checkpoint": "ckpts/sam3/sam3.pt",
                "two_stage_anchor_idx": "auto",
                "two_stage_auto_samples": 7,
                "two_stage_auto_max_fg_frac": 0.92,
                "two_stage_auto_min_fg_frac": 0.00008,
                "two_stage_auto_min_fg_pixels": 64,
                "score_thresh": 0.0,
            }
            raw_s3 = spec.get("sam3")
            if isinstance(raw_s3, dict):
                base_s3.update(raw_s3)
            sam3_opts = base_s3

        xmem_opts: Optional[Dict[str, Any]] = None
        if track_method == "xmem":
            raw_xmem = spec.get("xmem")
            xmem_opts = dict(raw_xmem) if isinstance(raw_xmem, dict) else None

        mask_env = env_for(env_map, stage="mask", method=mask_method)
        track_env = env_for(env_map, stage="track", method=track_method)
        inpaint_env = env_for(env_map, stage="inpaint", method=inpaint_method)

        if only == "eval":
            track_masks_dir = layout.track_masks_binary_dir
            inpaint_frames_dir = layout.inpaint_frames_dir
            summary = run_eval(
                EvalInputs(
                    output_dir=layout.eval_dir,
                    part_label=part_label,
                    experiment_name=name,
                    pred_mask_dir=track_masks_dir,
                    gt_mask_dir=gt_mask_dir,
                    pred_frames_dir=inpaint_frames_dir,
                    gt_frames_dir=None,
                    source_frames_dir=frames_dir,
                    merge_gt_objects=True,
                    video_metric_impl="internal",
                )
            )
            rows.append(_load_metrics_row(layout.eval_metrics_json, method_name=name))
            print(
                f"[compare] done: {name} (only_stage=eval) mask_score={summary.get('mask_score')}"
            )
            continue

        if env_policy == "force_single":
            if only is None or only == "mask":
                init_masks_dir = run_mask_stage(
                    run_dir=run_dir,
                    frames_dir=frames_dir,
                    method=mask_method,
                    overwrite=overwrite,
                    vggt4d_options=vggt4d_opts if mask_method in ("vggt4d", "vggt_framewise") else None,
                    repo_root=repo_root,
                )
            else:
                init_masks_dir = layout.mask_init_masks_dir

            if only == "mask":
                print(f"[compare] only_stage=mask done: {name}")
                continue

            if _track_uses_init_masks_dir(track_method) and not init_masks_sufficient_for_track(
                frames_dir, init_masks_dir, track_method
            ):
                print(f"[compare] skip track/inpaint: no init mask on first frame ({name})")
                summary = write_zero_eval_summary(
                    layout.eval_dir,
                    experiment_name=name,
                    part_label=part_label,
                )
                rows.append(_load_metrics_row(layout.eval_metrics_json, method_name=name))
                print(
                    f"[compare] done: {name} (skipped, zeros) mask_jm={summary.get('mask_jm')} "
                    f"mask_score={summary.get('mask_score')}"
                )
                continue

            if only is None or only == "track":
                track_masks_dir = run_track_stage(
                    run_dir=run_dir,
                    frames_dir=frames_dir,
                    in_masks_dir=init_masks_dir,
                    method=track_method,
                    overwrite=overwrite,
                    sam3_options=sam3_opts if track_method == "sam3" else None,
                    xmem_options=xmem_opts if track_method == "xmem" else None,
                    repo_root=repo_root,
                )
            else:
                track_masks_dir = layout.track_masks_binary_dir

            if only == "track":
                print(f"[compare] only_stage=track done: {name}")
                continue

            if only is None or only == "inpaint":
                inpaint_frames_dir = run_inpaint_stage(
                    run_dir=run_dir,
                    frames_dir=frames_dir,
                    masks_dir=track_masks_dir,
                    method=inpaint_method,
                    overwrite=overwrite,
                    diffueraser_options=diffueraser_opts,
                    propainter_options=propainter_opts,
                )
            else:
                inpaint_frames_dir = layout.inpaint_frames_dir

            if only == "inpaint":
                print(f"[compare] only_stage=inpaint done: {name}")
                continue
        else:
            if only is None or only == "mask":
                if mask_env:
                    mask_argv = [
                        "--run_dir",
                        str(run_dir),
                        "--frames_dir",
                        str(frames_dir),
                        "--method",
                        mask_method,
                        "--repo_root",
                        str(repo_root),
                        *([] if not overwrite else ["--overwrite"]),
                    ]
                    if vggt4d_opts:
                        mask_argv += mask_argv_from_vggt_opts(vggt4d_opts, mask_method=mask_method)
                    _run_stage_via_conda_run(
                        conda_exe=conda_exe,
                        env_name=mask_env,
                        module="object_removal.cli.mask",
                        argv=mask_argv,
                        cwd=repo_root,
                    )
                else:
                    run_mask_stage(
                        run_dir=run_dir,
                        frames_dir=frames_dir,
                        method=mask_method,
                        overwrite=overwrite,
                        vggt4d_options=vggt4d_opts if mask_method in ("vggt4d", "vggt_framewise") else None,
                        repo_root=repo_root,
                    )
            init_masks_dir = layout.mask_init_masks_dir

            if only == "mask":
                print(f"[compare] only_stage=mask done: {name}")
                continue

            if _track_uses_init_masks_dir(track_method) and not init_masks_sufficient_for_track(
                frames_dir, init_masks_dir, track_method
            ):
                print(f"[compare] skip track/inpaint: no init mask on first frame ({name})")
                summary = write_zero_eval_summary(
                    layout.eval_dir,
                    experiment_name=name,
                    part_label=part_label,
                )
                rows.append(_load_metrics_row(layout.eval_metrics_json, method_name=name))
                print(
                    f"[compare] done: {name} (skipped, zeros) mask_jm={summary.get('mask_jm')} "
                    f"mask_score={summary.get('mask_score')}"
                )
                continue

            if only is None or only == "track":
                if track_env:
                    track_argv = [
                        "--run_dir",
                        str(run_dir),
                        "--frames_dir",
                        str(frames_dir),
                        "--in_masks_dir",
                        str(init_masks_dir),
                        "--method",
                        track_method,
                        "--repo_root",
                        str(repo_root),
                        *([] if not overwrite else ["--overwrite"]),
                    ]
                    if track_method == "sam3" and sam3_opts:
                        track_argv += [
                            "--sam3-checkpoint",
                            str(sam3_opts.get("checkpoint", "ckpts/sam3/sam3.pt")),
                            "--sam3-two-stage-anchor-idx",
                            str(sam3_opts.get("two_stage_anchor_idx", "auto")),
                            "--sam3-two-stage-auto-samples",
                            str(int(sam3_opts.get("two_stage_auto_samples", 7))),
                            "--sam3-two-stage-auto-max-fg-frac",
                            str(float(sam3_opts.get("two_stage_auto_max_fg_frac", 0.92))),
                            "--sam3-two-stage-auto-min-fg-frac",
                            str(float(sam3_opts.get("two_stage_auto_min_fg_frac", 0.00008))),
                            "--sam3-two-stage-auto-min-fg-pixels",
                            str(int(sam3_opts.get("two_stage_auto_min_fg_pixels", 64))),
                            "--sam3-score-thresh",
                            str(float(sam3_opts.get("score_thresh", 0.0))),
                        ]
                    if track_method == "xmem" and xmem_opts:
                        track_argv += xmem_argv_from_opts(xmem_opts)
                    _run_stage_via_conda_run(
                        conda_exe=conda_exe,
                        env_name=track_env,
                        module="object_removal.cli.track",
                        argv=track_argv,
                        cwd=repo_root,
                    )
                else:
                    run_track_stage(
                        run_dir=run_dir,
                        frames_dir=frames_dir,
                        in_masks_dir=init_masks_dir,
                        method=track_method,
                        overwrite=overwrite,
                        sam3_options=sam3_opts if track_method == "sam3" else None,
                        xmem_options=xmem_opts if track_method == "xmem" else None,
                        repo_root=repo_root,
                    )
                track_masks_dir = layout.track_masks_binary_dir
            else:
                track_masks_dir = layout.track_masks_binary_dir

            if only == "track":
                print(f"[compare] only_stage=track done: {name}")
                continue

            if only is None or only == "inpaint":
                if inpaint_env:
                    inpaint_argv = [
                        "--run_dir",
                        str(run_dir),
                        "--frames_dir",
                        str(frames_dir),
                        "--masks_dir",
                        str(track_masks_dir),
                        "--method",
                        inpaint_method,
                        *([] if not overwrite else ["--overwrite"]),
                    ]
                    if inpaint_method == "diffueraser":
                        inpaint_argv += inpaint_argv_from_diffueraser_opts(diffueraser_opts)
                    if propainter_opts:
                        inpaint_argv += inpaint_argv_from_propainter_opts(propainter_opts)
                    _run_stage_via_conda_run(
                        conda_exe=conda_exe,
                        env_name=inpaint_env,
                        module="object_removal.cli.inpaint",
                        argv=inpaint_argv,
                        cwd=repo_root,
                    )
                else:
                    run_inpaint_stage(
                        run_dir=run_dir,
                        frames_dir=frames_dir,
                        masks_dir=track_masks_dir,
                        method=inpaint_method,
                        overwrite=overwrite,
                        diffueraser_options=diffueraser_opts,
                        propainter_options=propainter_opts,
                    )
                inpaint_frames_dir = layout.inpaint_frames_dir
            else:
                inpaint_frames_dir = layout.inpaint_frames_dir

            if only == "inpaint":
                print(f"[compare] only_stage=inpaint done: {name}")
                continue

        if export_mask_vis and only is None:
            vis_path = layout.track_mask_vis_mp4
            n_masks = len(list(track_masks_dir.glob("*.png")))
            if n_masks > 0:
                if export_mask_overlay_video(
                    frames_dir,
                    track_masks_dir,
                    vis_path,
                    fps=mask_vis_fps,
                    alpha=mask_vis_alpha,
                ):
                    print(f"[compare] mask_vis: {vis_path}")
                else:
                    print(f"[compare] mask_vis: failed ({vis_path})")
            else:
                print(f"[compare] mask_vis: skipped (no PNG under {track_masks_dir})")

        # Eval (full pipeline)
        summary = run_eval(
            EvalInputs(
                output_dir=layout.eval_dir,
                part_label=part_label,
                experiment_name=name,
                pred_mask_dir=track_masks_dir,
                gt_mask_dir=gt_mask_dir,
                pred_frames_dir=inpaint_frames_dir,
                gt_frames_dir=None,
                source_frames_dir=frames_dir,
                merge_gt_objects=True,
                video_metric_impl="internal",
            )
        )

        rows.append(_load_metrics_row(layout.eval_metrics_json, method_name=name))
        print(
            f"[compare] done: {name} (mask_jm={summary.get('mask_jm')} mask_fm={summary.get('mask_fm')} "
            f"mask_fr={summary.get('mask_fr')} mask_score={summary.get('mask_score')})"
        )

    rows.sort(key=lambda r: str(r.get("method", "")))
    _write_combined(rows, summary_root / "combined.csv", summary_root / "combined.md")
    print(f"Wrote: {summary_root / 'combined.csv'}")
    print(f"Wrote: {summary_root / 'combined.md'}")


if __name__ == "__main__":
    main()

