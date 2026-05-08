from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from object_removal.io.layout import RunLayout, ensure_layout_dirs
from object_removal.stages.inpaint_stage import run_inpaint_stage
from object_removal.stages.mask_stage import run_mask_stage
from object_removal.stages.track_stage import run_track_stage
from object_removal.stages.eval_stage import EvalInputs, run_eval


def _count_rgb_frames(frames_dir: Path) -> int:
    return sum(
        1
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _resolve_davis_paths(davis_root: Path, seq: str) -> Tuple[Path, Path]:
    frames_dir = davis_root / "JPEGImages" / "480p" / seq
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"DAVIS frames dir not found: {frames_dir}")
    n_frames = _count_rgb_frames(frames_dir)
    if n_frames == 0:
        raise FileNotFoundError(
            f"No JPEG/PNG frames under {frames_dir} (folder exists but is empty). "
            "Unpack DAVIS JPEGImages for this sequence."
        )

    gt1 = davis_root / "Annotations_unsupervised" / "480p" / seq
    gt2 = davis_root / "Annotations" / "480p" / seq
    gt_dir = gt1 if gt1.is_dir() else gt2
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"DAVIS GT mask dir not found: tried {gt1} and {gt2}")
    return frames_dir, gt_dir


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


def _load_metrics_row(path: Path, method_name: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": method_name,
        "part_label": data.get("part_label", ""),
        "experiment_name": data.get("experiment_name", ""),
        "mask_jm": _safe_float(data.get("mask_jm")),
        "mask_jr": _safe_float(data.get("mask_jr")),
        "video_psnr": _safe_float(data.get("video_psnr")),
        "video_ssim": _safe_float(data.get("video_ssim")),
        "mask_source": data.get("mask_source", ""),
        "video_source": data.get("video_source", ""),
        "metrics_path": str(path),
    }


def _write_combined(rows: List[Dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    import csv

    columns = [
        "method",
        "part_label",
        "experiment_name",
        "mask_jm",
        "mask_jr",
        "video_psnr",
        "video_ssim",
        "mask_source",
        "video_source",
        "metrics_path",
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


# 内置默认；`configs/pipelines.yaml` 中同名会覆盖，亦可新增 id。
_PIPELINES_BUILTIN: Dict[str, Dict[str, Union[str, Dict[str, Any]]]] = {
    # clean-slate MVP
    "baseline": {
        "mask": "baseline_yolo_motion",
        "track": "identity",
        "inpaint": "baseline_handcrafted",
    }
    ,
    "yolosam2": {
        "mask": "yolo_first",
        "track": "sam2",
        "inpaint": "propainter",
    },
    "yoloopt": {
        "mask": "yolo_first",
        "track": "optflow",
        "inpaint": "propainter",
    },
    "trackanything_diffueraser": {
        "mask": "yolo_first",
        "track": "trackanything",
        "inpaint": "diffueraser",
    },
    "vggt_trackanything_diffueraser": {
        "mask": "vggt4d",
        "track": "trackanything",
        "inpaint": "diffueraser",
    },
    "vggt4d": {
        "mask": "vggt_framewise",
        "track": "identity",
        "inpaint": "propainter",
    },
    "vggt_only_diffueraser": {
        "mask": "vggt_framewise",
        "track": "identity",
        "inpaint": "diffueraser",
    },
    "vggt4dsam3": {
        "mask": "vggt4d",
        "track": "sam3",
        "inpaint": "propainter",
        "sam3": {"two_stage_anchor_idx": "auto", "two_stage_auto_samples": 7, "score_thresh": 0.0},
    },
    "vggt4dsam3_diffueraser": {
        "mask": "vggt4d",
        "track": "sam3",
        "inpaint": "diffueraser",
        "sam3": {"two_stage_anchor_idx": "auto", "two_stage_auto_samples": 7, "score_thresh": 0.0},
    },
}


def _load_pipelines_yaml(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for --pipelines_config. Install: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pipelines YAML (expected mapping at root): {path}")
    pls = data.get("pipelines")
    if not isinstance(pls, dict) or not pls:
        raise ValueError(f"Invalid pipelines YAML (missing non-empty `pipelines:`): {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for pid, spec in pls.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid pipeline spec for {pid!r} in {path}")
        for k in ("mask", "track", "inpaint"):
            if k not in spec or not isinstance(spec[k], str) or not spec[k].strip():
                raise ValueError(f"Pipeline {pid!r} missing string field {k!r} in {path}")
        if "sam3" in spec and spec["sam3"] is not None and not isinstance(spec["sam3"], dict):
            raise ValueError(f"Pipeline {pid!r} field `sam3` must be a mapping or omitted in {path}")
        if "vggt4d" in spec and spec["vggt4d"] is not None and not isinstance(spec["vggt4d"], dict):
            raise ValueError(f"Pipeline {pid!r} field `vggt4d` must be a mapping or omitted in {path}")
        out[str(pid)] = dict(spec)
    return out


def build_pipeline_registry(config_path: Path) -> Dict[str, Dict[str, Union[str, Dict[str, Any]]]]:
    """Deep-copy builtin registry, then apply YAML `pipelines:` (override / add)."""
    reg: Dict[str, Dict[str, Union[str, Dict[str, Any]]]] = copy.deepcopy(_PIPELINES_BUILTIN)
    reg.update(_load_pipelines_yaml(config_path))
    return reg


def _load_env_map(path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid env map (expected dict): {path}")
    out: Dict[str, Dict[str, str]] = {}
    for stage in ("mask", "track", "inpaint", "eval"):
        v = data.get(stage, {})
        if not isinstance(v, dict):
            raise ValueError(f"Invalid env map for stage={stage}: {path}")
        out[stage] = {str(k): str(vv) for k, vv in v.items()}
    return out


def _env_for(env_map: Dict[str, Dict[str, str]], *, stage: str, method: str) -> str:
    stage_map = env_map.get(stage, {})
    if stage == "eval":
        return str(stage_map.get(method, stage_map.get("default", "")) or "")
    return str(stage_map.get(method, "") or "")


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
            "Examples:\n"
            "  %(prog)s --task davis:bmx-trees --out_root out/run1 --pipelines_config configs/pipelines.yaml "
            "--pipelines vggt_sam3_propainter\n"
            "  %(prog)s --task davis:bmx-trees --out_root out/run1 --pipelines_config configs/pipelines.yaml --all"
        ),
    )
    p.add_argument("--task", required=True, help="Task spec: davis:SEQ (MVP) or video:/path (future)")
    p.add_argument("--davis_root", default="data/DAVIS")
    p.add_argument(
        "--pipelines_config",
        default="configs/pipelines.yaml",
        help="YAML file: top-level `pipelines:` maps id -> {mask, track, inpaint, ...}. Merged over built-in defaults.",
    )
    mx = p.add_mutually_exclusive_group()
    mx.add_argument(
        "--all",
        action="store_true",
        help="Run every pipeline id present in the registry (builtin + YAML merge), sorted by id.",
    )
    mx.add_argument(
        "--pipelines",
        nargs="+",
        metavar="ID",
        help="Explicit pipeline id(s) from the registry (default: baseline, if present).",
    )
    p.add_argument("--out_root", required=True, help="Comparison output root")
    p.add_argument("--part_label", default="")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--conda_exe", default="conda", help="Conda executable used by `conda run`.")
    p.add_argument("--env_map", default="configs/env_map.json", help="JSON file mapping method->conda env.")
    p.add_argument(
        "--env_policy",
        default="auto",
        choices=["auto", "force_multi", "force_single"],
        help="auto/force_multi: use conda run when env is specified; force_single: run in current env only.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    env_policy = str(args.env_policy)
    repo_root = Path(os.getcwd()).resolve()
    env_map = _load_env_map((repo_root / str(args.env_map)).resolve())

    cfg_arg = Path(str(args.pipelines_config))
    pipelines_config_path = cfg_arg if cfg_arg.is_absolute() else (repo_root / cfg_arg).resolve()
    registry = build_pipeline_registry(pipelines_config_path)

    if bool(args.all):
        pipeline_ids = sorted(registry.keys())
    elif args.pipelines is not None:
        pipeline_ids = list(args.pipelines)
    else:
        pipeline_ids = ["baseline"] if "baseline" in registry else [sorted(registry.keys())[0]]

    for pid in pipeline_ids:
        if pid not in registry:
            raise ValueError(
                f"Unknown pipeline id {pid!r}. Known: {sorted(registry.keys())} "
                f"(config: {pipelines_config_path})"
            )

    if env_policy != "force_single":
        required_envs: List[str] = []
        for name in pipeline_ids:
            spec = registry.get(name)
            if spec is None:
                continue
            required_envs.append(_env_for(env_map, stage="mask", method=str(spec["mask"])))
            required_envs.append(_env_for(env_map, stage="track", method=str(spec["track"])))
            required_envs.append(_env_for(env_map, stage="inpaint", method=str(spec["inpaint"])))
        _require_envs(str(args.conda_exe), required_envs)

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    summary_root = out_root / "summary"
    meta_root = out_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    task = str(args.task)
    if task.startswith("davis:"):
        seq = task.split(":", 1)[1]
        frames_dir, gt_mask_dir = _resolve_davis_paths(Path(args.davis_root), seq)
        gt_frames_dir = frames_dir
        task_meta = {"type": "davis", "seq": seq, "davis_root": args.davis_root}
    else:
        raise ValueError("Only davis:SEQ is supported in MVP compare runner.")

    (meta_root / "task.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")
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
        sam3_opts: Optional[Dict[str, Any]] = None
        if track_method == "sam3":
            base_s3 = {
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

        mask_env = _env_for(env_map, stage="mask", method=mask_method)
        track_env = _env_for(env_map, stage="track", method=track_method)
        inpaint_env = _env_for(env_map, stage="inpaint", method=inpaint_method)

        if env_policy == "force_single":
            init_masks_dir = run_mask_stage(
                run_dir=run_dir,
                frames_dir=frames_dir,
                method=mask_method,
                overwrite=bool(args.overwrite),
                vggt4d_options=vggt4d_opts if mask_method in ("vggt4d", "vggt_framewise") else None,
                repo_root=repo_root,
            )
            track_masks_dir = run_track_stage(
                run_dir=run_dir,
                frames_dir=frames_dir,
                in_masks_dir=init_masks_dir,
                method=track_method,
                overwrite=bool(args.overwrite),
                sam3_options=sam3_opts if track_method == "sam3" else None,
                repo_root=repo_root,
            )
            inpaint_frames_dir = run_inpaint_stage(
                run_dir=run_dir,
                frames_dir=frames_dir,
                masks_dir=track_masks_dir,
                method=inpaint_method,
                overwrite=bool(args.overwrite),
            )
        else:
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
                    *([] if not args.overwrite else ["--overwrite"]),
                ]
                if vggt4d_opts:
                    if mask_method == "vggt4d" and vggt4d_opts.get("init_frame") is not None:
                        mask_argv += ["--vggt4d-init-frame", str(int(vggt4d_opts["init_frame"]))]
                    if vggt4d_opts.get("max_frames_for_vggt") is not None:
                        mask_argv += ["--vggt4d-max-frames", str(int(vggt4d_opts["max_frames_for_vggt"]))]
                _run_stage_via_conda_run(
                    conda_exe=str(args.conda_exe),
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
                    overwrite=bool(args.overwrite),
                    vggt4d_options=vggt4d_opts if mask_method in ("vggt4d", "vggt_framewise") else None,
                    repo_root=repo_root,
                )
            init_masks_dir = layout.mask_init_masks_dir

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
                    *([] if not args.overwrite else ["--overwrite"]),
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
                _run_stage_via_conda_run(
                    conda_exe=str(args.conda_exe),
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
                    overwrite=bool(args.overwrite),
                    sam3_options=sam3_opts if track_method == "sam3" else None,
                    repo_root=repo_root,
                )
            track_masks_dir = layout.track_masks_binary_dir

            if inpaint_env:
                _run_stage_via_conda_run(
                    conda_exe=str(args.conda_exe),
                    env_name=inpaint_env,
                    module="object_removal.cli.inpaint",
                    argv=[
                        "--run_dir",
                        str(run_dir),
                        "--frames_dir",
                        str(frames_dir),
                        "--masks_dir",
                        str(track_masks_dir),
                        "--method",
                        inpaint_method,
                        *([] if not args.overwrite else ["--overwrite"]),
                    ],
                    cwd=repo_root,
                )
            else:
                run_inpaint_stage(run_dir=run_dir, frames_dir=frames_dir, masks_dir=track_masks_dir, method=inpaint_method, overwrite=bool(args.overwrite))
            inpaint_frames_dir = layout.inpaint_frames_dir

        # Eval
        summary = run_eval(
            EvalInputs(
                output_dir=layout.eval_dir,
                part_label=args.part_label,
                experiment_name=name,
                pred_mask_dir=track_masks_dir,
                gt_mask_dir=gt_mask_dir,
                pred_frames_dir=inpaint_frames_dir,
                gt_frames_dir=gt_frames_dir,
                merge_gt_objects=True,
                video_metric_impl="internal",
            )
        )

        rows.append(_load_metrics_row(layout.eval_metrics_json, method_name=name))
        print(f"[compare] done: {name} (mask_jm={summary.get('mask_jm')} psnr={summary.get('video_psnr')})")

    rows.sort(key=lambda r: str(r.get("method", "")))
    _write_combined(rows, summary_root / "combined.csv", summary_root / "combined.md")
    print(f"Wrote: {summary_root / 'combined.csv'}")
    print(f"Wrote: {summary_root / 'combined.md'}")


if __name__ == "__main__":
    main()

