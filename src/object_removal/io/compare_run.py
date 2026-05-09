"""Resolve compare.yaml + pipelines.yaml into paths and registry (shared by compare CLI)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-export loaders from compare module would create cycles; duplicate minimal pieces.


def load_pipelines_yaml(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "pipelines config not found. Provide --pipelines_config or set `pipelines_config:` in compare YAML. "
            f"Tried: {path}"
        )
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from exc
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
        if "sam3" in spec and spec["sam3"] is not None and not isinstance(spec["sam3"], dict):
            raise ValueError(f"Pipeline {pid!r} field `sam3` must be a mapping or omitted in {path}")
        if "vggt4d" in spec and spec["vggt4d"] is not None and not isinstance(spec["vggt4d"], dict):
            raise ValueError(f"Pipeline {pid!r} field `vggt4d` must be a mapping or omitted in {path}")
        if "diffueraser" in spec and spec["diffueraser"] is not None and not isinstance(spec["diffueraser"], dict):
            raise ValueError(f"Pipeline {pid!r} field `diffueraser` must be a mapping or omitted in {path}")
        if "propainter" in spec and spec["propainter"] is not None and not isinstance(spec["propainter"], dict):
            raise ValueError(f"Pipeline {pid!r} field `propainter` must be a mapping or omitted in {path}")
        for k in ("mask", "track", "inpaint"):
            if k not in spec or not isinstance(spec[k], str) or not spec[k].strip():
                raise ValueError(f"Pipeline {pid!r} missing string field {k!r} in {path}")
        out[str(pid)] = dict(spec)
    return out


def load_compare_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid compare config (expected mapping at root): {path}")
    return dict(data)


def default_out_root_for_task(task: str, *, repo_root: Path) -> Path:
    if task.startswith("davis:"):
        seq = task.split(":", 1)[1].strip()
        if not seq:
            raise ValueError(f"Invalid task (empty seq): {task!r}")
        return (repo_root / "outputs" / "compare" / seq).resolve()
    raise ValueError(f"Cannot derive default out_root for task: {task!r} (only davis:SEQ supported)")


def load_env_map(path: Path) -> Dict[str, Dict[str, str]]:
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


def count_rgb_frames(frames_dir: Path) -> int:
    return sum(
        1
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def resolve_davis_paths(davis_root: Path, seq: str) -> Tuple[Path, Path]:
    frames_dir = davis_root / "JPEGImages" / "480p" / seq
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"DAVIS frames dir not found: {frames_dir}")
    if count_rgb_frames(frames_dir) == 0:
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


@dataclass(frozen=True)
class CompareRunContext:
    repo_root: Path
    compare_cfg_path: Path
    env_map_path: Path
    cfg: Dict[str, Any]
    task: str
    davis_root: Path
    pipelines_config_path: Path
    registry: Dict[str, Dict[str, Any]]
    pipeline_ids: List[str]
    out_root: Path
    runs_root: Path
    summary_root: Path
    meta_root: Path
    frames_dir: Path
    gt_mask_dir: Path
    gt_frames_dir: Path
    conda_exe: str
    env_map: Dict[str, Dict[str, str]]
    env_policy: str
    overwrite: bool
    part_label: str
    export_mask_vis: bool
    mask_vis_fps: float
    mask_vis_alpha: float
    only_stage: Optional[str]


def resolve_compare_context(
    *,
    repo_root: Path,
    compare_cfg_path: Path,
    task: Optional[str],
    davis_root: Optional[str],
    pipelines_config: Optional[str],
    pipelines_arg: Optional[List[str]],
    run_all_pipelines: bool,
    out_root: Optional[str],
    part_label: Optional[str],
    overwrite_cli: bool,
    conda_exe: Optional[str],
    env_map_path: Optional[str],
    env_policy: Optional[str],
    export_mask_vis_cli: bool,
    only_stage_cli: Optional[str],
) -> CompareRunContext:
    """Merge YAML + CLI overrides into a single context (compare main body)."""
    cfg = load_compare_yaml(compare_cfg_path)

    def _cfg_str(key: str, default: str = "") -> str:
        v = cfg.get(key)
        if v is None:
            return default
        return str(v).strip()

    task_res = (task if task is not None else _cfg_str("task", "")).strip()
    if not task_res:
        raise ValueError(
            "compare: missing `task`. Set `task:` in configs/compare.yaml (or pass --task). "
            f"Tried config file: {compare_cfg_path}"
        )

    davis_root_arg = davis_root if davis_root is not None else _cfg_str("davis_root", "data/DAVIS")
    if not davis_root_arg:
        davis_root_arg = "data/DAVIS"
    davis_root_p = Path(davis_root_arg)
    if not davis_root_p.is_absolute():
        davis_root_p = (repo_root / davis_root_p).resolve()

    pc_raw = pipelines_config if pipelines_config is not None else cfg.get("pipelines_config")
    if not pc_raw:
        pc_raw = "configs/pipelines.yaml"
    pipelines_config_path = Path(str(pc_raw).strip())
    if not pipelines_config_path.is_absolute():
        pipelines_config_path = (repo_root / pipelines_config_path).resolve()

    env_map_raw = env_map_path if env_map_path is not None else cfg.get("env_map", "configs/env_map.json")
    env_map_p = Path(str(env_map_raw).strip())
    if not env_map_p.is_absolute():
        env_map_p = (repo_root / env_map_p).resolve()
    env_map = load_env_map(env_map_p)

    conda_res = str(conda_exe if conda_exe is not None else cfg.get("conda_exe", "conda") or "conda")
    env_policy_res = str(env_policy if env_policy is not None else cfg.get("env_policy", "auto") or "auto")
    overwrite = bool(cfg.get("overwrite", False)) or bool(overwrite_cli)
    part_res = str(part_label if part_label is not None else cfg.get("part_label", "") or "")
    export_mask_vis = bool(cfg.get("export_mask_vis", False)) or bool(export_mask_vis_cli)
    mask_vis_fps = float(cfg.get("mask_vis_fps", 10.0) or 10.0)
    mask_vis_alpha = float(cfg.get("mask_vis_alpha", 0.5) or 0.5)

    only_stage: Optional[str] = None
    if only_stage_cli is not None and str(only_stage_cli).strip():
        only_stage = str(only_stage_cli).strip().lower()
    else:
        os_val = cfg.get("only_stage")
        if os_val is not None and str(os_val).strip():
            only_stage = str(os_val).strip().lower()
    if only_stage is not None and only_stage not in ("mask", "track", "inpaint", "eval"):
        raise ValueError(f"only_stage must be one of mask|track|inpaint|eval, got {only_stage!r}")

    registry = load_pipelines_yaml(pipelines_config_path)

    if run_all_pipelines:
        pipeline_ids = sorted(registry.keys())
    elif pipelines_arg is not None:
        pipeline_ids = [str(x).strip() for x in pipelines_arg if str(x).strip()]
        if not pipeline_ids:
            py = cfg.get("pipelines")
            if isinstance(py, list) and py:
                pipeline_ids = [str(x).strip() for x in py if str(x).strip()]
            if not pipeline_ids:
                raise ValueError(
                    "compare: empty pipeline list (--pipelines with no ids, and no non-empty `pipelines:` in YAML)."
                )
    elif bool(cfg.get("run_all_pipelines", False)):
        pipeline_ids = sorted(registry.keys())
    elif isinstance(cfg.get("pipelines"), list) and cfg["pipelines"]:
        pipeline_ids = [str(x).strip() for x in cfg["pipelines"] if str(x).strip()]
    else:
        raise ValueError(
            "compare: missing pipeline ids. Pass --pipelines, or set `pipelines:` in compare YAML, or use --all. "
            f"(pipelines config: {pipelines_config_path})"
        )

    for pid in pipeline_ids:
        if pid not in registry:
            raise ValueError(
                f"Unknown pipeline id {pid!r}. Known: {sorted(registry.keys())} "
                f"(config: {pipelines_config_path})"
            )

    # eval 只读各 run_dir 已有产物，可与 run_all_pipelines / 多条 pipelines 组合，逐条跑 eval。
    if only_stage is not None and only_stage != "eval" and len(pipeline_ids) != 1:
        raise ValueError(
            f"When only_stage={only_stage!r} is set, exactly one pipeline id is required "
            f"(got {len(pipeline_ids)}: {pipeline_ids})."
        )

    out_raw = out_root if out_root is not None else cfg.get("out_root")
    if out_raw is None or (isinstance(out_raw, str) and not str(out_raw).strip()):
        out_root_p = default_out_root_for_task(task_res, repo_root=repo_root)
    else:
        out_root_p = Path(str(out_raw).strip())
        if not out_root_p.is_absolute():
            out_root_p = (repo_root / out_root_p).resolve()

    if not task_res.startswith("davis:"):
        raise ValueError("Only davis:SEQ is supported in MVP compare runner.")

    seq = task_res.split(":", 1)[1]
    frames_dir, gt_mask_dir = resolve_davis_paths(davis_root_p, seq)
    gt_frames_dir = frames_dir

    runs_root = out_root_p / "runs"
    summary_root = out_root_p / "summary"
    meta_root = out_root_p / "meta"

    return CompareRunContext(
        repo_root=repo_root,
        compare_cfg_path=compare_cfg_path,
        env_map_path=env_map_p,
        cfg=cfg,
        task=task_res,
        davis_root=davis_root_p,
        pipelines_config_path=pipelines_config_path,
        registry=registry,
        pipeline_ids=pipeline_ids,
        out_root=out_root_p,
        runs_root=runs_root,
        summary_root=summary_root,
        meta_root=meta_root,
        frames_dir=frames_dir,
        gt_mask_dir=gt_mask_dir,
        gt_frames_dir=gt_frames_dir,
        conda_exe=conda_res,
        env_map=env_map,
        env_policy=env_policy_res,
        overwrite=overwrite,
        part_label=part_res,
        export_mask_vis=export_mask_vis,
        mask_vis_fps=mask_vis_fps,
        mask_vis_alpha=mask_vis_alpha,
        only_stage=only_stage,
    )


def env_for(env_map: Dict[str, Dict[str, str]], *, stage: str, method: str) -> str:
    stage_map = env_map.get(stage, {})
    if stage == "eval":
        return str(stage_map.get(method, stage_map.get("default", "")) or "")
    return str(stage_map.get(method, "") or "")
