from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Params:
    cfg: Path
    checkpoint: Path
    score_thresh: float = 0.0


def run(
    *,
    repo_root: Path,
    base_video_dir: Path,
    input_mask_dir: Path,
    video_list_file: Path,
    output_mask_dir: Path,
    params: Params,
) -> dict:
    """Run SAM2 VOS inference in-process.

    You must run this inside the `sam2` conda env (or any env where torch+sam2 work).
    """
    sam2_root = repo_root / "modules" / "sam2"
    tool = sam2_root / "tools" / "vos_inference.py"
    if not tool.is_file():
        raise FileNotFoundError(f"Missing SAM2 tool: {tool}")

    # make sure sam2 package import resolves
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    # Hydra composes with `compose(config_name=..., ...)` and search path `pkg://sam2`,
    # i.e. paths are relative to the *inner* package dir `modules/sam2/sam2/`.
    # A repo path like `modules/sam2/sam2/configs/...` becomes `sam2/configs/...` when
    # taken relative to `modules/sam2`; Hydra then wrongly looks for `sam2/sam2/configs/...`.
    # Strip a single leading `sam2/` so the argument matches upstream (e.g. `configs/sam2.1/...`).
    cfg_arg: str
    try:
        cfg_rel = params.cfg.resolve().relative_to(sam2_root.resolve())
        cfg_arg = cfg_rel.as_posix()
        if cfg_arg.startswith("sam2/"):
            cfg_arg = cfg_arg[len("sam2/") :]
    except Exception:
        cfg_arg = str(params.cfg)

    # Resolve checkpoint path before `chdir(sam2_root)` since upstream loads it from cwd.
    ckpt_path = params.checkpoint
    if not ckpt_path.is_absolute():
        ckpt_path = (repo_root / ckpt_path).resolve()

    # Same for all I/O paths: `vos_inference` opens these after we `chdir(sam2_root)`.
    launch_cwd = Path.cwd().resolve()

    def _abs(p: Path) -> Path:
        p = Path(p)
        return p.resolve() if p.is_absolute() else (launch_cwd / p).resolve()

    base_video_dir = _abs(base_video_dir)
    input_mask_dir = _abs(input_mask_dir)
    video_list_file = _abs(video_list_file)
    output_mask_dir = _abs(output_mask_dir)

    spec = importlib.util.spec_from_file_location("sam2_vos_inference", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec: {tool}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "vos_inference.py",
        "--sam2_cfg",
        cfg_arg,
        "--sam2_checkpoint",
        str(ckpt_path),
        "--base_video_dir",
        str(base_video_dir),
        "--input_mask_dir",
        str(input_mask_dir),
        "--video_list_file",
        str(video_list_file),
        "--output_mask_dir",
        str(output_mask_dir),
        "--score_thresh",
        str(params.score_thresh),
    ]

    old_argv = sys.argv
    import os

    old_cwd = Path.cwd()
    try:
        os.chdir(sam2_root)
        sys.argv = argv
        mod.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    return {"output_mask_dir": str(output_mask_dir)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="SAM2 VOS track method")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--base_video_dir", required=True)
    ap.add_argument("--input_mask_dir", required=True)
    ap.add_argument("--video_list_file", required=True)
    ap.add_argument("--output_mask_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--score_thresh", type=float, default=0.0)
    args = ap.parse_args()

    run(
        repo_root=Path(args.repo_root),
        base_video_dir=Path(args.base_video_dir),
        input_mask_dir=Path(args.input_mask_dir),
        video_list_file=Path(args.video_list_file),
        output_mask_dir=Path(args.output_mask_dir),
        params=Params(cfg=Path(args.cfg), checkpoint=Path(args.checkpoint), score_thresh=float(args.score_thresh)),
    )


if __name__ == "__main__":
    main()

