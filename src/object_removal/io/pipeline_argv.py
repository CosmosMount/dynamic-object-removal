"""Build CLI argv fragments from pipeline YAML option dicts (used by compare → conda run)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def mask_argv_from_vggt_opts(opts: Optional[Dict[str, Any]], *, mask_method: str) -> List[str]:
    """Map `vggt4d:` YAML block to `object_removal.cli.mask` flags."""
    if not opts:
        return []
    out: List[str] = []

    def _f(key: str) -> Optional[float]:
        v = opts.get(key)
        return float(v) if v is not None else None

    def _i(key: str) -> Optional[int]:
        v = opts.get(key)
        return int(v) if v is not None else None

    if opts.get("dyn_threshold_scale") is not None:
        out += ["--vggt4d-dyn-threshold-scale", str(_f("dyn_threshold_scale"))]
    if opts.get("threshold") is not None:
        out += ["--vggt4d-threshold", str(_i("threshold"))]
    if opts.get("merge_all") is not None:
        out += ["--vggt4d-merge-all", "true" if bool(opts.get("merge_all")) else "false"]
    if opts.get("split_cc") is not None:
        out += ["--vggt4d-split-cc", "true" if bool(opts.get("split_cc")) else "false"]
    if opts.get("cc_min_area") is not None:
        out += ["--vggt4d-cc-min-area", str(_i("cc_min_area"))]
    if opts.get("cc_max_objects") is not None:
        out += ["--vggt4d-cc-max-objects", str(_i("cc_max_objects"))]
    if opts.get("cc_close_kernel") is not None:
        out += ["--vggt4d-cc-close-kernel", str(_i("cc_close_kernel"))]
    if opts.get("max_frames_for_vggt") is not None:
        out += ["--vggt4d-max-frames", str(_i("max_frames_for_vggt"))]
    if opts.get("vggt_chunk_size") is not None:
        out += ["--vggt4d-chunk-size", str(_i("vggt_chunk_size"))]
    if mask_method == "vggt4d" and opts.get("init_frame") is not None:
        out += ["--vggt4d-init-frame", str(_i("init_frame"))]
    if opts.get("tail_policy") is not None:
        out += ["--vggt4d-tail-policy", str(opts["tail_policy"])]
    return out


def inpaint_argv_from_diffueraser_opts(opts: Optional[Dict[str, Any]]) -> List[str]:
    if opts is None:
        return []
    out: List[str] = []
    mapping = [
        ("video_length", "--diffueraser-video-length", int),
        ("mask_dilation_iter", "--diffueraser-mask-dilation-iter", int),
        ("mask_hole_shrink_iters", "--diffueraser-mask-hole-shrink-iters", int),
        ("max_img_size", "--diffueraser-max-img-size", int),
        ("ref_stride", "--diffueraser-ref-stride", int),
        ("neighbor_length", "--diffueraser-neighbor-length", int),
        ("subvideo_length", "--diffueraser-subvideo-length", int),
    ]
    for key, flag, typ in mapping:
        if opts.get(key) is not None:
            out += [flag, str(typ(opts[key]))]
    return out


def inpaint_argv_from_propainter_opts(opts: Optional[Dict[str, Any]]) -> List[str]:
    if not opts:
        return []
    out: List[str] = []
    if opts.get("resize_ratio") is not None:
        out += ["--propainter-resize-ratio", str(float(opts["resize_ratio"]))]
    if opts.get("subvideo_length") is not None:
        out += ["--propainter-subvideo-length", str(int(opts["subvideo_length"]))]
    if opts.get("neighbor_length") is not None:
        out += ["--propainter-neighbor-length", str(int(opts["neighbor_length"]))]
    if opts.get("raft_iter") is not None:
        out += ["--propainter-raft-iter", str(int(opts["raft_iter"]))]
    if opts.get("fp16") is not None:
        out += ["--propainter-fp16", "true" if bool(opts.get("fp16")) else "false"]
    if opts.get("save_frames") is not None:
        out += ["--propainter-save-frames", "true" if bool(opts.get("save_frames")) else "false"]
    return out
