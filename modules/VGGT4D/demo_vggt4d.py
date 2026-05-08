import argparse
import os
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (adaptive_multiotsu_variance,
                                             cluster_attention_maps,
                                             extract_dyn_map)
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (save_depth, save_depth_conf,
                                save_dynamic_masks, save_intrinsic_txt,
                                save_rgb, save_tum_poses)
from vggt.utils.load_fn import load_and_preprocess_images

device = torch.device("cuda") \
    if torch.cuda.is_available() \
    else torch.device("cpu")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VGGT_TRACKER_NAME = "model_tracker_fixed_e20.pt"
_VGGT_TRACKER_HF_REPO = "facebook/VGGT_tracker_fixed"
_VGGT_TRACKER_URL = (
    "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/"
    f"{_VGGT_TRACKER_NAME}"
)


def _scan_vggt4d_tracker_ckpt() -> Optional[Path]:
    """Resolve checkpoint vs cwd (compare/conda often run from repo root)."""
    cand = [
        _REPO_ROOT / "ckpts" / "vggt4d" / _VGGT_TRACKER_NAME,
        _REPO_ROOT / "ckpts" / _VGGT_TRACKER_NAME,
        Path("ckpts/vggt4d") / _VGGT_TRACKER_NAME,
        Path("ckpts") / _VGGT_TRACKER_NAME,
    ]
    for p in cand:
        if p.is_file():
            return p.resolve()
    return None


def _try_download_vggt4d_tracker_hf() -> bool:
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    dest_dir = _REPO_ROOT / "ckpts" / "vggt4d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=_VGGT_TRACKER_HF_REPO,
            filename=_VGGT_TRACKER_NAME,
            local_dir=str(dest_dir),
        )
    except Exception as exc:
        print(f"[VGGT4D] huggingface_hub download failed ({exc})")
        return False
    return (dest_dir / _VGGT_TRACKER_NAME).is_file()


def _try_download_vggt4d_tracker_url() -> bool:
    dest = _REPO_ROOT / "ckpts" / "vggt4d" / _VGGT_TRACKER_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_VGGT_TRACKER_URL, str(dest))
    except Exception as exc:
        print(f"[VGGT4D] direct URL download failed ({exc})")
        return False
    return dest.is_file()


def _ensure_vggt4d_tracker_ckpt() -> Path:
    p = _scan_vggt4d_tracker_ckpt()
    if p is not None:
        return p
    if _try_download_vggt4d_tracker_hf() or _try_download_vggt4d_tracker_url():
        p = _scan_vggt4d_tracker_ckpt()
        if p is not None:
            return p
    exp = _REPO_ROOT / "ckpts" / "vggt4d" / _VGGT_TRACKER_NAME
    raise FileNotFoundError(
        f"VGGT4D tracker checkpoint not found under {_REPO_ROOT}. "
        f"Expected e.g. {exp}\n"
        f"Install manually, e.g.:\n"
        f'  wget -c "{_VGGT_TRACKER_URL}?download=true" -O {exp}\n'
        f"Or install huggingface_hub and retry (auto-download uses {_VGGT_TRACKER_HF_REPO})."
    )


model = VGGTFor4D()
_ckpt = _ensure_vggt4d_tracker_ckpt()
model.load_state_dict(torch.load(str(_ckpt), weights_only=True))
model.eval()
model = model.to(device)


def process_scene(scene_dir: Path, output_dir: Path, dyn_threshold_scale: float = 1.0):
    """
    Process a single scene

    Args:
        scene_dir: Scene input directory path
        output_dir: Scene output directory path
        dyn_threshold_scale: Scale factor for dynamic mask threshold (<1 = more motion detected)
    """
    image_paths = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"Warning: No images found in {scene_dir}, skipping this scene")
        return

    print(f"Processing scene: {scene_dir.name} ({len(image_paths)} images)")

    images = load_and_preprocess_images(
        [str(image_path) for image_path in image_paths]).to(device)
    n_img, _, h_img, w_img = images.shape

    output_dir.mkdir(parents=True, exist_ok=True)

    # stage 1 predict depth map and dynamic map
    print("  Stage 1: predict depth map and dynamic map")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(
        model, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])

    dyn_maps = extract_dyn_map(qk_dict, images)
    # save memory usage
    # dyn_maps = batch_extract_dyn_map(qk_dict, images)

    n_img, _, h_img, w_img = images.shape

    h_tok, w_tok = h_img // 14, w_img // 14

    feat_map = rearrange(
        enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)

    norm_dyn_map, _ = cluster_attention_maps(
        feat_map, dyn_maps)

    upsampled_map = F.interpolate(rearrange(
        norm_dyn_map, "n_img h w -> n_img 1 h w"), size=(h_img, w_img), mode='bilinear', align_corners=False)
    upsampled_map = rearrange(
        upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy()) * dyn_threshold_scale
    dyn_masks = upsampled_map > thres

    # stage 2 refine extrinsics by dynamic map
    print("  Stage 2: refine extrinsics by dynamic map")
    if "enc_feat" in locals():
        del enc_feat
    if "feat_map" in locals():
        del feat_map

    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    pred_intrinsic = predictions1["intrinsic"]
    pred_cam2world2 = predictions2["cam2world"]

    pred_depths = predictions1["depth"]
    pred_conf = predictions1["depth_conf"]

    # save predictions
    final_prediction = {**predictions1}
    final_prediction["extrinsic"] = predictions2["extrinsic"]
    final_prediction["cam2world"] = pred_cam2world2

    # stage 3 refine dynamic map
    print("  Stage 3: refine dynamic map")
    if "feat_map" in locals():
        del feat_map
    torch.cuda.empty_cache()

    pred_intrinsic = final_prediction["intrinsic"]
    pred_cam2world = final_prediction["cam2world"]

    pred_depths = final_prediction["depth"]
    pred_conf = final_prediction["depth_conf"]

    refiner = RefineDynMask(images, torch.tensor(pred_depths).to(device),
                            dyn_masks.to(device),
                            torch.tensor(
                                pred_cam2world).float().to(device),
                            torch.tensor(pred_intrinsic).to(device),
                            device)

    refined_mask = refiner.refine_masks()
    del refiner

    print(f"  Saving predictions to {output_dir}\n")
    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, pred_cam2world2)
    save_dynamic_masks(output_dir, refined_mask)


def main(input_dir: str, output_dir: str, dyn_threshold_scale: float = 1.0):
    """
    Main function

    Args:
        input_dir: Input data directory path
        output_dir: Output result directory path
        dyn_threshold_scale: Scale factor for dynamic mask threshold (<1 = more motion detected)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    scene_dirs = sorted(scene_dirs)

    if len(scene_dirs) == 0:
        raise ValueError(f"No scene directories found in {input_dir}")

    print(f"Found {len(scene_dirs)} scenes, starting processing...\n")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_output_dir = output_dir / scene_name
        process_scene(scene_dir, scene_output_dir, dyn_threshold_scale)

    print(f"All scenes processed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT4D demo script")
    parser.add_argument("--input_dir", type=str, default=None, help="Input data directory path")
    parser.add_argument("--output_dir", type=str,
                        default=None, help="Output result directory path")
    parser.add_argument("--dyn_threshold_scale", type=float, default=1.0,
                        help="Scale factor for dynamic mask threshold (<1 = more motion detected, default: 1.0)")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, dyn_threshold_scale=args.dyn_threshold_scale)
