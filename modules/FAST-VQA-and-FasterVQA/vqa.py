import argparse
from pathlib import Path

import decord
import numpy as np
import torch
import yaml

# Without this, decord returns numpy arrays and torch.stack(imgs) fails.
decord.bridge.set_bridge("torch")

from fastvqa.inference_data import FragmentSampleFrames, SampleFrames, get_spatial_fragments
from fastvqa.models import DiViDeAddEvaluator

def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

_ROOT = Path(__file__).resolve().parent
_DEFAULT_FASTERVQA_WEIGHT = _ROOT.parent.parent / "ckpts" / "fastvqa" / "FAST_VQA_3D_1_1.pth"

opts = {
    "FasterVQA": _ROOT / "options" / "fast" / "f3dvqa-b.yml",
    "FasterVQA-MS": _ROOT / "options" / "fast" / "fastervqa-ms.yml",
    "FasterVQA-MT": _ROOT / "options" / "fast" / "fastervqa-mt.yml",
    "FAST-VQA": _ROOT / "options" / "fast" / "fast-b.yml",
    "FAST-VQA-M": _ROOT / "options" / "fast" / "fast-m.yml",
}


def _resolve_weight_path(model: str, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    if model in {"FasterVQA", "FasterVQA-MS", "FasterVQA-MT"}:
        return _DEFAULT_FASTERVQA_WEIGHT if not path.exists() else path
    return path


def _load_state_dict(weight_path: Path, device: str):
    ckpt = torch.load(weight_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ### can choose between
    ### options/fast/f3dvqa-b.yml
    ### options/fast/fast-b.yml
    ### options/fast/fast-m.yml
    
    parser.add_argument(
        "-m", "--model", type=str, 
        default="FasterVQA", 
        help="model type: can choose between FasterVQA, FasterVQA-MS, FasterVQA-MT, FAST-VQA, FAST-VQA-M",
    )
    
    ## can be your own
    parser.add_argument(
        "-v", "--video_path", type=str, 
        default="./demos/10053703034.mp4", 
        help="the input video path"
    )
    
    parser.add_argument(
        "-d", "--device", type=str, 
        default="cuda", 
        help="the running device"
    )
    
    
    args = parser.parse_args()

    video_reader = decord.VideoReader(args.video_path)
    
    opt_path = opts.get(args.model, opts["FAST-VQA"])
    with open(opt_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    weight_path = _resolve_weight_path(args.model, str(opt["test_load_path"]))
    evaluator.load_state_dict(_load_state_dict(weight_path, args.device))

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag",1) > 1:
            sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                           fragments_t=sample_args.get("t_frag",1),
                                           num_clips=sample_args.get("num_clips",1),
                                          )
        else:
            sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])
        
        num_clips = sample_args.get("num_clips",1)
        frames = sampler(len(video_reader))
        print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        
        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples[sample_type] = sampled_video.to(args.device)
        print(sampled_video.shape)
    result = evaluator(vsamples)
    score = sigmoid_rescale(result.mean().item(), model=args.model)
    print(f"The quality score of the video (range [0,1]) is {score:.5f}.")
