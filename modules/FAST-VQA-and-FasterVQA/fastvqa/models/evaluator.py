from functools import reduce

import torch
import torch.nn as nn

from .head import VQAHead
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny


class DiViDeAddEvaluator(nn.Module):
    """Minimal inference-only evaluator for the VQA CLI."""

    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        super().__init__()
        if divide_head:
            raise NotImplementedError("Inference-only build does not support divide_head=True")
        if var:
            raise NotImplementedError("Inference-only build does not support var=True")

        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        self.vqa_head = VQAHead(**vqa_head)

        for key, hypers in backbone.items():
            if key not in self.backbone_preserve_keys:
                continue
            t_backbone_size = hypers["type"] if backbone_size == "divided" else backbone_size
            if t_backbone_size == "swin_tiny":
                module = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                module = VideoBackbone()
            elif t_backbone_size == "swin_tiny_grpb_m":
                module = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                module = swin_3d_small(**backbone[key])
            else:
                raise NotImplementedError(
                    f"Unsupported backbone_size={t_backbone_size!r} in inference-only build"
                )
            setattr(self, key.split("_")[0] + "_backbone", module)

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        prev_mode = self.training
        if inference:
            self.eval()

        context = torch.no_grad() if inference else torch.enable_grad()
        with context:
            scores = []
            feats = {}
            for key, value in vclips.items():
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    value,
                    multi=self.multi,
                    layer=self.layer,
                    **kwargs,
                )
                scores.append(self.vqa_head(feat))
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))

            if reduce_scores:
                scores = reduce(lambda x, y: x + y, scores) if len(scores) > 1 else scores[0]
                if pooled:
                    scores = torch.mean(scores, (1, 2, 3, 4))

        self.train(prev_mode)
        if return_pooled_feats:
            return scores, feats
        return scores