import numpy as np
import torch
import yaml
from inference.inference_core import InferenceCore
from model.network import XMem
from torchvision import transforms
from util.mask_mapper import MaskMapper
from util.range_transform import im_normalization
_PALETTE = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
    ],
    dtype=np.uint8,
)


def _paint_mask_overlay(frame: np.ndarray, binary_mask: np.ndarray, obj_id: int) -> np.ndarray:
    """Small self-contained overlay painter so the XMem path has no tool dependencies."""
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.max() <= 0:
        return frame
    color = _PALETTE[(max(1, int(obj_id)) - 1) % len(_PALETTE)].astype(np.float32)
    out = frame.astype(np.float32).copy()
    mask = binary_mask.astype(bool)
    out[mask] = out[mask] * 0.45 + color * 0.55
    return out.clip(0, 255).astype(np.uint8)


class BaseTracker:
    def __init__(self, xmem_checkpoint, device, sam_model=None, model_type=None) -> None:
        """Minimal XMem wrapper for batch tracking."""
        with open("config/config.yaml", "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        network = XMem(config, xmem_checkpoint).to(device).eval()
        self.tracker = InferenceCore(network, config)
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device
        self.mapper = MaskMapper()

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        if first_frame_annotation is not None:
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.as_tensor(mask, device=self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None

        frame_tensor = self.im_transform(frame).to(self.device)
        probs, _ = self.tracker.step(frame_tensor, mask, labels)

        out_mask = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
        final_mask = np.zeros_like(out_mask)
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        painted_image = frame
        for obj_id in range(1, int(final_mask.max()) + 1):
            binary = (final_mask == obj_id).astype(np.uint8)
            if binary.max() == 0:
                continue
            painted_image = _paint_mask_overlay(painted_image, binary, obj_id)

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()
    # # first frame
    # first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/camel/00000.png'
    # # load frames
    # frames = []
    # for video_path in video_path_list:
    #     frames.append(np.array(Image.open(video_path).convert('RGB')))
    # frames = np.stack(frames, 0)    # N, H, W, C
    # # load first frame annotation
    # first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # print('first video done. clear.')

    # tracker.clear_memory()
    # # track anything given in the first frame annotation
    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/camel/{ti:05d}.png')

    # # failure case test
    # failure_path = '/ssd1/gaomingqi/failure'
    # frames = np.load(os.path.join(failure_path, 'video_frames.npy'))
    # # first_frame = np.array(Image.open(os.path.join(failure_path, 'template_frame.png')).convert('RGB'))
    # first_mask = np.array(Image.open(os.path.join(failure_path, 'template_mask.png')).convert('P'))
    # first_mask = np.clip(first_mask, 0, 1)

    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, probs, painted_image = tracker.track(frame, first_mask)
    #     else:
    #         mask, probs, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/failure/LJ/{ti:05d}.png')
    #     prob = Image.fromarray((probs[1].cpu().numpy()*255).astype('uint8'))

    #     # prob.save(f'/ssd1/gaomingqi/failure/probs/{ti:05d}.png')
