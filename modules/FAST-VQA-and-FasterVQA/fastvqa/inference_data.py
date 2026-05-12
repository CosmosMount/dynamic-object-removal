import numpy as np
import torch


def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    fallback_type="upsample",
    **kwargs,
):
    del nfrags, kwargs
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        orig_video = video
        video = torch.nn.functional.interpolate(
            video / 255.0,
            scale_factor=1 / ratio,
            mode="bilinear",
        )
        video = (video * 255.0).type_as(orig_video)
        _, res_h, res_w = video.shape[-3:]

    assert dur_t % aligned == 0, "Please provide matched vclip and align index"

    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength = res_h // fragments_h
    wlength = res_w // fragments_w

    if random:
        max_h = max(res_h - fsize_h, 0)
        max_w = max(res_w - fsize_w, 0)
    else:
        max_h = max(hlength - fsize_h, 0)
        max_w = max(wlength - fsize_w, 0)

    rnd_h = (
        torch.randint(max_h, (len(hgrids), len(wgrids), dur_t // aligned))
        if max_h > 0
        else torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int64)
    )
    rnd_w = (
        torch.randint(max_w, (len(hgrids), len(wgrids), dur_t // aligned))
        if max_w > 0
        else torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int64)
    )

    target_video = torch.zeros(video.shape[:-2] + (size_h, size_w), device=video.device, dtype=video.dtype)
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so = int(rnd_h[i, j, t])
                    w_so = int(rnd_w[i, j, t])
                else:
                    h_so = int(hs + rnd_h[i, j, t])
                    w_so = int(ws + rnd_w[i, j, t])
                h_eo = h_so + fsize_h
                w_eo = w_so + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo]
    return target_video


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def get_frame_indices(self, num_frames):
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t
        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0,
                tlength - self.fsize_t * self.frame_interval,
                size=len(tgrids),
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)
        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )
        return np.concatenate(ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        del train
        frame_inds = [self.get_frame_indices(total_frames) for _ in range(self.num_clips)]
        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds


class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def _get_test_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        clip_offsets = (
            self._get_train_clips(total_frames) if train else self._get_test_clips(total_frames)
        )
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)
