# NIQE implementation adapted from BasicSR (Apache 2.0)
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/niqe.py
# Prior parameters from basicsr/metrics/niqe_pris_params.npz (same repo).

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import convolve
    from scipy.special import gamma as scipy_gamma
except ImportError:  # pragma: no cover
    convolve = None  # type: ignore[misc, assignment]
    scipy_gamma = None  # type: ignore[misc, assignment]


def _estimate_aggd_param(block: np.ndarray) -> tuple[float, float, float]:
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(scipy_gamma(gam_reciprocal * 2)) / (
        scipy_gamma(gam_reciprocal) * scipy_gamma(gam_reciprocal * 3)
    )

    left_std = float(np.sqrt(np.mean(block[block < 0] ** 2)))
    right_std = float(np.sqrt(np.mean(block[block > 0] ** 2)))
    gammahat = left_std / (right_std + 1e-12)
    rhat = (np.mean(np.abs(block))) ** 2 / (np.mean(block**2) + 1e-12)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
    array_position = int(np.argmin((r_gam - rhatnorm) ** 2))

    alpha = float(gam[array_position])
    beta_l = left_std * float(np.sqrt(scipy_gamma(1 / alpha) / scipy_gamma(3 / alpha)))
    beta_r = right_std * float(np.sqrt(scipy_gamma(1 / alpha) / scipy_gamma(3 / alpha)))
    return alpha, beta_l, beta_r


def _compute_feature(block: np.ndarray) -> list[float]:
    feat: list[float] = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(np.roll(block, shift[0], axis=0), shift[1], axis=1)
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (scipy_gamma(2 / alpha) / scipy_gamma(1 / alpha))
        feat.extend([alpha, float(mean), beta_l, beta_r])
    return feat


def _imresize_half(img: np.ndarray) -> np.ndarray:
    """Downscale by 0.5 with area-like averaging (uint8 friendly)."""
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def _niqe_core(
    img: np.ndarray,
    mu_pris_param: np.ndarray,
    cov_pris_param: np.ndarray,
    gaussian_window: np.ndarray,
    block_size_h: int = 96,
    block_size_w: int = 96,
) -> float:
    assert img.ndim == 2
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    if num_block_h < 1 or num_block_w < 1:
        return float("nan")
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    dist_blocks: list[np.ndarray] = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode="nearest")
        sigma = np.sqrt(
            np.abs(convolve(np.square(img), gaussian_window, mode="nearest") - np.square(mu))
        )
        img_normalized = (img - mu) / (sigma + 1)

        feat_rows: list[list[float]] = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat_rows.append(_compute_feature(block))
        dist_blocks.append(np.array(feat_rows))

        if scale == 1:
            img = _imresize_half(img / 255.0) * 255.0

    distparam = np.concatenate(dist_blocks, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    if distparam_no_nan.shape[0] < 2:
        return float("nan")
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = mu_pris_param - mu_distparam
    quality = float(np.sqrt(np.squeeze(diff @ invcov_param @ diff.T)))
    return quality


def _bgr_to_y_matlab(img_bgr: np.ndarray) -> np.ndarray:
    """Y channel in [0,255] matching MATLAB rgb2ycbcr style used in IQA."""
    b, g, r = img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]
    y = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    return np.clip(y, 0.0, 255.0)


def compute_niqe_on_bgr(img_bgr: np.ndarray) -> Optional[float]:
    """Return NIQE for one BGR uint8/float image, or None if scipy missing."""
    if convolve is None or scipy_gamma is None:
        return None
    root = Path(__file__).resolve().parent
    npz_path = root / "niqe_pris_params.npz"
    if not npz_path.is_file():
        return None
    data = np.load(npz_path)
    mu_pris_param = np.ravel(data["mu_pris_param"]).astype(np.float64)
    cov_pris_param = np.asarray(data["cov_pris_param"], dtype=np.float64)
    gaussian_window = np.asarray(data["gaussian_window"], dtype=np.float64)

    img = img_bgr.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = _bgr_to_y_matlab(img)
    else:
        gray = np.squeeze(img)
    gray = np.round(gray).astype(np.float64)
    gh, gw = gray.shape[:2]
    if gh < 96 or gw < 96:
        return None
    out = _niqe_core(gray, mu_pris_param, cov_pris_param, gaussian_window)
    if math.isnan(out):
        return None
    return out
