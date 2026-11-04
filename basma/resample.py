"""Deterministic image resampling shared, operation for operation, with the browser port (web/resample.js).

Everything is float64 and uses only +, -, *, / in a fixed order, so Python (numpy) and JavaScript produce
bit-identical results from the same uint8 input. Two operations:

* block_reduce(img, k): mean of each k x k block (integer-factor box downscale), remainder rows/cols dropped,
  rounded with floor(x + 0.5) to uint8. Used first for very large images.
* resize_area(img, oh, ow): exact area-averaging resize (fractional box filter), separable, computed with
  cumulative sums so the cost is O(pixels). Output float64; round with to_u8 when a uint8 image is needed.
"""
import numpy as np

MAX_SIDE = 512          # the base image everything is computed from (also the stored training resolution)
VIEW = 224              # model input size
REDUCE_ABOVE = 1024     # block-reduce first when the longest side is above this


def to_u8(x):
    """Round half up, clip, uint8 (JS: Math.floor(x + 0.5))."""
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def block_reduce(img, k):
    """img: (H, W, C) uint8 -> (H//k, W//k, C) uint8, mean of k x k blocks (exact integer sums, half-up rounding)."""
    if k <= 1:
        return img
    h, w, c = img.shape
    hh, ww = h // k, w // k
    x = img[:hh * k, :ww * k].reshape(hh, k, ww * k * c).sum(axis=1, dtype=np.uint32)   # rows of each block
    x = x.reshape(hh, ww, k, c).sum(axis=2, dtype=np.uint32)                            # columns of each block
    return ((x + (k * k) // 2) // (k * k)).astype(np.uint8)                              # floor(sum/k^2 + 0.5)


def _area_axis0(img, n_out):
    """Area-average resample along axis 0. img float64 (H, ...) -> (n_out, ...)."""
    n_in = img.shape[0]
    if n_out == n_in:
        return img
    s = n_in / n_out
    cs = np.concatenate([np.zeros((1,) + img.shape[1:], dtype=np.float64), np.cumsum(img, axis=0, dtype=np.float64)], axis=0)
    idx = np.arange(n_out + 1, dtype=np.float64) * s          # edges a_i = i*s
    fl = np.floor(idx)
    fr = idx - fl
    fl_i = np.minimum(fl.astype(np.int64), n_in)              # cumsum index (clipped to H)
    pix_i = np.minimum(fl.astype(np.int64), n_in - 1)         # pixel index for the fractional part
    shape = (n_out + 1,) + (1,) * (img.ndim - 1)
    F = cs[fl_i] + fr.reshape(shape) * img[pix_i]             # integral up to each edge
    F[fl_i >= n_in] = cs[n_in]                                # edges at/after the end: whole image
    return (F[1:] - F[:-1]) / s


def resize_area(img, oh, ow):
    """img: (H, W, C) uint8 or float64 -> (oh, ow, C) float64 by exact area averaging."""
    x = img.astype(np.float64)
    x = _area_axis0(x, oh)
    x = _area_axis0(np.ascontiguousarray(x.transpose(1, 0, 2)), ow).transpose(1, 0, 2)
    return np.ascontiguousarray(x)


def fit_max_side(h, w, max_side):
    """Target (h, w) so the longest side == max_side (no upscaling); rounding half up like JS Math.round for positives."""
    m = max(h, w)
    if m <= max_side:
        return h, w
    sc = max_side / m
    return max(1, int(np.floor(h * sc + 0.5))), max(1, int(np.floor(w * sc + 0.5)))


def fit_min_side(h, w, min_side):
    """Target (h, w) so the shortest side == min_side (up or down)."""
    m = min(h, w)
    sc = min_side / m
    return max(min_side, int(np.floor(h * sc + 0.5))), max(min_side, int(np.floor(w * sc + 0.5)))


def to_base(img):
    """Decoded RGB uint8 (H, W, 3) -> base uint8 image with longest side <= MAX_SIDE. The first step of every
    inference; the stored training images went through the same geometry (PIL, so not bit-identical there)."""
    h, w = img.shape[:2]
    k = max(1, max(h, w) // REDUCE_ABOVE)
    if k > 1:
        img = block_reduce(img, k)
        h, w = img.shape[:2]
    th, tw = fit_max_side(h, w, MAX_SIDE)
    if (th, tw) == (h, w):
        return img
    return to_u8(resize_area(img, th, tw))
