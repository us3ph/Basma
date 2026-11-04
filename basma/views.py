"""Views of one image for the detector: a global view (short side -> 224, centre crop) and up to K texture-rich
224 x 224 crops of the base (<= 512) image. The inference path is deterministic and ported to JS (web/views.js);
training uses PIL with randomness (train_views)."""
import numpy as np
from .resample import VIEW, resize_area, fit_min_side, to_u8

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
K_CROPS = 3
GRID = 3


def normalize(view_f64):
    """(224, 224, 3) float64 in 0..255 -> (3, 224, 224) float32 normalized."""
    x = (view_f64.astype(np.float32) / np.float32(255.0) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def global_view(base):
    """base uint8 (H, W, 3) -> (224, 224, 3) float64: short side to 224 by area averaging, centre crop."""
    h, w = base.shape[:2]
    th, tw = fit_min_side(h, w, VIEW)
    x = resize_area(base, th, tw) if (th, tw) != (h, w) else base.astype(np.float64)
    oy, ox = (th - VIEW) // 2, (tw - VIEW) // 2
    return x[oy:oy + VIEW, ox:ox + VIEW]


def laplacian_var(gray):
    """Variance of the 4-neighbour Laplacian over interior pixels. gray float64 (h, w)."""
    c = gray[1:-1, 1:-1]
    lap = 4.0 * c - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    m = lap.sum() / lap.size
    d = lap - m
    return (d * d).sum() / lap.size


def grid_positions(n, size, grid=GRID):
    """Start offsets of `grid` windows of `size` spread over length n (deduplicated, ascending)."""
    if n <= size:
        return [0]
    pos = []
    for i in range(grid):
        p = int(np.floor((n - size) * i / (grid - 1) + 0.5))
        if p not in pos:
            pos.append(p)
    return pos


def texture_crops(base, k=K_CROPS):
    """base uint8 (H, W, 3) -> list of up to k (224, 224, 3) float64 crops, the most texture-rich windows of a
    3 x 3 grid at base resolution (images smaller than 224 are upscaled so the short side is 224)."""
    h, w = base.shape[:2]
    if min(h, w) < VIEW:
        th, tw = fit_min_side(h, w, VIEW)
        x = resize_area(base, th, tw)
    else:
        x = base.astype(np.float64)
    h, w = x.shape[:2]
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    cands = []
    for y in grid_positions(h, VIEW):
        for xx in grid_positions(w, VIEW):
            cands.append((laplacian_var(gray[y:y + VIEW, xx:xx + VIEW]), y, xx))
    # stable sort by score descending (ties keep grid order), identical in JS with a stable sort
    order = sorted(range(len(cands)), key=lambda i: -cands[i][0])
    out = []
    for i in order[:k]:
        _, y, xx = cands[i]
        out.append(x[y:y + VIEW, xx:xx + VIEW])
    return out


def inference_views(base):
    """-> (views float32 (V, 3, 224, 224), info dict). View 0 is the global view, then texture crops."""
    g = global_view(base)
    crops = texture_crops(base)
    views = np.stack([normalize(g)] + [normalize(c) for c in crops]).astype(np.float32)
    return views, {"n_crops": len(crops), "h": int(base.shape[0]), "w": int(base.shape[1])}


# ---------------------------------------------------------------- training (PIL, random)
def train_views(img, rng, n_crops=1):
    """PIL RGB image (already attacked) -> list of PIL 224 x 224 views: a random-resized global crop (+flip) and
    n_crops texture-biased random windows at the image's own resolution."""
    from PIL import Image
    w, h = img.size
    # global: random resized crop with area scale 0.45..1, ratio 3/4..4/3, then 224
    for _ in range(10):
        area = w * h * rng.uniform(0.45, 1.0)
        ratio = float(np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3))))
        cw, ch = int(round(np.sqrt(area * ratio))), int(round(np.sqrt(area / ratio)))
        if 0 < cw <= w and 0 < ch <= h:
            x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            g = img.crop((x0, y0, x0 + cw, y0 + ch))
            break
    else:
        s = min(w, h); g = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    g = g.resize((VIEW, VIEW), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]))
    if rng.random() < 0.5:
        g = g.transpose(Image.FLIP_LEFT_RIGHT)
    views = [g]
    # texture crops at native resolution (upscale small images so the short side is 224)
    src = img
    if min(w, h) < VIEW:
        th, tw = fit_min_side(h, w, VIEW)
        src = img.resize((tw, th), Image.BICUBIC); w, h = tw, th
    if rng.random() < 0.7 and (w > VIEW or h > VIEW):
        base = np.asarray(src)
        x = base.astype(np.float64)
        gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
        cands = [(laplacian_var(gray[y:y + VIEW, xx:xx + VIEW]), y, xx) for y in grid_positions(h, VIEW) for xx in grid_positions(w, VIEW)]
        cands.sort(key=lambda t: -t[0])
        top = cands[:max(1, min(5, len(cands)))]
    else:
        top = None
    for _ in range(n_crops):
        if top is not None:
            _, y, xx = top[rng.integers(0, len(top))]
            jy, jx = rng.integers(-8, 9), rng.integers(-8, 9)
            y, xx = int(np.clip(y + jy, 0, h - VIEW)), int(np.clip(xx + jx, 0, w - VIEW))
        else:
            y, xx = int(rng.integers(0, h - VIEW + 1)), int(rng.integers(0, w - VIEW + 1))
        c = src.crop((xx, y, xx + VIEW, y + VIEW))
        if rng.random() < 0.5:
            c = c.transpose(Image.FLIP_LEFT_RIGHT)
        views.append(c)
    return views


def pil_to_tensor(view):
    x = np.asarray(view, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1))
