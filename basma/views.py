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


def texture_crops(img, k=K_CROPS):
    """img uint8 (H, W, 3), the base or (v6) the image's own untouched pixels -> list of up to k (224, 224, 3) float64
    crops, the most texture-rich windows of a 3 x 3 grid (images smaller than 224 are upscaled so the short side is
    224). Only the candidate windows are converted to float, so a 24-megapixel image costs no more than a small one."""
    h, w = img.shape[:2]
    if min(h, w) < VIEW:                          # small image: upscaled in float64 (as before, bit-identical to the JS port)
        th, tw = fit_min_side(h, w, VIEW)
        small = resize_area(img, th, tw); h, w = small.shape[:2]
        window = lambda y, xx: small[y:y + VIEW, xx:xx + VIEW]
    else:
        window = lambda y, xx: img[y:y + VIEW, xx:xx + VIEW].astype(np.float64)
    cands = []
    for y in grid_positions(h, VIEW):
        for xx in grid_positions(w, VIEW):
            x = window(y, xx)
            gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
            cands.append((laplacian_var(gray), y, xx))
    # stable sort by score descending (ties keep grid order), identical in JS with a stable sort
    order = sorted(range(len(cands)), key=lambda i: -cands[i][0])
    return [window(cands[i][1], cands[i][2]) for i in order[:k]]


def inference_views(base, native=None):
    """-> (views float32 (V, 3, 224, 224), info dict). View 0 is the global view of the base; the texture crops come
    from `native` (the image's untouched pixels, v6) when given, else from the base."""
    g = global_view(base)
    crops = texture_crops(base if native is None else native)
    views = np.stack([normalize(g)] + [normalize(c) for c in crops]).astype(np.float32)
    return views, {"n_crops": len(crops), "h": int(base.shape[0]), "w": int(base.shape[1])}


# ---------------------------------------------------------------- training (PIL, random)
def tile_jitter(c, rng, p=0.30):
    """Label-independent softening / grain / sharpening of a native tile, so 'smooth = AI' cannot be learned."""
    from PIL import ImageFilter
    r = rng.random()
    if r < p * 0.4:
        return c.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.9))))
    if r < p * 0.8:
        x = np.asarray(c).astype(np.float32) + rng.normal(0, float(rng.uniform(1.0, 4.0)), size=np.asarray(c).shape).astype(np.float32)
        from PIL import Image
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if r < p:
        return c.filter(ImageFilter.UnsharpMask(radius=float(rng.uniform(0.8, 1.6)), percent=int(rng.integers(50, 140)), threshold=2))
    return c


def train_views(img, rng, n_crops=1, native=None):
    """PIL RGB image (already attacked) -> list of PIL 224 x 224 views: a random-resized global crop (+flip) and
    n_crops texture-biased random windows at the image's own resolution (v6: from `native`, the attacked window of
    the original pixels, when given)."""
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
    src = native if native is not None else img
    w, h = src.size
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
        views.append(tile_jitter(c, rng))
    return views


def pil_to_tensor(view):
    x = np.asarray(view, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1))
