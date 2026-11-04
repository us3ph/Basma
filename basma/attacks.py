"""Attack buckets: what happens to an image between generation and the detector (posting, resizing, colour
edits, screenshots...). Training draws one random bucket per sample; evaluation applies fixed conditions.
All functions take and return PIL RGB images."""
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FILTERS = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]
UI_COLORS = [(255, 255, 255), (30, 30, 30), (0, 0, 0), (240, 240, 240), (24, 25, 28), (255, 255, 255)]

# bucket name -> sampling probability during training
TRAIN_BUCKETS = {"clean": 0.14, "jpeg": 0.14, "webp": 0.05, "resize": 0.14, "screenshot": 0.16, "social": 0.15,
                 "color": 0.06, "blur_noise": 0.05, "crop": 0.05, "combo": 0.06}


def jpeg(img, q, subsampling=None):
    b = io.BytesIO()
    kw = {"quality": int(q)}
    if subsampling is not None:
        kw["subsampling"] = subsampling
    img.save(b, "JPEG", **kw)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def webp(img, q):
    b = io.BytesIO(); img.save(b, "WEBP", quality=int(q))
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def resize_scale(img, s, flt=Image.BICUBIC):
    w, h = img.size
    return img.resize((max(16, int(round(w * s))), max(16, int(round(h * s)))), flt)


def resize_max_side(img, m, flt=Image.BICUBIC):
    w, h = img.size
    if max(w, h) <= m:
        return img
    s = m / max(w, h)
    return img.resize((max(16, int(round(w * s))), max(16, int(round(h * s)))), flt)


def ui_bars(img, rng, color=None, top=None, bottom=None, side=None):
    """Screenshot-like framing: solid bars above/below (and sometimes beside) the image."""
    w, h = img.size
    color = color or UI_COLORS[rng.integers(0, len(UI_COLORS))]
    top = int(h * (rng.uniform(0.04, 0.16) if top is None else top))
    bottom = int(h * (rng.uniform(0.0, 0.14) if bottom is None else bottom))
    side = int(w * (rng.uniform(0.0, 0.12) if side is None else side)) if (side is not None or rng.random() < 0.35) else 0
    out = Image.new("RGB", (w + 2 * side, h + top + bottom), color)
    out.paste(img, (side, top))
    return out


def gamma(img, g):
    lut = [int(round(255 * ((i / 255.0) ** g))) for i in range(256)]
    return img.point(lut * 3)


def color_edit(img, rng, strength=1.0):
    b, c, s = [1 + strength * rng.uniform(-0.4, 0.4) for _ in range(3)]
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    if rng.random() < 0.5:
        img = gamma(img, float(np.exp(rng.uniform(-0.35, 0.35))))
    if rng.random() < 0.4:
        img = hue_shift(img, rng.uniform(-0.08, 0.08))
    if rng.random() < 0.08:
        img = img.convert("L").convert("RGB")
    return img


def hue_shift(img, frac):
    hsv = np.asarray(img.convert("HSV")).copy()
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + int(round(frac * 255))) % 256).astype(np.uint8)
    return Image.fromarray(hsv, "HSV").convert("RGB")


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def noise(img, sigma, rng):
    x = np.asarray(img).astype(np.float32) + rng.normal(0, sigma, size=np.asarray(img).shape).astype(np.float32)
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def sharpen(img, pct=120, r=1.5):
    return img.filter(ImageFilter.UnsharpMask(radius=r, percent=pct, threshold=2))


def crop_frac(img, area_frac, rng):
    w, h = img.size
    ratio = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    cw = int(round(min(w, np.sqrt(w * h * area_frac * ratio))))
    ch = int(round(min(h, w * h * area_frac / max(1, cw))))
    cw, ch = max(32, min(w, cw)), max(32, min(h, ch))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return img.crop((x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------- training buckets
def _screenshot(img, rng):
    s = rng.uniform(0.35, 1.0)
    img = resize_scale(img, s, [Image.BILINEAR, Image.BICUBIC][rng.integers(0, 2)])
    if rng.random() < 0.3:
        img = ui_bars(img, rng)
    if rng.random() < 0.2:
        img = gamma(img, float(np.exp(rng.uniform(-0.1, 0.1))))
    if rng.random() < 0.65:
        img = jpeg(img, rng.integers(70, 96))
    return img


def _social(img, rng):
    img = resize_max_side(img, int(rng.choice([256, 320, 400, 480, 512])), FILTERS[rng.integers(1, 5)])
    img = jpeg(img, rng.integers(60, 89), subsampling=int(rng.choice([0, 2])))
    if rng.random() < 0.4:  # re-shared once more
        img = resize_scale(img, rng.uniform(0.6, 1.0), FILTERS[rng.integers(1, 5)])
        img = jpeg(img, rng.integers(60, 86))
    return img


def _resize(img, rng):
    img = resize_scale(img, rng.uniform(0.3, 1.6), FILTERS[rng.integers(0, 5)])
    if rng.random() < 0.5:
        img = jpeg(img, rng.integers(60, 96))
    return img


def _blur_noise(img, rng):
    r = rng.random()
    if r < 0.4:
        return blur(img, rng.uniform(0.3, 1.5))
    if r < 0.8:
        return noise(img, rng.uniform(2, 8), rng)
    return sharpen(img, int(rng.integers(60, 200)), rng.uniform(0.8, 2.5))


def _crop(img, rng):
    img = crop_frac(img, rng.uniform(0.3, 0.9), rng)
    if rng.random() < 0.5:
        img = resize_scale(img, rng.uniform(1.0, 1.8), FILTERS[rng.integers(1, 4)])
    return img


BUCKET_FN = {
    "clean": lambda img, rng: img,
    "jpeg": lambda img, rng: jpeg(img, rng.integers(30, 96), subsampling=int(rng.choice([0, 2]))),
    "webp": lambda img, rng: webp(img, rng.integers(30, 91)),
    "resize": _resize,
    "screenshot": _screenshot,
    "social": _social,
    "color": lambda img, rng: color_edit(img, rng),
    "blur_noise": _blur_noise,
    "crop": _crop,
}


def _combo(img, rng):
    names = ["jpeg", "resize", "screenshot", "color", "blur_noise", "crop", "social"]
    k = int(rng.integers(2, 4))
    for i in rng.permutation(len(names))[:k]:
        img = BUCKET_FN[names[i]](img, rng)
    return img


BUCKET_FN["combo"] = _combo


def sample_bucket(rng, buckets=TRAIN_BUCKETS):
    names = list(buckets); p = np.array([buckets[n] for n in names], dtype=np.float64); p /= p.sum()
    return names[int(rng.choice(len(names), p=p))]


def attack_train(img, rng, buckets=TRAIN_BUCKETS):
    name = sample_bucket(rng, buckets)
    return BUCKET_FN[name](img, rng), name


# ---------------------------------------------------------------- fixed evaluation conditions
def _eval_screenshot(img, rng):
    img = resize_scale(img, 0.6, Image.BILINEAR)
    img = ui_bars(img, rng, color=(30, 30, 30), top=0.08, bottom=0.06, side=0.0)
    return jpeg(img, 85)


def _eval_social(img, rng):
    img = jpeg(resize_max_side(img, 320, Image.BICUBIC), 75)
    return jpeg(resize_scale(img, 0.85, Image.BILINEAR), 72)


def _eval_color(img, rng):
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.3)
    return hue_shift(img, 0.05)


def _eval_combo(img, rng):
    img = resize_scale(img, 0.7, Image.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return jpeg(img, 70)


EVAL_CONDITIONS = {
    "clean": lambda img, rng: img,
    "jpeg60": lambda img, rng: jpeg(img, 60, subsampling=2),
    "webp50": lambda img, rng: webp(img, 50),
    "resize50": lambda img, rng: resize_scale(img, 0.5, Image.BICUBIC),
    "screenshot": _eval_screenshot,
    "social": _eval_social,
    "color": _eval_color,
    "blur1": lambda img, rng: blur(img, 1.0),
    "crop50": lambda img, rng: crop_frac(img, 0.5, rng),
    "combo": _eval_combo,
}


def attack_eval(img, name, seed):
    rng = np.random.default_rng(seed)
    return EVAL_CONDITIONS[name](img, rng)
