"""Generator name canonicalisation and the train / held-out split by vendor family.

Every AI image gets (gen, family, vendor). Families in HOLDOUT_FAMILIES are never trained on: they are the
"unseen vendor" test. Real images get family "real:<source>"."""
import re

# vendor families the detector never sees in training (unseen-vendor evaluation)
HOLDOUT_FAMILIES = ("seedream", "hunyuan", "recraft", "reve", "grok", "qwen", "glm", "ernie", "mai", "muse")

_RULES = [  # (regex on the lower-cased name, family, vendor)
    (r"seedream|seedance|doubao|jimeng|dreamina|bytedance", "seedream", "bytedance"),   # whole vendor held out
    (r"hunyuan", "hunyuan", "tencent"),
    (r"recraft", "recraft", "recraft"),
    (r"halfmoon|reve", "reve", "reve"),
    (r"aurora|grok", "grok", "xai"),
    (r"qwen", "qwen", "alibaba"),
    (r"glm|cogview", "glm", "zhipu"),
    (r"ernie", "ernie", "baidu"),
    (r"mai[- ]image", "mai", "microsoft"),
    (r"\bmuse\b", "muse", "muse"),
    (r"pruna", "pruna", "pruna"),
    (r"dall-?e-?2|unclip|karlo", "unclip", "openai"),            # pixel-space diffusion (v4)
    (r"glide", "glide", "openai"),
    (r"\badm\b|guided[- ]diffusion", "adm", "openai"),
    (r"rddm|ddpm|pixel[- ]diffusion", "pixel-diffusion", "research"),
    (r"firefly|adobe", "firefly", "adobe"),
    (r"vqgan|vq-?diffusion|vqdm", "vq", "research"),
    (r"stylegan|sgxl|gan(\d|\b|_|-|$)", "gan", "research"),
    (r"latent[_ -]diffusion|\bldm\b", "ldm", "compvis"),
    (r"\b(dit|sit)\d*k?\b", "dit", "research"),
    (r"^reddit( video frame)?$", "wild-reddit", "unknown"),                       # in-the-wild posts, generator unknown
    (r"flux|blueberry", "flux", "bfl"),
    (r"dall|gpt|4o|openai|sora", "openai", "openai"),
    (r"imagen|banana|google|\bveo", "google", "google"),
    (r"midjourney|\bmj\b", "midjourney", "midjourney"),
    (r"stable|sdxl|sd3|sd35|sd21|sd15|sd14|sd_|^sd\b|stability", "stability", "stability"),
    (r"hidream", "hidream", "hidream"),
    (r"ideogram", "ideogram", "ideogram"),
    (r"frames|runway", "runway", "runway"),
    (r"lumina", "lumina", "opengvlab"),
    (r"janus", "janus", "deepseek"),
    (r"\bwan\b|wan-|wan2|wan 2", "wan", "alibaba"),
    (r"photon|luma", "luma", "luma"),
    (r"minimax|image-01|hailuo", "minimax", "minimax"),
    (r"z[- ]?image", "zimage", "alibaba"),
    (r"kling", "kling", "kuaishou"),
    (r"ltx|lightricks", "ltx", "lightricks"),
    (r"playground", "playground", "playground"),
    (r"kandinsky", "kandinsky", "sber"),
    (r"pixart|sana\b", "pixart", "nvidia-huawei"),
    (r"chroma", "chroma", "lodestones"),
    (r"bria", "bria", "bria"),
    (r"kolors", "kolors", "kuaishou"),
]


def canon(name):
    """Raw generator name -> (gen, family, vendor). gen is a compact slug of the raw name."""
    raw = (name or "").strip()
    low = raw.lower()
    slug = re.sub(r"[^a-z0-9.]+", "-", low).strip("-")[:60] or "unknown"
    for pat, fam, vendor in _RULES:
        if re.search(pat, low):
            return slug, fam, vendor
    return slug, "other", "other"


def is_holdout(family):
    return family in HOLDOUT_FAMILIES
