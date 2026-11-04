"""Generator name canonicalisation and the train / held-out split by vendor family.

Every AI image gets (gen, family, vendor). Families in HOLDOUT_FAMILIES are never trained on: they are the
"unseen vendor" test. Real images get family "real:<source>"."""
import re

# vendor families the detector never sees in training (unseen-vendor evaluation)
HOLDOUT_FAMILIES = ("seedream", "hunyuan", "recraft", "reve", "grok", "qwen", "glm", "ernie", "mai", "muse")

_RULES = [  # (regex on the lower-cased name, family, vendor)
    (r"seedream", "seedream", "bytedance"),
    (r"hunyuan", "hunyuan", "tencent"),
    (r"recraft", "recraft", "recraft"),
    (r"halfmoon|reve", "reve", "reve"),
    (r"aurora|grok", "grok", "xai"),
    (r"qwen", "qwen", "alibaba"),
    (r"glm", "glm", "zhipu"),
    (r"ernie", "ernie", "baidu"),
    (r"mai[- ]image", "mai", "microsoft"),
    (r"\bmuse\b", "muse", "muse"),
    (r"pruna", "pruna", "pruna"),
    (r"flux|blueberry", "flux", "bfl"),
    (r"dall|gpt|4o|openai", "openai", "openai"),
    (r"imagen|banana|google", "google", "google"),
    (r"midjourney|\bmj\b", "midjourney", "midjourney"),
    (r"stable|sdxl|sd3|sd35|sd21|sd15|sd14|sd_|^sd\b|stability", "stability", "stability"),
    (r"hidream", "hidream", "hidream"),
    (r"ideogram", "ideogram", "ideogram"),
    (r"frames|runway", "runway", "runway"),
    (r"lumina", "lumina", "opengvlab"),
    (r"janus", "janus", "deepseek"),
    (r"\bwan\b|wan-|wan2|wan 2", "wan", "alibaba"),
    (r"photon|luma", "luma", "luma"),
    (r"minimax|image-01", "minimax", "minimax"),
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
