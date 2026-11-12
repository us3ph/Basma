"""Python side of the resample/views parity test: random + real images -> base and views (hashes + a few values)."""
import json, hashlib, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/x-hunter/Basma")
from basma.resample import to_base
from basma.views import inference_views
rs = np.random.RandomState(0)
imgs = {"rand_700x1100": (rs.rand(700, 1100, 3) * 255).astype(np.uint8), "rand_150x200": (rs.rand(150, 200, 3) * 255).astype(np.uint8),
        "rand_2300x1700": (rs.rand(2300, 1700, 3) * 255).astype(np.uint8), "rand_224x224": (rs.rand(224, 224, 3) * 255).astype(np.uint8)}
for p in sorted(glob.glob("/home/x-hunter/Basma/data/img/*.jpg"))[:3]:
    imgs[p.split("/")[-1]] = np.asarray(Image.open(p).convert("RGB"))
out = []
for name, img in imgs.items():
    base = to_base(img); views, info = inference_views(base, img if img.shape != base.shape else None)   # v6: crops from the image's own pixels
    out.append({"name": name, "h": img.shape[0], "w": img.shape[1], "pixels_b64": __import__("base64").b64encode(img.tobytes()).decode(),
                "base_h": base.shape[0], "base_w": base.shape[1], "base_sha": hashlib.sha1(base.tobytes()).hexdigest(),
                "n_views": int(views.shape[0]), "views_sha": hashlib.sha1(views.tobytes()).hexdigest(), "views_sum": float(views.astype(np.float64).sum()),
                "views_first": [float(v) for v in views.reshape(-1)[:5]]})
json.dump(out, open("/home/x-hunter/Basma/web/test/vectors_views.json", "w"))
print("wrote", len(out), "vectors")
