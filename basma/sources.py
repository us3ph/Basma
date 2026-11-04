"""Streaming readers for every data source (Hugging Face parquet / zip / files / arrow, Google Drive zip).

A source lists *units* (a parquet row group, a chunk of zip members, a chunk of files, one arrow shard) and reads
one unit at a time into (raw_bytes, meta) items, so a data build downloads only what it keeps."""
import io
import os
import json
import random
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

_FS = None


def hffs():
    global _FS
    if _FS is None:
        _FS = HfFileSystem()
    return _FS


def list_files(repo, suffix="", prefix=""):
    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(suffix) and f.startswith(prefix))


def _bytes(v):
    """Image column value -> bytes (datasets Image struct {bytes, path} or plain binary)."""
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("bytes")
    return bytes(v)


class Source:
    """name: short id; label: 0 real / 1 AI / None (per row); split: 'train' (val carved out by hash) or an eval
    pool name 'eval:<pool>'; gen: constant generator name or None (per row)."""
    name = ""
    label = None
    split = "train"
    gen = None
    family_override = None   # e.g. "real:coco" for real sources, "cf" for Community Forensics

    def units(self, rng, budget):
        raise NotImplementedError

    def read(self, unit):
        raise NotImplementedError


# ---------------------------------------------------------------- parquet (row groups)
class ParquetSource(Source):
    def __init__(self, name, repo, image_cols, label=None, split="train", gen=None, meta_cols=(), files=None,
                 file_filter=None, row_meta=None, family_override=None):
        """image_cols: list of (column, label_or_None, gen_or_None); row_meta(row) -> dict with 'label'/'gen'/'extra'
        overrides for per-row labels (Community Forensics, Rapidata)."""
        self.name, self.repo, self.image_cols = name, repo, image_cols
        self.label, self.split, self.gen, self.meta_cols = label, split, gen, list(meta_cols)
        self._files, self.file_filter, self.row_meta, self.family_override = files, file_filter, row_meta, family_override

    def files(self):
        if self._files is None:
            self._files = [f for f in list_files(self.repo, ".parquet") if (self.file_filter is None or self.file_filter(f))]
        return self._files

    def units(self, rng, budget):
        """budget: dict(max_files, rgs_per_file, max_unit_mb, keep_big). Row-group counts/sizes come from the footers."""
        files = list(self.files())
        rng.shuffle(files)
        files = files[:budget.get("max_files", len(files))]
        out, big = [], []
        for f in files:
            with hffs().open(f"datasets/{self.repo}/{f}", "rb") as fh:
                md = pq.ParquetFile(fh).metadata
            rgs = [(rg, md.row_group(rg).total_byte_size) for rg in range(md.num_row_groups)]
            rng.shuffle(rgs)
            for rg, nbytes in rgs[:budget.get("rgs_per_file", md.num_row_groups)]:
                (big if nbytes > budget.get("max_unit_mb", 1e12) * 1e6 else out).append((f, rg))
        rng.shuffle(out); rng.shuffle(big)
        out += big[:budget.get("keep_big", 0)]
        rng.shuffle(out)
        return out

    def read(self, unit):
        f, rg = unit
        cols = [c for c, _, _ in self.image_cols] + self.meta_cols
        with hffs().open(f"datasets/{self.repo}/{f}", "rb", block_size=16 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=64, row_groups=[rg], columns=cols):
                for row in batch.to_pylist():
                    for col, lab, gen in self.image_cols:
                        b = _bytes(row.get(col))
                        if not b:
                            continue
                        meta = {"label": self.label if lab is None else lab, "gen": self.gen if gen is None else gen, "unit": f"{f}#{rg}"}
                        if self.row_meta is not None:
                            meta.update(self.row_meta(row, col))
                        yield b, meta


# ---------------------------------------------------------------- zip members (HF-hosted or local)
class ZipSource(Source):
    def __init__(self, name, label, split="train", gen=None, repo=None, zip_path=None, local_path=None, member_filter=None,
                 chunk=48, member_meta=None, family_override=None):
        self.name, self.label, self.split, self.gen = name, label, split, gen
        self.repo, self.zip_path, self.local_path, self.member_filter, self.chunk = repo, zip_path, local_path, member_filter, chunk
        self.member_meta, self.family_override = member_meta, family_override

    def _open(self):
        if self.local_path:
            return open(self.local_path, "rb")
        return hffs().open(f"datasets/{self.repo}/{self.zip_path}", "rb", block_size=4 * 1024 * 1024)

    def units(self, rng, budget):
        with self._open() as fh:
            z = zipfile.ZipFile(fh)
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and "__MACOSX" not in n
                     and (self.member_filter is None or self.member_filter(n))]
        rng.shuffle(names)
        names = names[:budget.get("max_images", len(names))]
        return [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]

    def read(self, unit):
        with self._open() as fh:
            z = zipfile.ZipFile(fh)
            for n in unit:
                meta = {"label": self.label, "gen": self.gen, "unit": n.rsplit("/", 1)[0]}
                if self.member_meta is not None:
                    meta.update(self.member_meta(n))
                yield z.read(n), meta


# ---------------------------------------------------------------- individual files in an HF dataset repo
class HfFilesSource(Source):
    def __init__(self, name, repo, label, gen, split="train", prefix="", exts=(".jpg", ".jpeg", ".png", ".webp"), chunk=64, family_override=None):
        self.name, self.repo, self.label, self.gen, self.split = name, repo, label, gen, split
        self.prefix, self.exts, self.chunk, self.family_override = prefix, exts, chunk, family_override

    def units(self, rng, budget):
        files = [f for f in list_files(self.repo, prefix=self.prefix) if f.lower().endswith(self.exts)]
        rng.shuffle(files)
        files = files[:budget.get("max_images", len(files))]
        return [files[i:i + self.chunk] for i in range(0, len(files), self.chunk)]

    def read(self, unit):
        def get(f):
            p = hf_hub_download(self.repo, f, repo_type="dataset")
            with open(p, "rb") as fh:
                b = fh.read()
            try:
                os.remove(p)
            except OSError:
                pass
            return f, b
        with ThreadPoolExecutor(8) as ex:
            for f, b in ex.map(get, unit):
                yield b, {"label": self.label, "gen": self.gen, "unit": f.rsplit("/", 1)[0]}


# ---------------------------------------------------------------- datasets save_to_disk arrow shards (GenImage mirror)
class ArrowSource(Source):
    def __init__(self, name, repo, dirs, split="eval:genimage", files_per_dir=1, family_override=None):
        self.name, self.repo, self.dirs, self.split, self.files_per_dir, self.family_override = name, repo, dirs, split, files_per_dir, family_override
        self.label, self.gen = None, None

    def units(self, rng, budget):
        out = []
        for d in self.dirs:
            files = [f for f in list_files(self.repo, ".arrow", prefix=d + "/")]
            rng.shuffle(files)
            out += [(d, f) for f in files[:self.files_per_dir]]
        return out

    def read(self, unit):
        d, f = unit
        info = json.load(open(hf_hub_download(self.repo, f"{d}/dataset_info.json", repo_type="dataset")))
        feats = info["features"]
        lab_names = feats.get("label", {}).get("names"); gen_names = feats.get("generator", {}).get("names")
        p = hf_hub_download(self.repo, f, repo_type="dataset")
        with pa.memory_map(p, "r") as mm:
            reader = pa.ipc.open_stream(mm)
            for batch in reader:
                cols = batch.to_pydict()
                for i in range(batch.num_rows):
                    lab = cols["label"][i]; lab = lab_names[lab] if lab_names and isinstance(lab, int) else lab
                    gen = cols.get("generator", [None] * batch.num_rows)[i]; gen = gen_names[gen] if gen_names and isinstance(gen, int) else gen
                    b = _bytes(cols["image"][i])
                    if not b:
                        continue
                    fake = str(lab).lower() in ("fake", "1", "ai")
                    yield b, {"label": 1 if fake else 0, "gen": f"genimage:{gen}" if fake else f"genimage-real:{gen}", "unit": f}
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------- Google Drive zip (WildRF)
def gdrive_download(file_id, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        return dest
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    subprocess.run(["curl", "-L", "-sS", "--retry", "5", "-o", dest, url], check=True)
    return dest


class GDriveZipSource(ZipSource):
    def __init__(self, name, file_id, local_path, **kw):
        super().__init__(name, local_path=local_path, **kw)
        self.file_id = file_id

    def _open(self):
        gdrive_download(self.file_id, self.local_path)
        return open(self.local_path, "rb")


# ---------------------------------------------------------------- per-row meta helpers
def cf_row_meta(row, col):
    """Community Forensics rows: label 0/1, model_name for AI, real_source for real."""
    lab = int(row.get("label", 0))
    gen = row.get("model_name") if lab == 1 else f"real:{row.get('real_source') or 'unknown'}"
    return {"label": lab, "gen": gen, "extra": {"arch": row.get("architecture"), "fmt": row.get("format"), "subset": row.get("subset")}}


def rapidata_row_meta(row, col):
    """Rapidata pairs: image1 <-> model1, image2 <-> model2."""
    return {"label": 1, "gen": row.get("model1") if col == "image1" else row.get("model2")}
