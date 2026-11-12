"""Plan the native-pixel re-fetch (v6): every image of every build is read again from its source so an untouched
window of the ORIGINAL pixels can be stored next to the 512-px overview.

  python scripts/refetch_plan.py --manifests 'runs/dl/manifests/*/manifest_*.jsonl' --parts 5 --out basma/refetch_parts.json

Groups the manifest rows by (source, unit), measures the bytes each unit needs (parquet row-group sizes from the
footers, file sizes for tar / arrow, the whole archive for zip and per-file sources whose manifests only record a
folder), checks that every source still exists, and splits the work into `parts` Kaggle kernels of equal bytes."""
import argparse
import collections
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import basma.build_data as B  # noqa: E402
from basma import sources as S  # noqa: E402

WHOLE = (S.ZipSource, S.HfFilesSource)          # their manifest unit is a folder: the whole source is scanned


def registry():
    """source name -> Source object (first plan that defines it; the same name always means the same repo)."""
    reg = {}
    for pn in ["cf", "recent", "reals", "eval", "v3ai", "v3real", "v4a", "v4b", "v5a", "v5b", "v5c", "v5d", "v5e"]:
        for src, _ in B.PLANS[pn](1.0):
            reg.setdefault(src.name, src)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True); ap.add_argument("--parts", type=int, default=5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = registry()
    units = collections.defaultdict(set); n_rows = collections.Counter(); n_dup = 0; seen = set()
    for p in sorted(glob.glob(a.manifests, recursive=True)):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in seen:
                n_dup += 1; continue
            seen.add(r["id"]); units[r["source"]].add(str(r.get("unit", ""))); n_rows[r["source"]] += 1
    missing = [s for s in units if s not in reg]
    if missing:
        raise SystemExit(f"sources without a reader: {missing}")
    print(f"{len(seen)} images ({n_dup} duplicate rows skipped) from {len(units)} sources", flush=True)
    fs = S.hffs(); api = S.HfApi()
    items = []                      # (source, kind, unit, bytes)
    def pq_size(repo, f):
        with fs.open(f"datasets/{repo}/{f}", "rb", block_size=1 << 18) as fh:
            md = pq.ParquetFile(fh).metadata
        return {rg: sum(md.row_group(rg).column(c).total_compressed_size for c in range(md.num_columns)) for rg in range(md.num_row_groups)}
    def file_size(repo, f):
        return fs.info(f"datasets/{repo}/{f}")["size"]
    jobs = []
    for name, us in sorted(units.items()):
        src = reg[name]
        if isinstance(src, S.ParquetSource):
            by_file = collections.defaultdict(list)
            for u in us:
                f, rg = u.rsplit("#", 1); by_file[f].append(int(rg))
            for f, rgs in by_file.items():
                jobs.append((name, "parquet", f, rgs))
        elif isinstance(src, (S.TarSource, S.ArrowSource)):
            for u in us:
                jobs.append((name, "file", u if isinstance(u, str) else u[1], None))
        elif isinstance(src, WHOLE):
            jobs.append((name, "whole", "", None))
        else:
            raise SystemExit(f"unknown source type for {name}: {type(src).__name__}")
    print(f"{len(jobs)} size lookups", flush=True)
    def measure(job):
        name, kind, f, rgs = job; src = reg[name]
        try:
            if kind == "parquet":
                sz = pq_size(src.repo, f); return [(name, "parquet", f"{f}#{rg}", sz.get(rg, 0)) for rg in rgs]
            if kind == "file":
                if isinstance(src, S.ArrowSource):
                    return [(name, "arrow", f, file_size(src.repo, f))]
                return [(name, "tar", f, min(file_size(src.repo, f), src.per_shard * 400_000))]   # streamed: first per_shard images only
            if isinstance(src, S.GDriveZipSource):
                return [(name, "whole", "", 6_400_000_000)]
            if isinstance(src, S.ZipSource):
                return [(name, "whole", "", file_size(src.repo, src.zip_path))]
            files = [x for x in S.list_files(src.repo, prefix=src.prefix) if x.lower().endswith(src.exts)]
            return [(name, "whole", "", len(files) * 250_000)]
        except Exception as e:  # noqa: BLE001
            return [(name, "ERROR", f, f"{type(e).__name__}: {str(e)[:100]}")]
    with ThreadPoolExecutor(24) as ex:
        for i, res in enumerate(ex.map(measure, jobs)):
            items += res
            if i % 200 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    errors = [it for it in items if it[1] == "ERROR"]
    for it in errors[:20]: print("ERROR", it)
    items = [it for it in items if it[1] != "ERROR"]
    by_src = collections.defaultdict(int)
    for name, kind, u, b in items: by_src[name] += b
    print("\nbytes to read per source (GB):")
    for name, b in sorted(by_src.items(), key=lambda kv: -kv[1]): print(f"  {name:18} {b/1e9:7.1f}  rows {n_rows[name]}")
    total = sum(by_src.values()); print(f"TOTAL {total/1e9:.0f} GB, errors {len(errors)}")
    # greedy split: biggest items first into the lightest part; a 'whole' source is one item
    parts = {chr(97 + i): {"bytes": 0, "items": []} for i in range(a.parts)}
    for name, kind, u, b in sorted(items, key=lambda it: -it[3]):
        k = min(parts, key=lambda k: parts[k]["bytes"]); parts[k]["bytes"] += b; parts[k]["items"].append([name, kind, u])
    for k, v in parts.items(): print(f"part {k}: {v['bytes']/1e9:.1f} GB, {len(v['items'])} items, sources {len({it[0] for it in v['items']})}")
    json.dump({"images": len(seen), "sources": {n: {"rows": n_rows[n], "bytes": by_src[n]} for n in units}, "errors": errors, "parts": parts}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
