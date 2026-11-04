"""report.json -> Markdown tables (docs/RESULTS.md). python -m basma.report_md runs/x/report/report.json"""
import json
import sys


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(path):
    rep = json.load(open(path))
    b = rep["basma"]; systems = [("Basma", b)] + [(n, r) for n, r in rep["baselines"].items()]
    md = ["# Basma results", "",
          f"Thresholds for every system are set the same way: the score above which at most 1 % of the validation real images "
          f"(train sources, all conditions pooled) fall. Basma's threshold is on its calibrated probability ({b['threshold']:.3f}); "
          f"baselines use their own raw scores. TPR = share of AI images above the threshold, FPR = share of real images above it. "
          f"Validation: {b['n_val']} rows, evaluation: {b['n_eval']} rows (image x condition).", ""]
    md += ["## Held-out pools (never trained on)", ""]
    rows = []
    for sp in sorted(b["eval_by_split"]):
        for name, r in systems:
            d = r["eval_by_split"].get(sp, {})
            rows.append([sp if name == "Basma" else "", name, d.get("n_ai", 0), d.get("n_real", 0), pct(d.get("tpr")), pct(d.get("fpr")), pct(d.get("auroc"))])
    md += [table(["pool", "system", "AI rows", "real rows", "TPR %", "FPR %", "AUROC %"], rows), ""]
    md += ["## By condition (held-out pools, all images)", ""]
    rows = []
    for c in b["eval_by_cond"]:
        for name, r in systems:
            d = r["eval_by_cond"].get(c, {})
            rows.append([c if name == "Basma" else "", name, pct(d.get("tpr")), pct(d.get("fpr")), pct(d.get("auroc"))])
    md += [table(["condition", "system", "TPR %", "FPR %", "AUROC %"], rows), ""]
    md += ["## AI generator families (held-out pools), TPR % per system", ""]
    fams = sorted(b["eval_by_family"], key=lambda f: -b["eval_by_family"][f]["n"])
    rows = [[f, b["eval_by_family"][f]["n"]] + [pct(r["eval_by_family"].get(f, {}).get("tpr")) for _, r in systems] for f in fams]
    md += [table(["family", "rows", *[n for n, _ in systems]], rows), ""]
    md += ["## Real sources (held-out pools), FPR % per system", ""]
    srcs = sorted(b["eval_real_by_source"], key=lambda f: -b["eval_real_by_source"][f]["n"])
    rows = [[s, b["eval_real_by_source"][s]["n"]] + [pct(r["eval_real_by_source"].get(s, {}).get("fpr")) for _, r in systems] for s in srcs]
    md += [table(["source", "rows", *[n for n, _ in systems]], rows), ""]
    md += ["## Pool x condition (Basma TPR % / FPR %)", ""]
    conds = list(b["eval_by_cond"])
    rows = []
    for sp, d in sorted(b["eval_split_x_cond"].items()):
        rows.append([sp] + [f"{pct(d.get(c, {}).get('tpr'))} / {pct(d.get(c, {}).get('fpr'))}" for c in conds])
    md += [table(["pool", *conds], rows), ""]
    md += ["## Validation (train sources, 8 % held out by hash), Basma", ""]
    rows = [[c, pct(d.get("tpr")), pct(d.get("fpr")), pct(d.get("auroc"))] for c, d in b["val_by_cond"].items()]
    md += [table(["condition", "TPR %", "FPR %", "AUROC %"], rows), ""]
    if "verdicts" in b:
        md += ["## Verdicts on the held-out pools", "", table(["label:verdict", "rows"], sorted(b["verdicts"].items())), ""]
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
