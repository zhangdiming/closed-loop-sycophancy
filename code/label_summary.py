import json, pathlib, collections
from bootstrap_ci import labels, load_rows, ROOT
rows = load_rows(ROOT / "runs/judge")
strict, _ = labels(rows, "strict")
typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
out = {}
for g in ["janus7b", "showo2_7b", "bagel", "lumina", "blip3o", "omnigen2"]:
    flags = {k for k, v in strict.items() if k[0] == g and v == "no" and typ[k] != "count"}
    rec = {"flag_strict": len(flags)}
    for tag, fp, lp in [("pass1", f"runs/human/{g}_flagged.json", f"data/human/{g}_labels.json"),
                        ("blind", f"runs/human/{g}_todo_flagged.json", f"data/human/{g}_todo_labels.json"),
                        ("blind2", f"runs/human/{g}_todo_flagged.json", f"data/human/{g}_todo2_labels.json")]:
        F = json.load(open(ROOT / fp)); L = json.load(open(ROOT / lp))["labels"] if (ROOT / lp).exists() else {}
        keys = [(g, f["image"], f["fid"]) for f in F]
        lab = {keys[int(i)]: l for i, l in L.items()}
        rec[tag] = {"items": len(F), "items_in_flags": sum(k in flags for k in keys), "labelled": len(lab),
                    "labelled_in_flags": sum(k in flags for k in lab), **collections.Counter(lab.values())}
    out[g] = rec; print(g, rec)
json.dump(out, open(ROOT / "runs/ci/label_summary.json", "w"), indent=1)
