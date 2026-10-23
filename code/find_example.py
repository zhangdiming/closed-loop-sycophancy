import json, sys, collections
from bootstrap_ci import labels, load_rows, answers, ROOT
J = sys.argv[1]
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human"); ans = answers(rows)
P = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
F = {f["fid"]: f for r in P.values() for f in r["facts"]}
insp = set()
for fn in (f"runs/human/{J}_flagged.json", f"runs/human/{J}_todo_flagged.json"):
    lab = {}
    for lf in (f"data/human/{J}_labels.json", f"data/human/{J}_todo_labels.json"):
        if fn.split('/')[-1].replace('_flagged', '') in lf.replace('_labels', ''): pass
    try:
        Fl = json.load(open(ROOT / fn)); L = json.load(open(ROOT / fn.replace("runs/human", "data/human").replace("_flagged", "_labels")))["labels"]
        insp |= {(f["image"], f["fid"]) for i, f in enumerate(Fl) if L.get(str(i)) == "violated"}
    except FileNotFoundError: pass
C = ["fresh", "ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline"]
out = []
for (img, fid) in insp:
    if F[fid]["type"] != sys.argv[2]: continue
    a = {c: ans.get((J, c, J), {}).get((img, fid), (None, None))[1] for c in C}
    if a["fresh"] == "no" and a["ctx"] == "yes":
        pid = img.rsplit("_s", 1)[0]
        others = {f["fid"]: o.get((J, img, f["fid"])) for f in P[pid]["facts"]}
        out.append((sum(v == "yes" for v in others.values()), img, P[pid]["prompt"], F[fid]["question"], a, {F[k]["question"]: v for k, v in others.items()}))
for r in sorted(out, reverse=True)[:12]: print(r)
