import sys, collections
from bootstrap_ci import labels, load_rows, answers, ROOT
J = sys.argv[1]; L = sys.argv[2] if len(sys.argv) > 2 else "human"
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, L); ans = answers(rows)
typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
cnt = collections.defaultdict(lambda: [0, 0])
for (j, c, g), d in ans.items():
    if j != J or c not in ("fresh", "ctx"): continue
    other = ans.get((J, "ctx" if c == "fresh" else "fresh", g), {})
    for k, (pid, a) in d.items():
        key = (g, *k)
        if k not in other or o.get(key) != "no": continue
        side = "own" if g == J else "oth"; cell = cnt[(typ[key], side, c)]; cell[0] += a == "no"; cell[1] += 1
for t in ("presence", "count", "color", "position"):
    print(t, " ".join(f"{s}-{c} {cnt[(t,s,c)][0]/max(cnt[(t,s,c)][1],1):.2f}(n={cnt[(t,s,c)][1]})" for s in ("own", "oth") for c in ("fresh", "ctx")))
