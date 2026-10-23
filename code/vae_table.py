import json
from bootstrap_ci import labels, load_rows, answers, ROOT
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human"); ans = answers(rows); out = {}
for J in ("bagel", "bagel_vae"):
    r = {}
    for side, gens in (("own", ["bagel"]), ("oth", ["janus7b", "showo2_7b"])):
        for c in ("fresh", "ctx"):
            h = n = fh = fn = 0
            for g in gens:
                d = ans.get((J, c, g), {}); d2 = ans.get((J, "ctx" if c == "fresh" else "fresh", g), {})
                for k, (pid, a) in d.items():
                    if k not in d2: continue
                    l = o.get((g, *k))
                    if l == "no": n += 1; h += a == "no"
                    elif l == "yes": fn += 1; fh += a == "no"
            r[f"{side}_{c}"] = h / max(n, 1); r[f"fa_{side}_{c}"] = fh / max(fn, 1); r[f"n_{side}"] = n
    out[J] = r; print(J, {k: round(v, 3) for k, v in r.items()})
json.dump(out, open(ROOT / "runs/ci/bagel_vae_table.json", "w"), indent=1)
