import json, sys, math, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
key = json.load(open(ROOT / "data/audit/key.json")); lab = json.load(open(sys.argv[1]))["audit"]["labels"]
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
L = {it["idx"]: lab.get(str(it["idx"])) for it in key}
out = {"n_labelled": sum(v is not None for v in L.values()), "n_items": len(key)}
for t in ("all", "presence", "count", "color", "position"):
    s = [it for it in key if it["set"] == "S" and (t == "all" or it["type"] == t) and L[it["idx"]] in ("violated", "satisfied")]
    k = sum(L[it["idx"]] == "violated" for it in s); out[f"S_miss_{t}"] = (k, len(s)) + wilson(k, len(s))[1:] + (wilson(k, len(s))[0],)
c = [it for it in key if it["set"] == "C" and L[it["idx"]] in ("violated", "satisfied")]
k = sum(L[it["idx"]] == "violated" for it in c); out["C_confirm"] = (k, len(c), wilson(k, len(c)))
out["ambiguous"] = collections.Counter(it["set"] for it in key if L[it["idx"]] == "ambiguous")
R = [it for it in key if it["set"] == "R"]; agree = [(it["oracle"] == "yes") == (L[it["idx"]] == "satisfied") for it in R if L[it["idx"]] in ("violated", "satisfied") and it["oracle"]]
out["R_fact_agreement"] = (sum(agree), len(agree))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for it in R: per[it["pid"]][it["run"]].append(L[it["idx"]])
ok = {}
for pid, runs in per.items():
    if any(x in (None, "ambiguous") for v in runs.values() for x in v): continue
    ok[pid] = {r: all(x == "satisfied" for x in v) for r, v in runs.items()}
f = sum(v["bagel__bagel"] for v in ok.values()); cl = sum(v["bagel__bagel__ctx"] for v in ok.values())
b01 = sum(v["bagel__bagel"] and not v["bagel__bagel__ctx"] for v in ok.values()); b10 = sum(v["bagel__bagel__ctx"] and not v["bagel__bagel"] for v in ok.values())
nn = b01 + b10; kk = min(b01, b10); p = min(1.0, 2 * sum(math.comb(nn, j) for j in range(kk + 1)) / 2 ** nn) if nn else 1.0
out["R_prompts"] = {"usable": len(ok), "fresh_correct": f, "closed_correct": cl, "fresh_only": b01, "closed_only": b10, "mcnemar_p": p}
print(json.dumps(out, indent=1, default=str)); json.dump(out, open(ROOT / "runs/ci/audit.json", "w"), indent=1, default=str)
