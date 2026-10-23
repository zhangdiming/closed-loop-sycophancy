import argparse, json, math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path.cwd()))
from bootstrap_ci import _repair_status
ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--pairs", nargs="*")
ap.add_argument("--rule", default="oracle", choices=["oracle", "agree", "either"], help="oracle only (paper), or detector re-scoring as in repair_strict.py")
a = ap.parse_args()
facts = {r["pid"]: r["facts"] for r in (json.loads(l) for l in open("data/prompts.jsonl", encoding="utf-8"))}
def _load(d, j):
    o = {}
    for f in (d / "oracle").glob(f"{j}__round*.jsonl"):
        rd = f.stem.split("__")[1]
        for r in map(json.loads, open(f)): o[(rd, r["image"], r["fid"])] = r["answer"]
    return o
def status(name):
    if a.rule == "oracle": return _repair_status(name)
    d = pathlib.Path("runs/repair") / name; q, w = _load(d, "qwen3vl8b"), _load(d, "owlv2"); out = {}
    for t in map(json.loads, open(d / "trace.jsonl", encoding="utf-8")):
        rd, im = t["rounds"][-1]["image"].split("/"); ok = True
        for fc in facts[t["pid"]]:
            x = q.get((rd, im, fc["fid"])); y = w.get((rd, im, fc["fid"]), x) if fc["type"] != "color" else x
            if x is None: ok = None; break
            bad = (x == "no" and y == "no") if a.rule == "agree" else (x == "no" or y == "no")
            ok = ok and not bad
        if ok is not None: out[t["pid"]] = [(ok, None, None)]
    return out
PAIRS = {"bagel_geneval": ("bagel__bagel", "bagel__bagel__ctx"), "bagel_erase": ("bagel__bagel_dqt", "bagel__bagel__ctx_dqt"),
         "omnigen2_erase": ("omnigen2__omnigen2_dqt", "omnigen2__omnigen2__ctx_dqt")}
def run(name, s): return name if s == 0 else f"{name}_s{s}"
def mcnemar(b, c):
    n = b + c; k = min(b, c); return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
out = {}
for key, (fr, cl) in PAIRS.items():
    if a.pairs and key not in a.pairs: continue
    rows = []; per = {}
    for s in (0, 1, 2):
        try: F, C = status(run(fr, s)), status(run(cl, s))
        except Exception as e: per[s] = f"missing: {e}"; continue
        P = sorted(set(F) & set(C))
        if len(P) < 280: per[s] = f"incomplete ({len(P)})"; continue
        f = np.array([F[p][-1][0] for p in P], float); c = np.array([C[p][-1][0] for p in P], float)
        b01, b10 = int(((f == 1) & (c == 0)).sum()), int(((f == 0) & (c == 1)).sum())
        per[s] = {"fresh": float(f.mean()), "closed": float(c.mean()), "gap": float(f.mean() - c.mean()), "disc": [b01, b10], "p": mcnemar(b01, b10)}
        rows += [(p, fi, ci) for p, fi, ci in zip(P, f, c)]
    if not rows: out[key] = {"per_seed": per}; continue
    pids = sorted({r[0] for r in rows}); idx = {p: [] for p in pids}
    for i, r in enumerate(rows): idx[r[0]].append(i)
    f = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows]); d = f - c
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        sel = np.concatenate([idx[p] for p in rng.choice(pids, len(pids))]); bs.append(d[sel].mean())
    b01, b10 = int(((f == 1) & (c == 0)).sum()), int(((f == 0) & (c == 1)).sum())
    D = np.array([d[idx[p]].sum() for p in pids]); D = D[D != 0]; obs = abs(D.sum()); NP = 1_000_000; hits = 0
    for _ in range(NP // 100_000):
        sg = rng.choice([-1.0, 1.0], size=(100_000, len(D))); hits += int((np.abs(sg @ D) >= obs - 1e-9).sum())
    p_cluster = (hits + 1) / (NP + 1)
    out[key] = {"per_seed": per, "seeds_pooled": len({0, 1, 2} & {s for s, v in per.items() if isinstance(v, dict)}),
                "pooled": {"n": len(rows), "fresh": float(f.mean()), "closed": float(c.mean()), "gap": float(d.mean()),
                           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "disc": [b01, b10], "p": mcnemar(b01, b10), "p_cluster_signflip": p_cluster, "n_prompts_nonzero": int(len(D))}}
    print(key, json.dumps(out[key], indent=None))
if a.json: json.dump(out, open(a.json, "w"), indent=1)
