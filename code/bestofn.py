import collections, glob, json, pathlib, sys
from math import comb
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent

def load(judge, gen, ctx):
    f = ROOT / "runs/judge" / (f"{judge}__{gen}.jsonl" if ctx == "fresh" else f"{judge}__{gen}__ctx.jsonl")
    d = collections.defaultdict(dict)
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); d[r["image"]][r["fid"]] = r
    return d

def main():
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    rng = np.random.default_rng(0); res = {}
    for g in ("bagel", "janus7b", "showo2_7b"):
        ora = load("qwen3vl8b", g, "fresh"); J = {c: load(g, g, c) for c in ("fresh", "closed")}
        by = collections.defaultdict(list)
        for im in ora: by[im.rsplit("_s", 1)[0]].append(im)
        P, ok = [], collections.defaultdict(list)
        for pid, ims in sorted(by.items()):
            ims = sorted(ims, key=lambda x: int(x.rsplit("_s", 1)[1].split(".")[0]))
            fids = set.intersection(*[set(ora[i]) for i in ims] + [set(J[c].get(i, {})) for c in J for i in ims])
            if not fids or len(ims) < 2: continue
            good = [all(ora[i][f]["answer"] == "yes" for f in fids) for i in ims]
            P.append(pid); ok["random"].append(np.mean(good)); ok["oracle_best"].append(float(any(good)))
            for c in J:
                cnt = [sum(J[c][i][f]["answer"] == "yes" for f in fids) for i in ims]
                prb = [np.mean([J[c][i][f]["p_yes"] for f in fids]) for i in ims]
                ok[f"{c}_counted"].append(float(good[int(np.argmax(cnt))])); ok[f"{c}_prob"].append(float(good[int(np.argmax(prb))]))
                ok[f"{c}_tie"].append(float(cnt.count(max(cnt)) > 1))
        A = {k: np.array(v) for k, v in ok.items()}; n = len(P); idx = rng.integers(0, n, (2000, n))
        r = {"n_prompts": n, "n_seeds": len(by[P[0]])}
        print(f"\n== {g}: {n} prompts, {r['n_seeds']} seeds")
        for k in ("random", "oracle_best", "fresh_counted", "closed_counted", "fresh_prob", "closed_prob", "fresh_tie", "closed_tie"):
            r[k] = float(A[k].mean()); print(f"  {k:15s} {A[k].mean():.3f}")
        for rule in ("counted", "prob"):
            f_, c_ = A[f"fresh_{rule}"], A[f"closed_{rule}"]; d = f_ - c_; bd = d[idx].mean(1)
            b01, b10 = int(((f_ == 1) & (c_ == 0)).sum()), int(((f_ == 0) & (c_ == 1)).sum()); m = b01 + b10
            p = min(1.0, 2 * sum(comb(m, j) for j in range(min(b01, b10) + 1)) / 2 ** m) if m else 1.0
            r[f"gap_{rule}"] = {"est": float(d.mean()), "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], "discordant": [b01, b10], "mcnemar_p": p}
            print(f"  fresh-closed {rule:7s} {d.mean():+.3f} [{np.percentile(bd, 2.5):+.3f},{np.percentile(bd, 97.5):+.3f}]  {b01} vs {b10}  p={p:.4f}")
        res[g] = r
    if out: json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
