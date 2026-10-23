import collections, json, pathlib, sys
from math import comb
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ["base", "fresh", "closed", "fresh_sub"]
if "--models" in sys.argv: MODELS = sys.argv[sys.argv.index("--models") + 1].split(",")

def main():
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    P = {r["pid"]: r for r in map(json.loads, open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    ok, acc = {}, {}
    for m in MODELS:
        O = collections.defaultdict(dict)
        for r in map(json.loads, open(ROOT / f"runs/dpo/eval_{m}/oracle.jsonl")): O[r["image"]][r["fid"]] = r["answer"]
        ok[m], acc[m] = {}, {}
        for im, d in O.items():
            pid = im.rsplit("_s", 1)[0]; fids = [f["fid"] for f in P[pid]["facts"]]
            if not all(f in d for f in fids): continue
            ok[m][im] = float(all(d[f] == "yes" for f in fids)); acc[m][im] = np.mean([d[f] == "yes" for f in fids])
    common = sorted(set.intersection(*[set(ok[m]) for m in MODELS]))
    pids = sorted({im.rsplit("_s", 1)[0] for im in common}); ix = {p: i for i, p in enumerate(pids)}
    g = np.array([ix[im.rsplit("_s", 1)[0]] for im in common]); rng = np.random.default_rng(0)
    W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=2000).astype(float)
    N = np.bincount(g, minlength=len(pids)).astype(float)
    res = {"n_images": len(common), "n_prompts": len(pids)}
    def boot(x):
        H = np.bincount(g, weights=x, minlength=len(pids)); return (W @ H) / (W @ N)
    print(f"{len(common)} images, {len(pids)} prompts")
    for m in MODELS:
        x = np.array([ok[m][i] for i in common]); a = np.array([acc[m][i] for i in common]); b = boot(x)
        res[m] = {"all_ok": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "fact_acc": float(a.mean()),
                  "by_tag": {t: float(np.mean([ok[m][i] for i in common if P[i.rsplit('_s', 1)[0]]["tag"] == t])) for t in sorted({P[p]["tag"] for p in pids})}}
        print(f"  {m:10s} all-ok {x.mean():.3f} [{res[m]['ci95'][0]:.3f},{res[m]['ci95'][1]:.3f}]  fact-acc {a.mean():.3f}  " +
              " ".join(f"{t[:6]}={v:.2f}" for t, v in res[m]["by_tag"].items()))
    if "fresh_s1" in MODELS and "closed_s1" in MODELS:
        for k, ms in (("fresh_2s", ("fresh", "fresh_s1")), ("closed_2s", ("closed", "closed_s1"))):
            ok[k] = {i: np.mean([ok[m][i] for m in ms]) for i in common}
    pairs = [("fresh", "base"), ("closed", "base"), ("fresh_sub", "base"), ("fresh", "closed"), ("fresh_sub", "closed"), ("fresh", "fresh_sub"),
             ("fresh_s1", "closed_s1"), ("fresh_s1", "base"), ("closed_s1", "base"), ("fresh_2s", "closed_2s"), ("fresh_2s", "base"), ("closed_2s", "base")]
    for m1, m2 in [q for q in pairs if q[0] in ok and q[1] in ok]:
        x1 = np.array([ok[m1][i] for i in common]); x2 = np.array([ok[m2][i] for i in common]); d = x1 - x2; b = boot(d)
        b01, b10 = int((x1 > x2).sum()), int((x1 < x2).sum()); n = b01 + b10
        p = min(1.0, 2 * sum(comb(n, j) for j in range(min(b01, b10) + 1)) / 2 ** n) if n else 1.0
        res[f"{m1}-{m2}"] = {"est": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "discordant": [b01, b10], "mcnemar_p": p}
        print(f"  {m1}-{m2}: {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]  {b01} vs {b10}  p={p:.4f}")
    if out: json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
