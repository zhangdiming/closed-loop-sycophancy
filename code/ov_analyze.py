import argparse, collections, glob, json, numpy as np
from bootstrap_ci import load_rows, labels, ROOT
from pub_analyze import auroc

def det_at(s, t, fa=0.05):
    neg, pos = np.sort(s[t == 0]), s[t == 1]
    if len(np.unique(s)) < 5 or not len(neg) or not len(pos): return float("nan")
    return float((pos < neg[int(np.floor(fa * len(neg)))]).mean())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    facts = {r["pid"]: [f["fid"] for f in r["facts"]] for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl"))}
    lab, _ = labels([r for r in load_rows(str(ROOT / "runs/judge")) if r["judge"] in ("qwen3vl8b", "owlv2")], "human")
    nat = {}
    for f in glob.glob(str(ROOT / "runs/pub/ov_native*.jsonl")):
        for l in open(f): r = json.loads(l); nat[(r["gen"], r["image"])] = r
    atom = collections.defaultdict(dict)
    for f in glob.glob(str(ROOT / "runs/judge_ov/omniverifier7b__*.jsonl")):
        for l in open(f):
            r = json.loads(l); c = "closed" if r.get("context") else "fresh"
            atom[(r["gen"], r["image"])].setdefault(c, {})[r["fid"]] = r["p_yes"]
    res = {}; rng = np.random.default_rng(0)
    for gsel in ("bagel", "janus7b", "omnigen2", "all"):
        keys = [k for k in nat if gsel in ("all", k[0])]
        truth, pid = {}, {}
        for k in keys:
            ls = [lab.get((k[0], k[1], fid)) for fid in facts[nat[k]["pid"]]]
            if None in ls: continue
            truth[k] = int("no" in ls); pid[k] = nat[k]["pid"]
        out = {"n_images": len(truth), "n_flawed": sum(truth.values())}
        rows = {}
        nk = [k for k in truth if nat[k]["answer"] in ("true", "false")]
        rows["native"] = (nk, np.array([0.0 if nat[k]["answer"] == "false" else 1.0 for k in nk]), True)
        out["native_parsed"] = len(nk) / max(len(truth), 1)
        for c in ("fresh", "closed"):
            ck = [k for k in truth if c in atom.get(k, {}) and len(atom[k][c]) == len(facts[pid[k]])]
            rows[c] = (ck, np.array([np.mean(list(atom[k][c].values())) for k in ck]), False)
        for name, (ks, sc, binary) in rows.items():
            if not ks: continue
            t = np.array([truth[k] for k in ks])
            fl = (sc < 0.5).astype(float) if binary else np.array([float(min(atom[k][name].values()) < 0.5) for k in ks])
            pids = sorted({pid[k] for k in ks}); ix = {p: i for i, p in enumerate(pids)}; g = np.array([ix[pid[k]] for k in ks])
            W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=2000)
            bd = [((w[g] * fl * (t == 1)).sum() / max((w[g] * (t == 1)).sum(), 1)) for w in W]
            q = {"n": len(ks), "det": float(fl[t == 1].mean()), "fa": float(fl[t == 0].mean()), "det_ci": [float(x) for x in np.percentile(bd, [2.5, 97.5])],
                 "auroc": None if binary else auroc(sc[t == 1], sc[t == 0]), "det_fa5": None if binary else det_at(sc, t)}
            out[name] = q
            print(f"{gsel:9s} {name:7s} n={q['n']:4d} det {q['det']:.3f} [{q['det_ci'][0]:.2f},{q['det_ci'][1]:.2f}] FA {q['fa']:.3f}"
                  + ("" if binary else f" D@5 {q['det_fa5']:.2f} AUROC {q['auroc']:.3f}"))
        both = [k for k in rows["native"][0] if k in set(rows["fresh"][0]) and truth[k] == 1]
        if both:
            nf = {k: nat[k]["answer"] == "false" for k in both}; ff = {k: min(atom[k]["fresh"].values()) < 0.5 for k in both}
            b01 = sum(ff[k] and not nf[k] for k in both); b10 = sum(nf[k] and not ff[k] for k in both)
            from math import comb
            n = b01 + b10; pv = min(1.0, 2 * sum(comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n) if n else 1.0
            ps = sorted({pid[k] for k in both}); ix2 = {p_: i for i, p_ in enumerate(ps)}; g2 = np.array([ix2[pid[k]] for k in both])
            d = np.array([float(ff[k]) - float(nf[k]) for k in both]); W2 = rng.multinomial(len(ps), np.ones(len(ps)) / len(ps), size=2000)
            bs = [(w[g2] * d).sum() / max(w[g2].sum(), 1) for w in W2]
            out["fresh_minus_native"] = {"det_gap": float(d.mean()), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "discordant": [b01, b10], "mcnemar_p": pv}
            print(f"{gsel:9s} fresh - native detection {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] discordant {b01} vs {b10}, exact McNemar p={pv:.4f}")
        print(f"{gsel:9s} images {out['n_images']} flawed {out['n_flawed']} native parsed {out['native_parsed']:.2f}")
        res[gsel] = out
    if a.json: json.dump(res, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
