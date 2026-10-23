import collections, glob, json, sys, numpy as np

def load(d):
    rows = {}
    for f in glob.glob(f"{d}/mask_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            rows[(r["judge"], r["gen"], r["image"], r["fid"], r["ctx"], r["mask"], r["layers"])] = r
    return list(rows.values())

def main():
    d = sys.argv[1]; out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    rows = load(d); res = {}
    rng = np.random.default_rng(0)
    for J in sorted({r["judge"] for r in rows}):
        own = {"janus7b": "janus7b", "janus1b": "janus1b", "omnigen2": "omnigen2", "bagel": "bagel"}[J]
        for which in ("own", "others"):
            R = [r for r in rows if r["judge"] == J and ((r["gen"] == own) == (which == "own"))]
            keys = sorted({(r["ctx"], r["mask"], r["layers"]) for r in R})
            items = sorted({(r["gen"], r["image"], r["fid"]) for r in R if r["label"] == "violated"})
            sat = sorted({(r["gen"], r["image"], r["fid"]) for r in R if r["label"] == "satisfied"})
            P = {(r["gen"], r["image"], r["fid"], r["ctx"], r["mask"], r["layers"]): r["p_yes"] for r in R}
            pid = {(r["gen"], r["image"], r["fid"]): r["pid"] for r in R}
            full = [k for k in items if all((*k, *c) in P for c in keys)]
            if not full: continue
            pids = sorted({pid[k] for k in full}); ix = {p: i for i, p in enumerate(pids)}
            M = np.array([[P[(*k, *c)] < 0.5 for c in keys] for k in full], float)
            g = np.array([ix[pid[k]] for k in full], int)
            H = np.zeros((len(pids), len(keys))); N = np.zeros(len(pids))
            np.add.at(H, g, M); np.add.at(N, g, 1)
            W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=2000).astype(float)
            det = H.sum(0) / N.sum(); bdet = (W @ H) / (W @ N)[:, None]
            fa = {c: float(np.mean([P[(*k, *c)] < 0.5 for k in sat if (*k, *c) in P])) for c in keys}
            ci = {c: keys.index(c) for c in keys}
            fr, cl, il = ci[("fresh", "none", "all")], ci[("closed", "none", "all")], ci[("inline", "none", "all")]
            tab = {}
            for c in keys:
                base = il if c[0] == "inline" else cl
                rec = (det[ci[c]] - det[base]) / (det[fr] - det[base]) if abs(det[fr] - det[base]) > 0.03 else float("nan")
                brec = (bdet[:, ci[c]] - bdet[:, base]) / (bdet[:, fr] - bdet[:, base])
                tab["|".join(c)] = {"det": float(det[ci[c]]), "det_ci": [float(x) for x in np.percentile(bdet[:, ci[c]], [2.5, 97.5])],
                                    "fa": fa[c], "recovered": float(rec),
                                    "rec_ci": [float(x) for x in np.nanpercentile(brec, [2.5, 97.5])] if rec == rec else None}
            res[f"{J}:{which}"] = {"n_viol": len(full), "n_sat": len(sat), "cond": tab}
            print(f"\n== {J} on {which} images: {len(full)} violated facts, {len(sat)} satisfied")
            for c in keys:
                t = tab["|".join(c)]
                rec = "" if t["recovered"] != t["recovered"] else f"  recovered {100*t['recovered']:5.0f}% [{100*t['rec_ci'][0]:.0f}, {100*t['rec_ci'][1]:.0f}]"
                print(f"{c[0]:7s} {c[1]:11s} {c[2]:5s} det {t['det']:.3f} [{t['det_ci'][0]:.2f},{t['det_ci'][1]:.2f}]  FA {t['fa']:.3f}{rec}")
    if out: json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
