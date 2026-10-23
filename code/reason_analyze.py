import collections, glob, json, sys, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
OWN = {"bagel": "bagel", "janus7b": "janus7b", "omnigen2": "omnigen2"}


def main():
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    R = {}
    for f in glob.glob(str(ROOT / "runs/reason/*.jsonl")):
        for l in open(f):
            r = json.loads(l); R[(r["judge"], r["gen"], r["image"], r["fid"], r["ctx"])] = r
    M = {}
    for f in glob.glob(str(ROOT / "runs/mask/mask_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["mask"] == "none" and r["ctx"] in ("fresh", "closed"): M[(r["judge"], r["gen"], r["image"], r["fid"], r["ctx"])] = r["p_yes"]
    rng = np.random.default_rng(0); res = {}
    for J in sorted({k[0] for k in R}):
        for which in ("own", "others"):
            keys = sorted({k[1:4] for k in R if k[0] == J and ((k[1] == OWN[J]) == (which == "own"))
                           and all((J, *k[1:4], c) in R for c in ("fresh", "closed"))})
            if not keys: continue
            lab = {k: R[(J, *k, "fresh")]["label"] for k in keys}; pid = {k: R[(J, *k, "fresh")]["pid"] for k in keys}
            rej = lambda k, c: R[(J, *k, c)]["verdict"] == "no"
            unp = np.mean([R[(J, *k, c)]["verdict"] is None for k in keys for c in ("fresh", "closed")])
            have_ft = [k for k in keys if all((J, *k, c) in M for c in ("fresh", "closed"))]
            row = {"n_viol": sum(lab[k] == "violated" for k in keys), "n_sat": sum(lab[k] == "satisfied" for k in keys), "unparsed": float(unp)}
            for name, fn, ks in (("reason", rej, keys), ("first_token", lambda k, c: M[(J, *k, c)] < 0.5, have_ft)):
                V = [k for k in ks if lab[k] == "violated"]; S = [k for k in ks if lab[k] == "satisfied"]
                if not V: continue
                pids = sorted({pid[k] for k in V}); ix = {p: i for i, p in enumerate(pids)}
                X = np.array([[fn(k, "fresh"), fn(k, "closed")] for k in V], float); g = np.array([ix[pid[k]] for k in V])
                H = np.zeros((len(pids), 2)); N = np.zeros(len(pids)); np.add.at(H, g, X); np.add.at(N, g, 1)
                W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=2000).astype(float)
                B = (W @ H) / (W @ N)[:, None]; d = B[:, 1] - B[:, 0]
                row[name] = {"n_viol": len(V), "det_fresh": float(X[:, 0].mean()), "det_closed": float(X[:, 1].mean()),
                             "change": float(X[:, 1].mean() - X[:, 0].mean()), "change_ci": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                             "fa_fresh": float(np.mean([fn(k, "fresh") for k in S])) if S else None,
                             "fa_closed": float(np.mean([fn(k, "closed") for k in S])) if S else None}
            res[f"{J}:{which}"] = row
            print(f"\n== {J} on {which}: {row['n_viol']} violated, {row['n_sat']} satisfied, unparsed {unp:.3f}")
            for name in ("reason", "first_token"):
                if name not in row: continue
                q = row[name]
                print(f"  {name:11s} det {q['det_fresh']:.3f} -> {q['det_closed']:.3f}  change {q['change']:+.3f} [{q['change_ci'][0]:+.2f},{q['change_ci'][1]:+.2f}]"
                      f"  FA {q['fa_fresh']} -> {q['fa_closed']}  (n={q['n_viol']})")
    if out: json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
