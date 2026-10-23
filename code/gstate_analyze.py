import argparse, collections, json
import numpy as np


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--json"); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file)]
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    forms = sorted({c.rsplit("_", 1)[0] for c in conds})
    base = "vit" if "vit" in forms else "main"
    R = collections.defaultdict(dict); lab, pid = {}, {}; mass = collections.defaultdict(list)
    for r in rows:
        k = (r["image"], r["fid"])
        if r["p_yes"] is None: continue
        R[k][r["cond"]] = (r["p_yes"] < 0.5, r["answer"] == "no", 1 - r["p_yes"]); lab[k] = r["label"]; pid[k] = r["pid"]
        mass[r["cond"]].append(r["mass"])
    keys = [k for k in R if all(c in R[k] for c in conds) and lab[k] in ("yes", "no")]
    pids = sorted({pid[k] for k in keys}); P = {p: i for i, p in enumerate(pids)}
    rng = np.random.default_rng(0); boot = rng.integers(0, len(pids), size=(a.B, len(pids)))
    cnt = np.stack([np.bincount(b, minlength=len(pids)) for b in boot]).astype(float)
    res = {"n_images": len({k[0] for k in keys}), "n_viol": sum(lab[k] == "no" for k in keys), "n_sat": sum(lab[k] == "yes" for k in keys),
           "mass": {c: float(np.mean(mass[c])) for c in conds}, "auroc": {}, "rules": {}}
    viol = [k for k in keys if lab[k] == "no"]; sat = [k for k in keys if lab[k] == "yes"]
    for c in conds: res["auroc"][c] = auroc([R[k][c][2] for k in viol], [R[k][c][2] for k in sat])
    diffs = [(f"request, {f}", f"{f}_cl", f"{f}_fresh") for f in forms]
    for f in forms:
        if f == base: continue
        for ctx in ("fresh", "cl"):
            if not f.startswith("state"): continue
            if f"{f.replace('state', 'reenc')}_{ctx}" in conds:
                diffs.append((f"{f} vs {f.replace('state', 'reenc')}, {ctx}", f"{f}_{ctx}", f"{f.replace('state', 'reenc')}_{ctx}"))
            diffs.append((f"{f} vs {base}, {ctx}", f"{f}_{ctx}", f"{base}_{ctx}"))
    for rule, ix in (("pyes", 0), ("decoded", 1)):
        out = {"det": {}, "fa": {}, "diff": {}}
        for which, ks in (("det", viol), ("fa", sat)):
            idx = np.array([P[pid[k]] for k in ks], dtype=int); w = cnt[:, idx]
            X = {c: np.array([R[k][c][ix] for k in ks], float) for c in conds}
            def bs(v): e = (w * v).sum(1) / w.sum(1); return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
            def bs90(v): e = (w * v).sum(1) / w.sum(1); return [float(np.percentile(e, 5)), float(np.percentile(e, 95))]
            for c in conds: out[which][c] = {"est": float(X[c].mean()), "ci95": bs(X[c])}
            for name, c1, c2 in diffs:
                d = X[c1] - X[c2]; out["diff"].setdefault(name, {})[which] = {"est": float(d.mean()), "ci95": bs(d), "ci90": bs90(d)}
        res["rules"][rule] = out
    print(f"{a.file}: images {res['n_images']}  violated {res['n_viol']}  satisfied {res['n_sat']}")
    for c in conds:
        o = res["rules"]["pyes"]
        print(f"  {c:16s} mass {res['mass'][c]:.3f}  AUROC {res['auroc'][c]:.3f}  det {o['det'][c]['est']:.3f} [{o['det'][c]['ci95'][0]:.2f},{o['det'][c]['ci95'][1]:.2f}]"
              f"  FA {o['fa'][c]['est']:.3f}  | decoded det {res['rules']['decoded']['det'][c]['est']:.3f} FA {res['rules']['decoded']['fa'][c]['est']:.3f}")
    for name, _, _ in diffs:
        d = res["rules"]["pyes"]["diff"][name]
        print(f"  {name:28s} det {d['det']['est']:+.3f} [{d['det']['ci95'][0]:+.3f},{d['det']['ci95'][1]:+.3f}]  FA {d['fa']['est']:+.3f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
