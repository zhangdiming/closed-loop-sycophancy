import argparse, collections, json
import numpy as np
from bootstrap_ci import labels, load_rows, ROOT

CONDS = ["vit_fresh", "vit_cl", "vae_fresh", "vae_cl"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--json"); a = ap.parse_args()
    o, _ = labels(load_rows(ROOT / "runs/judge"), "human")
    R = collections.defaultdict(dict); pid = {}; mass = collections.defaultdict(list); nonyn = collections.defaultdict(list)
    for l in open(a.file):
        r = json.loads(l); k = (r["gen"], r["image"], r["fid"])
        if r["p_yes"] is None: continue
        R[k][r["cond"]] = (r["answer"] == "no", r["p_yes"] < 0.5); pid[k] = r["pid"]
        mass[r["cond"]].append(r["mass"]); nonyn[r["cond"]].append(r["answer"] not in ("yes", "no"))
    keys = [k for k in R if all(c in R[k] for c in CONDS) and o.get(k) in ("yes", "no")]
    res = {"mass": {c: float(np.mean(mass[c])) for c in CONDS}, "nonyn": {c: float(np.mean(nonyn[c])) for c in CONDS}, "rules": {}}
    rng = np.random.default_rng(0)
    sets = {"own_det": [k for k in keys if k[0] == "bagel" and o[k] == "no"], "oth_det": [k for k in keys if k[0] != "bagel" and o[k] == "no"],
            "oth_fa": [k for k in keys if k[0] != "bagel" and o[k] == "yes"]}
    for rule, ix in (("decoded", 0), ("pyes", 1)):
        out = {}
        for s, ks in sets.items():
            out[s] = {c: float(np.mean([R[k][c][ix] for k in ks])) for c in CONDS}; out[f"n_{s}"] = len(ks)
        X = {s: {c: np.array([R[k][c][ix] for k in sets[s]], float) for c in CONDS} for s in ("own_det", "oth_det")}
        Pid = {s: np.array([pid[k] for k in sets[s]]) for s in ("own_det", "oth_det")}
        allp = sorted(set(Pid["own_det"]) | set(Pid["oth_det"]))
        ix_ = {s: collections.defaultdict(list) for s in X}
        for s in X:
            for i, p in enumerate(Pid[s]): ix_[s][p].append(i)
        def stats(sel):
            st = {}
            for s in X:
                v = {c: X[s][c][sel[s]] for c in CONDS}
                st[f"cost_vit_{s}"] = v["vit_fresh"].mean() - v["vit_cl"].mean(); st[f"cost_vae_{s}"] = v["vae_fresh"].mean() - v["vae_cl"].mean()
                st[f"ratio_{s}"] = st[f"cost_vae_{s}"] / st[f"cost_vit_{s}"] if st[f"cost_vit_{s}"] else np.nan
            for f in ("vit", "vae"):
                o_, t_ = X["own_det"], X["oth_det"]
                st[f"self_fresh_{f}"] = o_[f"{f}_fresh"][sel["own_det"]].mean() - t_[f"{f}_fresh"][sel["oth_det"]].mean()
                st[f"self_specific_{f}"] = (o_[f"{f}_cl"][sel["own_det"]].mean() - o_[f"{f}_fresh"][sel["own_det"]].mean()) - \
                                           (t_[f"{f}_cl"][sel["oth_det"]].mean() - t_[f"{f}_fresh"][sel["oth_det"]].mean())
            return st
        est = stats({s: np.arange(len(Pid[s])) for s in X}); B = collections.defaultdict(list)
        for _ in range(2000):
            smp = rng.choice(allp, len(allp))
            sel = {s: np.array([i for p in smp for i in ix_[s].get(p, [])], int) for s in X}
            if all(len(v) for v in sel.values()):
                for kk, vv in stats(sel).items(): B[kk].append(vv)
        out["terms"] = {kk: {"est": float(vv), "ci95": [float(np.nanpercentile(B[kk], 2.5)), float(np.nanpercentile(B[kk], 97.5))]} for kk, vv in est.items()}
        res["rules"][rule] = out
        print(rule, "n", {s: out[f"n_{s}"] for s in sets})
        for s in sets: print(f"  {s:8s}", {c: round(out[s][c], 3) for c in CONDS})
        for kk, vv in out["terms"].items(): print(f"  {kk:22s} {vv['est']:+.3f} [{vv['ci95'][0]:+.3f}, {vv['ci95'][1]:+.3f}]")
    print("mass", {c: round(v, 3) for c, v in res["mass"].items()}, "non-yes/no", {c: round(v, 3) for c, v in res["nonyn"].items()})
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
