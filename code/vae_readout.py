import argparse, collections, json
import numpy as np
from bootstrap_ci import labels, load_rows, ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    A = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] not in ("bagel", "bagel_vae") or r["gen"] not in ("bagel", "janus7b", "showo2_7b"): continue
        cm = r.get("context_mode")
        c = "fresh" if not r.get("context") else ("ctx" if cm in (None, "assistant") else None)
        if c is None or r.get("perturb"): continue
        A[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["answer"] == "no",
                                                                (r["p_yes"] is not None and r["p_yes"] < 0.5), r["answer"] not in ("yes", "no"))
    res = {}; rng = np.random.default_rng(0)
    for J in ("bagel", "bagel_vae"):
        for rule, ix in (("decoded", 1), ("pyes", 2)):
            out = {}; per = {}
            for side, gens in (("own", ["bagel"]), ("oth", ["janus7b", "showo2_7b"])):
                for want, key in (("no", "det"), ("yes", "fa")):
                    if side == "own" and key == "fa": continue
                    pid, f, c, nf, nc = [], [], [], [], []
                    for g in gens:
                        df, dc = A[(J, "fresh", g)], A[(J, "ctx", g)]
                        for k, v in df.items():
                            if k not in dc or o.get((g, *k)) != want: continue
                            pid.append(v[0]); f.append(v[ix]); c.append(dc[k][ix]); nf.append(v[3]); nc.append(dc[k][3])
                    f, c = np.array(f, float), np.array(c, float)
                    out[f"{key}_{side}"] = [float(f.mean()), float(c.mean())]; out[f"n_{key}_{side}"] = len(f)
                    out[f"nonyn_{key}_{side}"] = [float(np.mean(nf)), float(np.mean(nc))]
                    per[(side, key)] = (np.array(pid), f, c)
            (po, fo, co), (pt, ft, ct) = per[("own", "det")], per[("oth", "det")]
            allp = sorted(set(po) | set(pt)); io = collections.defaultdict(list); it = collections.defaultdict(list)
            for i, p in enumerate(po): io[p].append(i)
            for i, p in enumerate(pt): it[p].append(i)
            def terms(so, st): return fo[so].mean() - ft[st].mean(), (co[so].mean() - fo[so].mean()) - (ct[st].mean() - ft[st].mean())
            e = terms(np.arange(fo.size), np.arange(ft.size)); B = []
            for _ in range(2000):
                smp = rng.choice(allp, len(allp))
                so = np.array([i for p in smp for i in io.get(p, [])], int); st = np.array([i for p in smp for i in it.get(p, [])], int)
                if so.size and st.size: B.append(terms(so, st))
            B = np.array(B)
            out["self_fresh"] = {"est": float(e[0]), "ci95": [float(np.percentile(B[:, 0], 2.5)), float(np.percentile(B[:, 0], 97.5))]}
            out["self_specific"] = {"est": float(e[1]), "ci95": [float(np.percentile(B[:, 1], 2.5)), float(np.percentile(B[:, 1], 97.5))]}
            res[f"{J}:{rule}"] = out
            print(J, rule, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in out.items()}))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
