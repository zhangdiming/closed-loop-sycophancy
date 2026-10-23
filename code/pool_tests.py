import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, GEN_OF
from theory_probe import CONDS, UNIFIED, spearman


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p) - np.log1p(-p)


def huber_fit(x, y, k=1.345, it=50):
    X = np.c_[np.ones(x.size), x]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b; s = np.median(np.abs(r)) / 0.6745 + 1e-9; w = np.minimum(1, k * s / np.maximum(np.abs(r), 1e-12))
        b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    P = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] in ("qwen3vl8b", "owlv2") or r.get("p_yes") is None: continue
        c = cond_of(r)
        if c is None: continue
        P[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["p_yes"])
    D = {}
    for J in sorted({k[0] for k in P}):
        own = GEN_OF.get(J, J)
        for c in CONDS:
            pid, pf, pc, viol = [], [], [], []
            for (j, cc, g), d in P.items():
                if j != J or cc != "fresh" or g == own or (J, c, g) not in P: continue
                dc = P[(J, c, g)]
                for k, (p, py) in d.items():
                    lab = o.get((g,) + k)
                    if k not in dc or lab is None: continue
                    pid.append(p); pf.append(py); pc.append(dc[k][1]); viol.append(lab == "no")
            if len(pid) < 50 or sum(viol) < 10: continue
            D[(J, c)] = dict(pid=np.array(pid), pf=np.array(pf), pc=np.array(pc), viol=np.array(viol))
    rates = lambda lp, v: (float((lp[v] < 0).mean()), float((lp[~v] < 0).mean()))
    res = {"shared": {}, "transfer": {}, "robust": {}}
    for J in sorted({k[0] for k in D}):
        cs = [c for c in CONDS if (J, c) in D]
        if "ctx" not in cs: continue
        X, Y, idx = [], [], []
        for i, c in enumerate(cs):
            q = D[(J, c)]; lf, lc = logit(q["pf"]), logit(q["pc"])
            Z = np.zeros((lf.size, len(cs))); Z[:, i] = 1; X.append(np.c_[Z, lf]); Y.append(lc - lf); idx.append(i)
        b = np.linalg.lstsq(np.vstack(X), np.concatenate(Y), rcond=None)[0]; lam_s = -b[-1]
        per = {}
        for i, c in enumerate(cs):
            q = D[(J, c)]; lf, lc = logit(q["pf"]), logit(q["pc"]); d = lc - lf; v = q["viol"]
            ic, sl = np.linalg.lstsq(np.c_[np.ones(lf.size), lf], d, rcond=None)[0]
            pred_free = (1 + sl) * lf + ic; pred_sh = (1 - lam_s) * lf + b[i]
            ss = ((d - d.mean()) ** 2).sum()
            obs = rates(lc, v); fr = rates(pred_free, v); sh = rates(pred_sh, v)
            per[c] = {"lambda_free": float(-sl), "r2_free": float(1 - ((lc - pred_free) ** 2).sum() / ss), "r2_shared": float(1 - ((lc - pred_sh) ** 2).sum() / ss),
                      "det_obs": obs[0], "fa_obs": obs[1], "det_free": fr[0], "fa_free": fr[1], "det_shared": sh[0], "fa_shared": sh[1]}
        res["shared"][J] = {"lambda_shared": float(lam_s), "contexts": per}
        q = D[(J, "ctx")]; lf0 = logit(q["pf"]); s0 = np.linalg.lstsq(np.c_[np.ones(lf0.size), lf0], logit(q["pc"]) - lf0, rcond=None)[0][1]
        tr = {}
        for c in cs:
            if c == "ctx": continue
            q = D[(J, c)]; lf, lc = logit(q["pf"]), logit(q["pc"]); v = q["viol"]
            ic = float(np.mean((lc - lf) - s0 * lf)); pred = (1 + s0) * lf + ic
            obs, pr, nc = rates(lc, v), rates(pred, v), rates(lf, v)
            tr[c] = {"intercept": ic, "det_obs": obs[0], "det_pred": pr[0], "fa_obs": obs[1], "fa_pred": pr[1],
                     "det_err": abs(obs[0] - pr[0]), "fa_err": abs(obs[1] - pr[1]),
                     "det_nochange": nc[0], "fa_nochange": nc[1], "det_err_nochange": abs(obs[0] - nc[0]), "fa_err_nochange": abs(obs[1] - nc[1])}
        res["transfer"][J] = {"lambda_closed_loop": float(-s0), "contexts": tr}
        q = D[(J, "ctx")]; out = {}
        for name, eps, fit in (("ols_1e-6", 1e-6, "ols"), ("ols_1e-3", 1e-3, "ols"), ("huber_1e-6", 1e-6, "huber")):
            lf, lc = logit(q["pf"], eps), logit(q["pc"], eps)
            ic, sl = (np.linalg.lstsq(np.c_[np.ones(lf.size), lf], lc - lf, rcond=None)[0] if fit == "ols" else huber_fit(lf, lc - lf))
            out[name] = {"lambda": float(-sl), "icpt": float(ic), "m": float(ic / (1 + sl))}
        res["robust"][J] = out
        print(J, f"lambda shared {lam_s:.2f}", {c: round(per[c]["lambda_free"], 2) for c in cs},
              "R2 free/shared", {c: (round(per[c]["r2_free"], 3), round(per[c]["r2_shared"], 3)) for c in cs}, flush=True)
        print("   transfer", {c: (round(t["det_obs"], 2), round(t["det_pred"], 2), round(t["fa_obs"], 3), round(t["fa_pred"], 3)) for c, t in tr.items()},
              " robust m", {k: round(v["m"], 2) for k, v in out.items()}, flush=True)
    Q = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] != "janus7b" or r["gen"] != "janus7b" or r.get("p_yes") is None or r.get("context"): continue
        Q[r.get("perturb") or "none"][(r["image"], r["fid"])] = r["p_yes"]
    res["null"] = {}
    for k in sorted(Q):
        if k == "none": continue
        ks = [x for x in Q[k] if x in Q["none"]]; lf = logit([Q["none"][x] for x in ks]); lp = logit([Q[k][x] for x in ks])
        ic, sl = np.linalg.lstsq(np.c_[np.ones(lf.size), lf], lp - lf, rcond=None)[0]
        res["null"][k] = {"n": len(ks), "lambda": float(-sl), "icpt": float(ic)}
    print("null slopes", res["null"], flush=True)
    js = sorted(J for J in UNIFIED if (J, "ctx") in D)

    def stat(sel_pids):
        m, loss = [], []
        for J in js:
            q = D[(J, "ctx")]; w = np.array([sel_pids.get(p, 0) for p in q["pid"]], float)
            lf, lc, v = logit(q["pf"]), logit(q["pc"]), q["viol"]
            X = np.c_[np.ones(lf.size), lf] * np.sqrt(w)[:, None]; ic, sl = np.linalg.lstsq(X, (lc - lf) * np.sqrt(w), rcond=None)[0]
            m.append(ic / (1 + sl)); loss.append(((lf[v] < 0) * w[v]).sum() / w[v].sum() - ((lc[v] < 0) * w[v]).sum() / w[v].sum())
        return spearman(m, loss)
    allp = sorted({p for J in js for p in D[(J, "ctx")]["pid"]}); rng = np.random.default_rng(0)
    est = stat({p: 1 for p in allp}); bs = []
    for _ in range(1000):
        s = collections.Counter(rng.choice(allp, len(allp))); bs.append(stat(s))
    res["rho"] = {"judges": js, "est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print("rho", res["rho"])
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
