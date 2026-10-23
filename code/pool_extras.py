import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, GEN_OF

UNIFIED = ["bagel", "blip3o", "janus1b", "janus7b", "lumina", "omnigen2", "showo2_1.5b", "showo2_7b"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    P = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] not in UNIFIED or r.get("p_yes") is None: continue
        c = cond_of(r)
        if c not in ("fresh", "ctx"): continue
        P[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["p_yes"])
    res = {"correction": {}, "selfspec": {}}; rng = np.random.default_rng(0)
    for J in UNIFIED:
        own = GEN_OF.get(J, J)
        def collect(side, need_label):
            pid, lf, lc, lab = [], [], [], []
            for (j, cc, g), d in P.items():
                if j != J or cc != "fresh" or (J, "ctx", g) not in P: continue
                if (side == "own") != (g == own): continue
                dc = P[(J, "ctx", g)]
                for k, (p, py) in d.items():
                    if k not in dc: continue
                    l = o.get((g,) + k)
                    if need_label and l is None: continue
                    pid.append(p); lf.append(py); lc.append(dc[k][1]); lab.append(l)
            return np.array(pid), logit(lf), logit(lc), np.array(lab, dtype=object)
        _, af, ac, _ = collect("oth", False)
        icpt, slope = np.linalg.lstsq(np.c_[np.ones(af.size), af], ac - af, rcond=None)[0]
        pid, lf, lc, lab = collect("oth", True); v = lab == "no"; s_ = lab == "yes"
        corr = (lc - icpt) / (1 + slope)
        det = lambda x, m: float((x[m] < 0).mean())
        df, dc, dk = det(lf, v), det(lc, v), det(corr, v)
        res["correction"][J] = {"slope": float(slope), "icpt": float(icpt), "det_fresh": df, "det_closed": dc, "det_corrected": dk,
                                "fa_fresh": det(lf, s_), "fa_closed": det(lc, s_), "fa_corrected": det(corr, s_),
                                "recovered": (dk - dc) / (df - dc) if df - dc > 0.02 else None}
        opid, of, oc, olab = collect("own", True); ov = olab == "no"
        if ov.sum() < 10: continue
        pred = lambda x: (1 + slope) * x + icpt
        def ss(sel_o, sel_t):
            vo, vt = ov[sel_o], v[sel_t]
            obs = (det(oc[sel_o], vo) - det(of[sel_o], vo)) - (det(lc[sel_t], vt) - det(lf[sel_t], vt))
            prd = (det(pred(of[sel_o]), vo) - det(of[sel_o], vo)) - (det(pred(lf[sel_t]), vt) - det(lf[sel_t], vt))
            return obs, prd
        obs, prd = ss(np.arange(of.size), np.arange(lf.size))
        allp = sorted(set(opid) | set(pid)); io = collections.defaultdict(list); it = collections.defaultdict(list)
        for i, p in enumerate(opid): io[p].append(i)
        for i, p in enumerate(pid): it[p].append(i)
        bo, bp, bd = [], [], []
        for _ in range(1000):
            smp = rng.choice(allp, len(allp))
            so = np.array([i for p in smp for i in io.get(p, [])], int); st = np.array([i for p in smp for i in it.get(p, [])], int)
            if ov[so].sum() == 0 or v[st].sum() == 0: continue
            x, y = ss(so, st); bo.append(x); bp.append(y); bd.append(x - y)
        res["selfspec"][J] = {"n_own_viol": int(ov.sum()), "observed": obs, "predicted": prd,
                              "observed_ci95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                              "predicted_ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
                              "residual": obs - prd, "residual_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                              "own_fresh_mean_logit_viol": float(of[ov].mean()), "oth_fresh_mean_logit_viol": float(lf[v].mean())}
        c, q = res["correction"][J], res["selfspec"][J]
        print(f"{J:12s} det fresh {c['det_fresh']:.3f} closed {c['det_closed']:.3f} corrected {c['det_corrected']:.3f} recovered {c['recovered']}  "
              f"FA {c['fa_fresh']:.3f}/{c['fa_closed']:.3f}/{c['fa_corrected']:.3f} | selfspec obs {obs:+.3f} pred {prd:+.3f}", flush=True)
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
