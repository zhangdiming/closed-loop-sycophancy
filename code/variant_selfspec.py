import argparse, collections, json
import numpy as np
from bootstrap_ci import load_rows, labels, cond_of, ROOT

JUDGES = ["bagel", "janus1b", "janus7b", "omnigen2", "showo2_1.5b", "showo2_7b"]
CONDS = ["ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline", "ctx-request_inline", "ctx-assistant_instr"]
OTHERS = ("janus7b", "showo2_7b")
TYPES = ("presence", "count", "color", "position")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    A = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] not in JUDGES or r.get("p_yes") is None: continue
        c = cond_of(r)
        if c is None or (c != "fresh" and c not in CONDS): continue
        if r["gen"] != r["judge"] and r["gen"] not in OTHERS: continue
        A[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["type"], r["answer"] == "no", r["p_yes"])
    res = {}; rng = np.random.default_rng(0)
    for J in JUDGES:
        for C in CONDS:
            gens = [g for g in dict.fromkeys([J, *OTHERS]) if (J, C, g) in A and (J, "fresh", g) in A]
            if J not in gens or len(gens) < 2: continue
            pid, side, typ, lab, fr, cl, lf, lc = [], [], [], [], [], [], [], []
            for g in gens:
                df, dc = A[(J, "fresh", g)], A[(J, C, g)]
                for k in df.keys() & dc.keys():
                    l = o.get((g,) + k)
                    pid.append(df[k][0]); side.append(g == J); typ.append(df[k][1]); lab.append(l)
                    fr.append(df[k][2]); cl.append(dc[k][2]); lf.append(df[k][3]); lc.append(dc[k][3])
            pid, side, typ, lab = np.array(pid), np.array(side), np.array(typ), np.array(lab, dtype=object)
            fr, cl, lf, lc = np.array(fr, float), np.array(cl, float), logit(lf), logit(lc)
            viol = lab == "no"
            oth = ~side
            icpt, slope = np.linalg.lstsq(np.c_[np.ones(oth.sum()), lf[oth]], (lc - lf)[oth], rcond=None)[0]
            pred = ((1 + slope) * lf + icpt) < 0
            frp = lf < 0
            ov, tv = viol & side, viol & oth

            def terms(io, it):
                raw = (cl[io].mean() - fr[io].mean()) - (cl[it].mean() - fr[it].mean())
                prd = (pred[io].mean() - frp[io].mean()) - (pred[it].mean() - frp[it].mean())
                tt = [t for t in TYPES if (typ[io] == t).any() and (typ[it] == t).any()]
                ws = sum((typ[io] == t).sum() for t in tt)
                std = sum((typ[io] == t).sum() / ws * ((cl[io][typ[io] == t].mean() - fr[io][typ[io] == t].mean())
                                                       - (cl[it][typ[it] == t].mean() - fr[it][typ[it] == t].mean())) for t in tt)
                return raw, std, raw - prd, cl[io].mean() - fr[io].mean(), cl[it].mean() - fr[it].mean()
            io_all, it_all = np.where(ov)[0], np.where(tv)[0]
            est = terms(io_all, it_all)
            P = sorted(set(pid[ov]) | set(pid[tv])); po = collections.defaultdict(list); pt = collections.defaultdict(list)
            for i in io_all: po[pid[i]].append(i)
            for i in it_all: pt[pid[i]].append(i)
            B = []
            for _ in range(a.B):
                smp = rng.choice(P, len(P))
                io = np.array([i for p in smp for i in po.get(p, [])], int); it = np.array([i for p in smp for i in pt.get(p, [])], int)
                if io.size and it.size: B.append(terms(io, it))
            B = np.array(B); names = ["raw", "std", "residual", "own_change", "oth_change"]
            out = {"n_own": int(ov.sum()), "n_oth": int(tv.sum()), "slope": float(slope), "icpt": float(icpt)}
            for i, nm in enumerate(names):
                out[nm] = {"est": float(est[i]), "ci95": [float(np.percentile(B[:, i], 2.5)), float(np.percentile(B[:, i], 97.5))],
                           "ci90": [float(np.percentile(B[:, i], 5)), float(np.percentile(B[:, i], 95))]}
            res[f"{J}:{C}"] = out
            f = lambda d: f"{d['est']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] (90% lo {d['ci90'][0]:+.3f})"
            print(f"{J:12s} {C:24s} n {out['n_own']:4d}/{out['n_oth']:4d}  own {out['own_change']['est']:+.3f} oth {out['oth_change']['est']:+.3f}  "
                  f"raw {f(out['raw'])}  std {f(out['std'])}  resid {f(out['residual'])}", flush=True)
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
