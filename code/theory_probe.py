import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, GEN_OF

B = int(__import__('os').environ.get('B', 2000))
CONDS = ["ctx", "ctx-assistant_nocaption", "ctx-user_prompt", "ctx-caption_inline", "ctx-assistant_mismatch", "ctx-assistant_cf"]
UNIFIED = {"bagel", "janus1b", "janus7b", "showo2_1.5b", "showo2_7b", "omnigen2", "blip3o", "lumina"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p) - np.log1p(-p)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--labels", default="human"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, a.labels)
    P = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] in ("qwen3vl8b", "owlv2") or r.get("p_yes") is None: continue
        c = cond_of(r)
        if c is None: continue
        P[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["p_yes"])
    rng = np.random.default_rng(0); res = {}
    for J in sorted({k[0] for k in P}):
        own = GEN_OF.get(J, J)
        for c in CONDS:
            pid, lf, lc, viol = [], [], [], []
            for (j, cc, g), d in P.items():
                if j != J or cc != "fresh" or g == own or (J, c, g) not in P: continue
                dc = P[(J, c, g)]
                for k, (p, py) in d.items():
                    lab = o.get((g,) + k)
                    if k not in dc or lab is None: continue
                    pid.append(p); lf.append(py); lc.append(dc[k][1]); viol.append(lab == "no")
            if len(pid) < 50 or sum(viol) < 10: continue
            lf, lc, viol = logit(lf), logit(lc), np.array(viol); d = lc - lf
            pids = sorted(set(pid)); ix = {p: i for i, p in enumerate(pids)}; g = np.array([ix[p] for p in pid])
            members = [np.where(g == i)[0] for i in range(len(pids))]

            def stats(sel):
                v, s = viol[sel], ~viol[sel]; ds = d[sel]
                delta = float(np.median(ds)); delta_sat = float(np.median(ds[s]))
                det_f, det_c = float((lf[sel][v] < 0).mean()), float((lc[sel][v] < 0).mean())
                det_p = float((lf[sel][v] + delta < 0).mean())
                fa_f, fa_c, fa_p = float((lf[sel][s] < 0).mean()), float((lc[sel][s] < 0).mean()), float((lf[sel][s] + delta < 0).mean())
                loss = det_f - det_c; share = (det_f - det_p) / loss if abs(loss) > 0.02 else np.nan
                X = np.c_[np.ones(sel.size), lf[sel]]; icpt, slope = (float(x) for x in np.linalg.lstsq(X, ds, rcond=None)[0])
                lp = (1 + slope) * lf[sel] + icpt
                det_pool, fa_pool = float((lp[v] < 0).mean()), float((lp[s] < 0).mean())
                share_pool = (det_f - det_pool) / loss if abs(loss) > 0.02 else np.nan
                def auc(x):
                    pos, neg = -x[v], -x[s]; r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
                    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
                return dict(delta=delta, delta_sat=delta_sat, det_fresh=det_f, det_c=det_c, det_pred=det_p, fa_fresh=fa_f, fa_c=fa_c,
                            fa_pred=fa_p, share=share, d_viol=float(ds[v].mean()), d_sat=float(ds[s].mean()),
                            d_gap=float(ds[v].mean() - ds[s].mean()), slope=slope, icpt=icpt, det_pool=det_pool, fa_pool=fa_pool,
                            share_pool=share_pool, auc_fresh=auc(lf[sel]), auc_c=auc(lc[sel]))
            est = stats(np.arange(len(pid)))
            bs = collections.defaultdict(list)
            for _ in range(B):
                sel = np.concatenate([members[i] for i in rng.integers(0, len(pids), len(pids))])
                for k, v in stats(sel).items(): bs[k].append(v)
            ho = []
            for _ in range(20):
                perm = rng.permutation(len(pids)); fit = np.concatenate([members[i] for i in perm[: len(pids) // 2]])
                tst = np.concatenate([members[i] for i in perm[len(pids) // 2:]])
                c1, s1 = np.linalg.lstsq(np.c_[np.ones(fit.size), lf[fit]], d[fit], rcond=None)[0]
                vt = viol[tst]; lp = (1 + s1) * lf[tst] + c1
                ho.append(((lp[vt] < 0).mean(), (lc[tst][vt] < 0).mean()))
            ho = np.array(ho); est["det_pool_heldout"] = float(ho[:, 0].mean()); est["det_obs_heldout"] = float(ho[:, 1].mean())
            est["det_err_heldout"] = float(np.abs(ho[:, 0] - ho[:, 1]).mean())
            out = {k: {"est": v, "ci95": [float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))] if k in bs else None} for k, v in est.items()}
            out["in_model"] = bool(-est["slope"] >= 0)
            out["_n"] = {"items": len(pid), "viol": int(viol.sum()), "prompts": len(pids)}
            res[f"{J}:{c}"] = out
            e = est
            print(f"{J:15s} {c:24s} n={len(pid):5d}/{int(viol.sum()):4d}  delta {e['delta']:+.2f} (sat {e['delta_sat']:+.2f})  "
                  f"det {e['det_fresh']:.3f}->{e['det_c']:.3f} pred {e['det_pred']:.3f} share {e['share']:.2f} "
                  f"[{out['share']['ci95'][0]:.2f},{out['share']['ci95'][1]:.2f}]  FA {e['fa_fresh']:.3f}->{e['fa_c']:.3f} pred {e['fa_pred']:.3f}  "
                  f"d viol/sat {e['d_viol']:+.2f}/{e['d_sat']:+.2f}  slope {e['slope']:+.2f} icpt {e['icpt']:+.2f}  POOL det {e['det_pool']:.3f} FA {e['fa_pool']:.3f} "
                  f"heldout det {e['det_pool_heldout']:.3f} vs {e['det_obs_heldout']:.3f} |err| {e['det_err_heldout']:.3f}  share {e['share_pool']:.2f} [{out['share_pool']['ci95'][0]:.2f},{out['share_pool']['ci95'][1]:.2f}]  AUROC {e['auc_fresh']:.3f}->{e['auc_c']:.3f}", flush=True)
    dh = json.load(open(ROOT / "runs/ci/did_human.json"))
    js = [J for J in sorted({k.split(":")[0] for k in res}) if f"{J}:ctx" in res and J in dh and "framing" in dh[J]]
    js = [J for J in js if J != "bagel_vae"]
    x = [res[f"{J}:ctx"]["delta"]["est"] for J in js]; y = [dh[J]["framing"]["est"] for J in js]
    m = [res[f"{J}:ctx"]["icpt"]["est"] / (1 + res[f"{J}:ctx"]["slope"]["est"]) for J in js]
    res["_audit"] = {"judges": js, "delta": x, "margin": m, "closed_loop_term": y,
                     "spearman_delta": spearman(x, y), "spearman_margin": spearman(m, [-v for v in y])}
    print("\nlabel-free audit:", [(J, round(a_, 2), round(b_, 2), round(c_, 3)) for J, a_, b_, c_ in zip(js, x, m, y)],
          "Spearman(delta, term)", round(res["_audit"]["spearman_delta"], 3), "Spearman(margin, loss)", round(res["_audit"]["spearman_margin"], 3))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
