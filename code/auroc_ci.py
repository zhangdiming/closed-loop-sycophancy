import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--labels", default="human"); ap.add_argument("--json"); a = ap.parse_args()
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, a.labels)
TYP = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
def cond(r):
    if r.get("perturb"): return None
    if not r.get("context"): return "fresh"
    m = r.get("context_mode"); return "ctx" if m in (None, "assistant") else f"ctx-{m}"
P = collections.defaultdict(dict)
for r in rows:
    if r["judge"] in ("qwen3vl8b", "owlv2") or r.get("p_no") is None: continue
    c = cond(r)
    if c: P[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["p_no"])
def pauc(pos, neg, fmax=0.05):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-x, kind="mergesort"); x, y = x[o], y[o]
    last = np.r_[np.diff(x) != 0, True]
    tp = np.cumsum(y)[last]; fp = np.cumsum(1 - y)[last]
    tpr = np.r_[0, tp / len(pos)]; fpr = np.r_[0, fp / len(neg)]
    keep = fpr <= fmax; fx = np.append(fpr[keep], fmax); ty = np.append(tpr[keep], np.interp(fmax, fpr, tpr))
    pa = np.trapezoid(ty, fx) if hasattr(np, "trapezoid") else np.trapz(ty, fx); mn = fmax ** 2 / 2
    return 0.5 * (1 + (pa - mn) / (fmax - mn))
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    x = np.concatenate([pos, neg]); rk = x.argsort().argsort().astype(float) + 1
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True); s = np.zeros(len(cnt)); np.add.at(s, inv, rk); rk = (s / cnt)[inv]
    return (rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
CONDS = ["fresh", "ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline", "ctx-assistant_mismatch", "ctx-assistant_cf"]
rng = np.random.default_rng(0); out = {}
judges = ["bagel", "omnigen2", "janus1b", "showo2_1.5b", "janus7b", "lumina", "blip3o", "showo2_7b"]
print(f"labels={a.labels}; AUROC of p(no), violated vs satisfied; others = common set (Janus-Pro-7B, Show-o2-7B images)")
for J in judges:
    gens = [g for g in ("janus7b", "showo2_7b") if g != J]
    for side, gl in (("oth", gens), ("own", [J])):
        avail = [c for c in CONDS if all((J, c, g) in P for g in gl)]
        if "fresh" not in avail or "ctx" not in avail: continue
        items = collections.defaultdict(list)
        for g in gl:
            keys = set.intersection(*[set(P[(J, c, g)]) for c in avail])
            for k in keys:
                lab = o.get((g,) + k)
                if lab not in ("yes", "no"): continue
                items[P[(J, "fresh", g)][k][0]].append((lab == "no", [P[(J, c, g)][k][1] for c in avail], TYP.get((g,) + k)))
        pids = sorted(items)
        def stats(sel):
            lab = np.array([v for p in sel for v, _, _ in items[p]]); X = np.array([x for p in sel for _, x, _ in items[p]]); ty = np.array([t for p in sel for _, _, t in items[p]])
            out = [auc(X[lab, i], X[~lab, i]) for i in range(len(avail))] + [pauc(X[lab, i], X[~lab, i]) for i in range(len(avail))]
            macro = []
            for i in range(len(avail)):
                v = [auc(X[lab & (ty == t), i], X[~lab & (ty == t), i]) for t in ("presence", "count", "color", "position") if (lab & (ty == t)).sum() >= 5]
                macro.append(np.nanmean(v) if v else np.nan)
            return out + macro
        est = stats(pids); B = []
        for _ in range(2000): B.append(stats([pids[i] for i in rng.integers(0, len(pids), len(pids))]))
        B = np.array(B); res = {}; L = len(avail)
        for i, c in enumerate(avail):
            res[c] = {"auc": float(est[i]), "ci95": [float(np.nanpercentile(B[:, i], 2.5)), float(np.nanpercentile(B[:, i], 97.5))],
                      "pauc": float(est[L + i]), "pauc_ci95": [float(np.nanpercentile(B[:, L + i], 2.5)), float(np.nanpercentile(B[:, L + i], 97.5))],
                      "macro": float(est[2 * L + i])}
            if c != "fresh":
                for nm, off in (("d", 0), ("dp", L), ("dm", 2 * L)):
                    d = B[:, off + i] - B[:, off]; res[c][f"{nm}_vs_fresh"] = float(est[off + i] - est[off]); res[c][f"{nm}_ci95"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        nv = sum(v for p in pids for v, _, _ in items[p]); res["_n"] = {"viol": int(nv), "sat": int(sum(len(items[p]) for p in pids) - nv)}
        out[f"{J}:{side}"] = res
        cl = res.get("ctx", {})
        print(f"{J:12s} {side}: n={nv:4d}/{res['_n']['sat']:4d}  AUROC {res['fresh']['auc']:.3f}->{cl.get('auc', float('nan')):.3f} ({cl.get('d_vs_fresh', float('nan')):+.3f})  pAUC05 {res['fresh']['pauc']:.3f}->{cl.get('pauc', float('nan')):.3f} ({cl.get('dp_vs_fresh', float('nan')):+.3f} [{cl.get('dp_ci95', [float('nan')]*2)[0]:+.3f},{cl.get('dp_ci95', [float('nan')]*2)[1]:+.3f}])  macro {res['fresh']['macro']:.3f}->{cl.get('macro', float('nan')):.3f} ({cl.get('dm_vs_fresh', float('nan')):+.3f})")
if a.json: json.dump(out, open(a.json, "w"), indent=1)
