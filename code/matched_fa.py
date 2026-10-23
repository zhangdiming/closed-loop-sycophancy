import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, ROOT
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
P = collections.defaultdict(dict)
for r in rows:
    if r["judge"] in ("qwen3vl8b", "owlv2") or r.get("p_no") is None or r.get("perturb"): continue
    if r.get("context") and r.get("context_mode") not in (None, "assistant"): continue
    P[(r["judge"], "ctx" if r.get("context") else "fresh", r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["p_no"], r["answer"])
out = {}
print(f"{'judge':12s} {'det fresh':>9s} {'det closed':>10s} {'det closed @FA-matched':>23s} {'FA fresh':>8s} {'FA closed@matched':>17s}")
for J in ["bagel", "omnigen2", "janus1b", "showo2_1.5b", "janus7b", "lumina", "blip3o", "showo2_7b"]:
    data = []
    for g in [g for g in ("janus7b", "showo2_7b") if g != J]:
        if (J, "fresh", g) not in P or (J, "ctx", g) not in P: continue
        for k in set(P[(J, "fresh", g)]) & set(P[(J, "ctx", g)]):
            lab = o.get((g,) + k)
            if lab in ("yes", "no"):
                f, c = P[(J, "fresh", g)][k], P[(J, "ctx", g)][k]
                data.append((f[0], lab == "no", f[2] == "no", c[2] == "no", c[1]))
    pids = sorted({d[0] for d in data}); rng = np.random.default_rng(0); res = []
    for _ in range(20):
        perm = rng.permutation(pids); fit = set(perm[: len(perm) // 2])
        A = [d for d in data if d[0] in fit]; T = [d for d in data if d[0] not in fit]
        fa_target = np.mean([d[2] for d in A if not d[1]])
        sat_c = np.sort([d[4] for d in A if not d[1]])[::-1]
        thr = sat_c[int(np.floor(fa_target * len(sat_c)))] if fa_target * len(sat_c) >= 1 else sat_c[0] + 1e-12
        det = np.mean([d[4] > thr for d in T if d[1]]); fa = np.mean([d[4] > thr for d in T if not d[1]])
        res.append((det, fa))
    res = np.array(res)
    det_f = np.mean([d[2] for d in data if d[1]]); det_c = np.mean([d[3] for d in data if d[1]]); fa_f = np.mean([d[2] for d in data if not d[1]])
    out[J] = {"det_fresh": det_f, "det_closed": det_c, "det_closed_matched": float(res[:, 0].mean()), "det_closed_matched_sd": float(res[:, 0].std()),
              "fa_fresh": fa_f, "fa_closed_matched": float(res[:, 1].mean()), "n_viol": int(sum(d[1] for d in data))}
    r = out[J]
    print(f"{J:12s} {det_f:9.3f} {det_c:10.3f} {r['det_closed_matched']:17.3f} ±{r['det_closed_matched_sd']:.3f} {fa_f:8.3f} {r['fa_closed_matched']:17.3f}")
json.dump(out, open(ROOT / "runs/ci/matched_fa.json", "w"), indent=1)
