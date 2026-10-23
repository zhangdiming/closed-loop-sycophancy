import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, GEN_OF

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p) - np.log1p(-p)

ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human"); out = {}
for J in ("bagel", "showo2_7b"):
    P = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] != J or r.get("p_yes") is None or r["gen"] == GEN_OF.get(J, J): continue
        c = cond_of(r)
        if c in ("fresh", "ctx"): P[c][(r["gen"], r["image"], r["fid"])] = r["p_yes"]
    ks = [k for k in P["fresh"] if k in P["ctx"]]
    lf = logit([P["fresh"][k] for k in ks]); lc = logit([P["ctx"][k] for k in ks])
    icpt, slope = np.linalg.lstsq(np.c_[np.ones(lf.size), lf], lc - lf, rcond=None)[0]
    lab = [o.get(k) for k in ks]; vi = [i for i, l in enumerate(lab) if l == "no"]; si = [i for i, l in enumerate(lab) if l == "yes"]
    rng = np.random.default_rng(0); si = sorted(rng.choice(si, min(1500, len(si)), replace=False).tolist())
    out[J] = {"slope": float(slope), "icpt": float(icpt), "violated": [[float(lf[i]), float(lc[i])] for i in vi],
              "satisfied": [[float(lf[i]), float(lc[i])] for i in si]}
    print(J, len(vi), len(si), round(-slope, 3), round(icpt, 3))
json.dump(out, open(a.json, "w"))
