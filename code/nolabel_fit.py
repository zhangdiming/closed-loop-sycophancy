import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path.cwd()))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, GEN_OF
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p) - np.log1p(-p)
rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
P = collections.defaultdict(dict)
for r in rows:
    if r["judge"] in ("qwen3vl8b", "owlv2") or r.get("p_yes") is None: continue
    c = cond_of(r)
    if c not in ("fresh", "ctx"): continue
    P[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = r["p_yes"]
out = {}
for J in ["bagel", "janus1b", "omnigen2", "showo2_1.5b", "janus7b", "blip3o", "lumina", "showo2_7b"]:
    own = GEN_OF.get(J, J); res = {}
    for name, need in (("labelled", True), ("all", False)):
        lf, lc = [], []
        for (j, cc, g), d in P.items():
            if j != J or cc != "fresh" or g == own or (J, "ctx", g) not in P: continue
            dc = P[(J, "ctx", g)]
            for k, py in d.items():
                if k not in dc or (need and o.get((g,) + k) is None): continue
                lf.append(py); lc.append(dc[k])
        lf, lc = logit(lf), logit(lc)
        b, s = np.linalg.lstsq(np.c_[np.ones(lf.size), lf], lc - lf, rcond=None)[0]
        res[name] = dict(n=int(lf.size), lam=float(-s), icpt=float(b), m=float(b / (1 + s)))
    out[J] = res; print(J, res, flush=True)
json.dump(out, open("runs/ci_r18/nolabel_fit.json", "w"), indent=1)
