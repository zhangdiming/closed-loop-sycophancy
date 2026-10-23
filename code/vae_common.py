import argparse, collections, json
import numpy as np
from bootstrap_ci import labels, load_rows, ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human"); A = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] not in ("bagel", "bagel_vae") or r["gen"] not in ("bagel", "janus7b", "showo2_7b") or r.get("perturb"): continue
        c = "fresh" if not r.get("context") else ("ctx" if r.get("context_mode") in (None, "assistant") else None)
        if c and r["p_yes"] is not None: A[(r["judge"], c)][(r["gen"], r["image"], r["fid"])] = (r["pid"], r["p_yes"] < 0.5)
    res = {}; rng = np.random.default_rng(0)
    for side, gens in (("own", {"bagel"}), ("oth", {"janus7b", "showo2_7b"})):
        keys = [k for k in A[("bagel", "fresh")] if k[0] in gens and o.get(k) == "no" and all(k in A[(j, c)] for j in ("bagel", "bagel_vae") for c in ("fresh", "ctx"))]
        pid = np.array([A[("bagel", "fresh")][k][0] for k in keys]); X = {(j, c): np.array([A[(j, c)][k][1] for k in keys], float) for j in ("bagel", "bagel_vae") for c in ("fresh", "ctx")}
        cv = X[("bagel", "fresh")] - X[("bagel", "ctx")]; ce = X[("bagel_vae", "fresh")] - X[("bagel_vae", "ctx")]
        P = sorted(set(pid)); ix = {p: np.where(pid == p)[0] for p in P}; rb = []
        for _ in range(2000):
            s = np.concatenate([ix[p] for p in rng.choice(P, len(P))]); rb.append(ce[s].mean() / cv[s].mean())
        res[side] = {"n": len(keys), "vit": [float(X[("bagel", "fresh")].mean()), float(X[("bagel", "ctx")].mean())],
                     "vae": [float(X[("bagel_vae", "fresh")].mean()), float(X[("bagel_vae", "ctx")].mean())],
                     "cost_vit": float(cv.mean()), "cost_vae": float(ce.mean()), "ratio": float(ce.mean() / cv.mean()),
                     "ratio_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]}
        print(side, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in res[side].items()}))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
