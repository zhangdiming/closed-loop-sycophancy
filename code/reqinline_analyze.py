import argparse, json
import numpy as np
from bootstrap_ci import labels, load_rows, ROOT

OWN = {"showo2_7b": "showo2_7b", "janus7b": "janus7b", "bagel": "bagel"}
CF = {"fresh": "", "ctx": "__ctx", "caption": "__ctx-caption_inline", "request": "__ctx-request_inline"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    o, _ = labels(load_rows(ROOT / "runs/judge"), "human"); res = {}; rng = np.random.default_rng(0)
    for J in ("showo2_7b", "janus7b", "bagel"):
        for side in ("own", "oth"):
            gens = [g for g in ("janus7b", "showo2_7b") if (g == OWN[J]) == (side == "own")]
            if not gens: continue
            D = {c: {} for c in CF}
            for g in gens:
                for c, suf in CF.items():
                    f = ROOT / f"runs/judge/{J}__{g}{suf}.jsonl"
                    for r in (json.loads(l) for l in open(f)):
                        if r.get("perturb") or r["p_yes"] is None: continue
                        D[c][(g, r["image"], r["fid"])] = (r["pid"], r["answer"] == "no")
            keys = [k for k in D["request"] if all(k in D[c] for c in CF)]
            out = {}
            for want, key in (("no", "det"), ("yes", "fa")):
                ks = [k for k in keys if o.get(k) == want]
                if not ks: continue
                pid = np.array([D["fresh"][k][0] for k in ks]); X = {c: np.array([D[c][k][1] for k in ks], float) for c in CF}
                d = X["request"] - X["caption"]; P = sorted(set(pid)); ix = {p: np.where(pid == p)[0] for p in P}; bs = []
                for _ in range(2000):
                    s = np.concatenate([ix[p] for p in rng.choice(P, len(P))]); bs.append(d[s].mean())
                out[key] = {"n": len(ks), **{c: float(X[c].mean()) for c in CF}, "request_minus_caption": float(d.mean()),
                            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            res[f"{J}:{side}"] = out
            dd = out["det"]; fa = out.get("fa", {})
            print(f"{J:10s} {side}  n={dd['n']:4d}  det fresh {dd['fresh']:.3f} closed {dd['ctx']:.3f} caption {dd['caption']:.3f} request {dd['request']:.3f}  "
                  f"req-cap {dd['request_minus_caption']:+.3f} [{dd['ci95'][0]:+.3f},{dd['ci95'][1]:+.3f}]  FA cap {fa.get('caption', float('nan')):.3f} req {fa.get('request', float('nan')):.3f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
