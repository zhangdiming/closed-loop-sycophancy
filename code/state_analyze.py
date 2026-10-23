import argparse, collections, json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
CONDS = ["vit_fresh", "vit_cl", "vae_fresh", "vae_cl", "state_fresh", "state_cl", "statevae_fresh", "statevae_cl"]
DIFFS = [("request, ViT (main)", "vit_cl", "vit_fresh"), ("request, re-encoded VAE+ViT", "vae_cl", "vae_fresh"),
         ("request, run state (latent+ViT)", "state_cl", "state_fresh"), ("request, run latent only", "statevae_cl", "statevae_fresh"),
         ("state vs re-encoded, fresh", "state_fresh", "vae_fresh"), ("state vs re-encoded, closed loop", "state_cl", "vae_cl"),
         ("state vs ViT, closed loop", "state_cl", "vit_cl")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
    R = collections.defaultdict(dict); lab = {}; pid = {}; mass = collections.defaultdict(list)
    for l in open(ROOT / "runs/state/bagel_state.jsonl"):
        r = json.loads(l); k = (r["image"], r["fid"])
        if r["p_yes"] is None: continue
        R[k][r["cond"]] = r["p_yes"] < 0.5; lab[k] = r["label"]; pid[k] = r["pid"]; mass[r["cond"]].append(r["mass"])
    keys = [k for k in R if all(c in R[k] for c in CONDS) and lab[k] in ("yes", "no")]
    regen = [json.loads(l) for l in open(ROOT / "runs/state/regen.jsonl")]
    pids = sorted({pid[k] for k in keys}); P = {p: i for i, p in enumerate(pids)}
    rng = np.random.default_rng(0); boot = rng.integers(0, len(pids), size=(a.B, len(pids)))
    res = {"n_images": len({k[0] for k in keys}), "n_viol": sum(lab[k] == "no" for k in keys), "n_sat": sum(lab[k] == "yes" for k in keys),
           "regen_identical": sum(r["max_abs_diff"] == 0 for r in regen), "regen_total": len(regen),
           "mass": {c: float(np.mean(mass[c])) for c in CONDS}, "det": {}, "fa": {}, "diff": {}}
    for which, want in (("det", "no"), ("fa", "yes")):
        ks = [k for k in keys if lab[k] == want]; idx = np.array([P[pid[k]] for k in ks])
        X = {c: np.array([R[k][c] for k in ks], float) for c in CONDS}
        cnt = np.stack([np.bincount(b, minlength=len(pids)) for b in boot]).astype(float)
        w = cnt[:, idx]
        def bs(v): e = (w * v).sum(1) / w.sum(1); return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
        for c in CONDS: res[which][c] = {"est": float(X[c].mean()), "ci95": bs(X[c])}
        for name, c1, c2 in DIFFS:
            d = X[c1] - X[c2]; res["diff"].setdefault(name, {})[which] = {"est": float(d.mean()), "ci95": bs(d)}
    print(f"images {res['n_images']}  violated facts {res['n_viol']}  satisfied {res['n_sat']}  regenerated identical {res['regen_identical']}/{res['regen_total']}")
    for c in CONDS:
        print(f"{c:16s} det {res['det'][c]['est']:.3f} [{res['det'][c]['ci95'][0]:.3f},{res['det'][c]['ci95'][1]:.3f}]  FA {res['fa'][c]['est']:.3f}  mass {res['mass'][c]:.3f}")
    for name, _, _ in DIFFS:
        d = res["diff"][name]
        print(f"{name:34s} det {d['det']['est']:+.3f} [{d['det']['ci95'][0]:+.3f},{d['det']['ci95'][1]:+.3f}]  FA {d['fa']['est']:+.3f} [{d['fa']['ci95'][0]:+.3f},{d['fa']['ci95'][1]:+.3f}]")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
