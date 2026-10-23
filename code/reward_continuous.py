import argparse, collections, glob, itertools, json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--judges", nargs="*", default=["bagel", "janus7b", "showo2_7b", "blip3o"])
ap.add_argument("--json"); a = ap.parse_args()
rows = []
for f in glob.glob(str(ROOT / "runs/judge/*.jsonl")): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
ora = {(r["gen"], r["image"], r["fid"]): r["answer"] for r in rows if r["judge"] == "qwen3vl8b" and not r.get("context")}
sc = collections.defaultdict(lambda: collections.defaultdict(dict))
pid_of = {}
for r in rows:
    if r["judge"] not in a.judges or r.get("perturb"): continue
    if r.get("context") and r.get("context_mode") not in (None, "assistant"): continue
    if r.get("p_yes") is None: continue
    sc[(r["judge"], "closed" if r.get("context") else "fresh", r["gen"])][r["image"]][r["fid"]] = r["p_yes"]; pid_of[(r["gen"], r["image"])] = r["pid"]
sumlogit = lambda ps: float(np.sum(np.log(np.clip(ps, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(ps, 1e-6, 1 - 1e-6))))
out = {}; rng = np.random.default_rng(0)
print(f"{'judge':10s} {'gen':10s} {'pairs':>6s}  {'fresh: correct':>22s}  {'closed: correct':>22s}  {'closed - fresh':>22s}")
for J in a.judges:
    for g in ["bagel", "janus7b", "showo2_7b"]:
        if (J, "fresh", g) not in sc or (J, "closed", g) not in sc: continue
        F, C = sc[(J, "fresh", g)], sc[(J, "closed", g)]
        byp = collections.defaultdict(list)
        for im in F:
            if im in C and (g, im) in pid_of: byp[pid_of[(g, im)]].append(im)
        pairs = collections.defaultdict(list)
        for pid, ims in byp.items():
            for i1, i2 in itertools.combinations(ims, 2):
                fids = sorted(set(F[i1]) & set(F[i2]) & set(C[i1]) & set(C[i2]))
                if not fids: continue
                o1 = sum(ora.get((g, i1, f)) == "yes" for f in fids); o2 = sum(ora.get((g, i2, f)) == "yes" for f in fids)
                if o1 == o2: continue
                better, worse = (i1, i2) if o1 > o2 else (i2, i1)
                def corr(S, agg):
                    sb, sw = agg([S[better][f] for f in fids]), agg([S[worse][f] for f in fids])
                    return 1.0 if sb > sw else (0.5 if sb == sw else 0.0)
                pairs[pid].append((corr(F, np.mean), corr(C, np.mean), corr(F, sumlogit), corr(C, sumlogit)))
        pids = sorted(pairs); P = [np.array(pairs[p]) for p in pids]
        tot = np.concatenate(P); est = tot.mean(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(pids), len(pids)); b = np.concatenate([P[i] for i in idx]); boots.append(b.mean(0))
        boots = np.array(boots); d = boots[:, 1] - boots[:, 0]; dl = boots[:, 3] - boots[:, 2]
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        out[f"{J}:{g}"] = {"pairs": int(len(tot)), "fresh": float(est[0]), "fresh_ci": ci(boots[:, 0]), "closed": float(est[1]), "closed_ci": ci(boots[:, 1]),
                           "diff": float(est[1] - est[0]), "diff_ci": ci(d),
                           "fresh_logit": float(est[2]), "closed_logit": float(est[3]), "diff_logit": float(est[3] - est[2]), "diff_logit_ci": ci(dl)}
        r = out[f"{J}:{g}"]
        print(f"{J:10s} {g:10s} {r['pairs']:6d}  {r['fresh']:.3f} [{r['fresh_ci'][0]:.2f},{r['fresh_ci'][1]:.2f}]  {r['closed']:.3f} [{r['closed_ci'][0]:.2f},{r['closed_ci'][1]:.2f}]  {r['diff']:+.3f} [{r['diff_ci'][0]:+.2f},{r['diff_ci'][1]:+.2f}]   sum-logit {r['fresh_logit']:.3f}->{r['closed_logit']:.3f} {r['diff_logit']:+.3f} [{r['diff_logit_ci'][0]:+.2f},{r['diff_logit_ci'][1]:+.2f}]")
if a.json: json.dump(out, open(a.json, "w"), indent=1)
