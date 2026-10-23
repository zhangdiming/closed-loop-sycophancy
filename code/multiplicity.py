import argparse, json, math, os


def p_deficit(est, ci95):
    se = (ci95[1] - ci95[0]) / (2 * 1.959964)
    return 0.5 * math.erfc(-(est / se) / math.sqrt(2)) if se > 0 else (0.0 if est < 0 else 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ci"); ap.add_argument("--json"); a = ap.parse_args()
    ld = lambda f: json.load(open(os.path.join(a.ci, f)))
    T = []
    def add(name, d):
        if isinstance(d, dict) and "est" in d and "ci95" in d: T.append((name, d["est"], d["ci95"]))
    for L in ("human", "hybrid", "strict", "vlm"):
        for j, v in ld(f"did_{L}.json").items():
            for t in ("self_fresh", "selfspec"):
                if L == "human" and t == "selfspec" and j != "bagel_vae": continue
                add(f"did/{L}/{j}/{t}", v.get(t))
        for j, v in ld(f"stdid_{L}.json").items():
            for t in ("selfspec_std", "self_fresh_std"): add(f"stdid/{L}/{j}/{t}", v.get(t))
        for j, v in ld(f"matched_{L}.json").items():
            for t in ("self_fresh", "selfspec"): add(f"matched/{L}/{j}/{t}", v.get(t))
    for k, v in ld("variant_selfspec.json").items():
        for t in ("raw", "std", "residual"): add(f"variant/{k}/{t}", v.get(t))
    for j, v in ld("pool_extras.json")["selfspec"].items():
        T.append((f"pyes/{j}/observed", v["observed"], v["observed_ci95"])); T.append((f"pyes/{j}/residual", v["residual"], v["residual_ci95"]))
    MATCHED = ("state vs re-encoded", "state vs reenc,", "statevit vs reencvit", "state vs main")
    for f in ("state_probe.json", "state_janus7b.json", "state_janus1b.json", "state_showo2_1.5b.json", "state_showo2_7b.json"):
        if not os.path.exists(os.path.join(a.ci, f)): continue
        d = ld(f); rules = d.get("rules", {"pyes": d})
        for rule, r in rules.items():
            for name, x in r.get("diff", {}).items():
                if name.startswith(MATCHED): add(f"state/{f[:-5]}/{rule}/{name}", x.get("det"))
    for k, v in ld("exhaustive.json")["did"].items():
        if k.startswith("exhaustive:"):
            for t in ("self_fresh", "selfspec"): add(f"exhaustive/{k[11:]}/{t}", v.get(t))
    P = sorted(((p_deficit(e, c), n, e, c) for n, e, c in T), key=lambda x: x[0]); m = len(P)
    holm = [min(1.0, max((m - i) * P[i][0] for i in range(k + 1))) for k in range(m)]
    bh = [0.0] * m; run = 1.0
    for k in range(m - 1, -1, -1): run = min(run, P[k][0] * m / (k + 1)); bh[k] = run
    rows = [{"test": n, "est": e, "ci95": c, "p_one_sided": p, "p_holm": h, "q_bh": q} for (p, n, e, c), h, q in zip(P, holm, bh)]
    res = {"m": m, "n_p05": sum(r["p_one_sided"] < 0.05 for r in rows), "holm_05": [r for r in rows if r["p_holm"] < 0.05],
           "bh_05": [r for r in rows if r["q_bh"] < 0.05], "all": rows}
    print(f"secondary own-image tests: {m}; one-sided p<0.05: {res['n_p05']}; Holm<0.05: {len(res['holm_05'])}; BH q<0.05: {len(res['bh_05'])}")
    for r in rows[:25]:
        print(f"  {r['test']:55s} {r['est']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p={r['p_one_sided']:.2e} holm={r['p_holm']:.3f} q={r['q_bh']:.3f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
