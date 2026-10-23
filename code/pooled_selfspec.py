import argparse, collections, json
import numpy as np
from bootstrap_ci import load_rows, labels, answers, GEN_OF, ROOT

GROUPS = {"all8": ["bagel", "blip3o", "janus1b", "janus7b", "lumina", "omnigen2", "showo2_1.5b", "showo2_7b"],
          "strong4": ["bagel", "janus1b", "omnigen2", "showo2_1.5b"], "less4": ["blip3o", "janus7b", "lumina", "showo2_7b"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human"); ans = answers(rows)
    H = collections.defaultdict(lambda: collections.defaultdict(float)); N = collections.defaultdict(lambda: collections.defaultdict(float))
    for (j, cond, g), d in ans.items():
        if j not in GROUPS["all8"] or cond not in ("fresh", "ctx"): continue
        side = "own" if g == GEN_OF.get(j, j) else "oth"
        for (im, fid), (pid, an) in d.items():
            if o.get((g, im, fid)) == "no":
                H[(j, side, cond)][pid] += an == "no"; N[(j, side, cond)][pid] += 1
    pids = sorted({p for c in N.values() for p in c}); ix = {p: i for i, p in enumerate(pids)}
    cells = sorted(N); Hm = np.zeros((len(cells), len(pids))); Nm = np.zeros_like(Hm)
    for k, c in enumerate(cells):
        for p, v in N[c].items(): Nm[k, ix[p]] = v; Hm[k, ix[p]] = H[c][p]
    ci = {c: k for k, c in enumerate(cells)}
    def terms(w):
        R = (w @ Hm.T) / (w @ Nm.T)
        out = {}
        for j in GROUPS["all8"]:
            r = lambda s, c: R[:, ci[(j, s, c)]]
            out[j] = (r("own", "ctx") - r("own", "fresh")) - (r("oth", "ctx") - r("oth", "fresh"))
        return out
    point = terms(np.ones((1, len(pids))))
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=a.B).astype(float)
    bs = terms(W)
    res = {"per_judge": {j: float(point[j][0]) for j in GROUPS["all8"]}}
    for g, js in GROUPS.items():
        est = float(np.mean([point[j][0] for j in js])); dist = np.mean([bs[j] for j in js], axis=0)
        c95 = [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]; c90 = [float(np.percentile(dist, 5)), float(np.percentile(dist, 95))]
        res[g] = {"judges": js, "est": est, "ci95": c95, "ci90": c90, "equivalent_0.10": bool(c90[0] > -0.10 and c90[1] < 0.10),
                  "deficit_0.10_rejected": bool(c90[0] > -0.10), "p_one_sided_deficit": float((dist <= 0).mean()) if est > 0 else float((dist >= 0).mean())}
        print(f"{g:8s} mean self-specific {est:+.3f}  95% [{c95[0]:+.3f}, {c95[1]:+.3f}]  90% [{c90[0]:+.3f}, {c90[1]:+.3f}]  equivalent(0.10) {res[g]['equivalent_0.10']}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
