import argparse, collections, json
from bootstrap_ci import load_rows, labels, answers, GEN_OF, ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge_dpg"); o, _ = labels(rows, "vlm"); ans = answers(rows)
    S = collections.defaultdict(lambda: [0, 0])
    for (j, c, g), d in ans.items():
        if c not in ("fresh", "ctx"): continue
        side = "own" if g == GEN_OF.get(j, j) else "oth"
        for (im, fid), (pid, an) in d.items():
            if o.get((g, im, fid)) == "yes": s = S[(j, side, c)]; s[0] += 1; s[1] += an == "no"
    res = {f"{j}:{side}:{c}": {"n": n, "fa": k / n} for (j, side, c), (n, k) in sorted(S.items())}
    for k, v in res.items(): print(f"{k:28s} n {v['n']:5d}  FA {v['fa']:.3f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
