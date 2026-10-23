import argparse, collections, json
from bootstrap_ci import load_rows, labels, cond_of, ROOT

JUDGES = ["bagel", "blip3o", "janus1b", "janus7b", "lumina", "omnigen2", "showo2_1.5b", "showo2_7b"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    S = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if r["judge"] not in JUDGES or r.get("p_yes") is None: continue
        c = cond_of(r)
        if c not in ("fresh", "ctx") or o.get((r["gen"], r["image"], r["fid"])) != "no": continue
        d, p = r["answer"] == "no", r["p_yes"] < 0.5; s = S[(r["judge"], c)]
        s[0] += 1; s[1] += d == p; s[2] += d; s[3] += p
    res = {}
    for (j, c), (n, ag, d, p) in sorted(S.items()):
        res[f"{j}:{c}"] = {"n": n, "agree": ag / n, "det_decoded": d / n, "det_pyes": p / n}
        print(f"{j:12s} {c:6s} n {n:5d} agree {ag / n:.3f}  det decoded {d / n:.3f}  p(yes)<0.5 {p / n:.3f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
