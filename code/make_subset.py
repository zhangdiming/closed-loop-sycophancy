import argparse, json, pathlib, random
ROOT = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--gens", nargs="+"); ap.add_argument("--n_sat", type=int, default=800)
ap.add_argument("--out", default=str(ROOT / "data/subset_r2.json")); a = ap.parse_args()
sub = {}
for g in a.gens:
    rows = [json.loads(l) for l in open(ROOT / f"runs/judge/qwen3vl8b__{g}.jsonl", encoding="utf-8")]
    viol = sorted({(r["image"], r["fid"]) for r in rows if r["answer"] == "no"})
    sat = sorted({(r["image"], r["fid"]) for r in rows if r["answer"] == "yes"})
    random.Random(0).shuffle(sat)
    sub[g] = [list(k) for k in viol + sat[: a.n_sat]]
    print(f"{g}: {len(viol)} oracle violations + {min(a.n_sat, len(sat))} satisfied")
json.dump(sub, open(a.out, "w"))
