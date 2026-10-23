import argparse, collections, glob, json, pathlib, random
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/dpo/train"); ap.add_argument("--oracle"); a = ap.parse_args()
    J = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in glob.glob(str(ROOT / a.dir / "judge_p*.jsonl")):
        for r in map(json.loads, open(f)): J[r["ctx"]][r["image"]][r["fid"]] = r["answer"]
    prompts = {r["pid"]: r for r in map(json.loads, open(ROOT / "data/train_prompts.jsonl", encoding="utf-8"))}
    O = None
    if a.oracle:
        O = collections.defaultdict(dict)
        for r in map(json.loads, open(ROOT / a.oracle)): O[r["image"]][r["fid"]] = r["answer"]
    pairs = {}
    for ctx in ("fresh", "closed"):
        P = []
        for pid, pr in sorted(prompts.items()):
            ims = [f"{pid}_s{s}.png" for s in range(4)]; fids = [x["fid"] for x in pr["facts"]]
            if not all(all(fd in J[ctx].get(i, {}) for fd in fids) for i in ims): continue
            sc = [sum(J[ctx][i][fd] == "yes" for fd in fids) for i in ims]
            if max(sc) == min(sc): continue
            P.append({"pid": pid, "prompt": pr["prompt"], "chosen": ims[sc.index(max(sc))], "rejected": ims[sc.index(min(sc))]})
        pairs[ctx] = P
    pairs["fresh_sub"] = random.Random(0).sample(pairs["fresh"], min(len(pairs["fresh"]), len(pairs["closed"])))
    for k, P in pairs.items():
        json.dump(P, open(ROOT / f"runs/dpo/pairs_{k}.json", "w"), indent=0)
        msg = f"{k:10s} {len(P)} pairs"
        if O:
            c = t = inv = 0
            for p in P:
                fids = [x["fid"] for x in prompts[p["pid"]]["facts"]]
                oc, orj = (sum(O[p[s]].get(fd) == "yes" for fd in fids) for s in ("chosen", "rejected"))
                c += oc > orj; t += oc == orj; inv += oc < orj
            msg += f"  oracle: correct {c/len(P):.3f} tied {t/len(P):.3f} inverted {inv/len(P):.3f}"
        print(msg)

if __name__ == "__main__":
    main()
