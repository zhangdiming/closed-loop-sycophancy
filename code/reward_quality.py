import argparse, collections, glob, itertools, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--oracle", default="qwen3vl8b"); ap.add_argument("--judges", nargs="*")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(str(ROOT / "runs/judge/*.jsonl")): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    def cond(r):
        if r.get("perturb"): return None
        return ("ctx" if r.get("context_mode") in (None, "assistant") else f"ctx-{r['context_mode']}") if r.get("context") else "fresh"
    score = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        c = cond(r)
        if c is None or r["judge"] == "owlv2": continue
        score[(r["judge"], c, r["gen"])][r["image"]][r["fid"]] = r["answer"]
    judges = a.judges or sorted({j for (j, _, _) in score if j != a.oracle})
    print(f"{'judge':12s} {'cond':22s} {'gen':12s} {'pairs':>6s} {'inverted':>9s} {'tied':>6s} {'correct':>8s}")
    for J in judges:
        for (j, c, g) in sorted(score):
            if j != J: continue
            ora = score.get((a.oracle, "fresh", g))
            if not ora: continue
            jd = score[(j, c, g)]
            by_pid = collections.defaultdict(list)
            for im in jd:
                if im in ora: by_pid[im.rsplit("_s", 1)[0]].append(im)
            n = inv = tie = ok = 0
            for pid, ims in by_pid.items():
                for a_, b_ in itertools.combinations(sorted(ims), 2):
                    fids = set(ora[a_]) & set(ora[b_]) & set(jd[a_]) & set(jd[b_])
                    if not fids: continue
                    oa, ob = sum(ora[a_][f] == "yes" for f in fids), sum(ora[b_][f] == "yes" for f in fids)
                    if oa == ob: continue
                    ja, jb = sum(jd[a_][f] == "yes" for f in fids), sum(jd[b_][f] == "yes" for f in fids)
                    n += 1
                    if ja == jb: tie += 1
                    elif (ja > jb) == (oa > ob): ok += 1
                    else: inv += 1
            if n >= 20: print(f"{J:12s} {c:22s} {g:12s} {n:6d} {inv/n:9.3f} {tie/n:6.3f} {ok/n:8.3f}")

if __name__ == "__main__":
    main()
