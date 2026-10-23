import collections, glob, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
JUDGES = ["bagel", "blip3o", "janus1b", "janus7b", "lumina", "omnigen2", "showo2_1.5b", "showo2_7b", "qwen25vl7b"]

def main():
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    S = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0])
    for f in glob.glob(str(ROOT / "runs/judge/*.jsonl")):
        name = pathlib.Path(f).stem.split("__")
        if name[0] not in JUDGES: continue
        ctx = "fresh" if len(name) == 2 else ("closed" if name[2] == "ctx" else None)
        if ctx is None: continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("p_yes") is None or r.get("mass") is None: continue
            s = S[(name[0], ctx)]; s[0] += 1; s[1] += r["mass"]; s[2] += r["mass"] < 0.5
            dec_no = r["answer"] == "no"; s[3] += dec_no == (r["p_yes"] < 0.5); s[4] += r["answer"] not in ("yes", "no")
    res = {}
    print(f"{'judge':12s} {'ctx':7s} {'n':>7s} {'mass':>6s} {'mass<.5':>8s} {'agree':>6s} {'offfmt':>7s}")
    for (j, c), (n, m, lo, ag, off) in sorted(S.items()):
        res[f"{j}:{c}"] = {"n": n, "mass": m / n, "low_mass": lo / n, "agree": ag / n, "off_format": off / n}
        print(f"{j:12s} {c:7s} {n:7d} {m/n:6.3f} {lo/n:8.3f} {ag/n:6.3f} {off/n:7.3f}")
    if out: json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
