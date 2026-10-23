import json, collections, pathlib, math
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
facts = {r["pid"]: r["facts"] for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
def load(d, j):
    o = {}
    for f in (d / "oracle").glob(f"{j}__round*.jsonl"):
        rd = f.stem.split("__")[1]
        for r in map(json.loads, open(f)): o[(rd, r["image"], r["fid"])] = r["answer"]
    return o
def status(name, rule):
    d = ROOT / "runs/repair" / name; q, w = load(d, "qwen3vl8b"), load(d, "owlv2"); out = {}
    for t in map(json.loads, open(d / "trace.jsonl", encoding="utf-8")):
        rd, im = t["rounds"][-1]["image"].split("/"); ok = True
        for fc in facts[t["pid"]]:
            a = q.get((rd, im, fc["fid"])); b = w.get((rd, im, fc["fid"]), a) if fc["type"] != "color" else a
            if a is None: ok = None; break
            bad = (a == "no") if rule == "oracle" else (a == "no" and b == "no") if rule == "agree" else (a == "no" or b == "no")
            ok = ok and not bad
        if ok is not None: out[t["pid"]] = ok
    return out
res = {}
for g, ns in {"bagel": ["bagel__bagel", "bagel__bagel__ctx", "bagel__bagel__ctx_thr"], "janus7b": ["janus7b__janus7b", "janus7b__janus7b__ctx"],
              "showo2_7b": ["showo2_7b__showo2_7b", "showo2_7b__showo2_7b__ctx"],
              "bagel_dq": ["bagel__bagel_dq", "bagel__bagel__ctx_dq"], "bagel_dqt": ["bagel__bagel_dqt", "bagel__bagel__ctx_dqt"],
              "omnigen2_dqt": ["omnigen2__omnigen2_dqt", "omnigen2__omnigen2__ctx_dqt"], "janus7b_dqt": ["janus7b__janus7b_dqt", "janus7b__janus7b__ctx_dqt"],
              "bagel_always": ["bagel__bagel", "bagel__none"], "bagel_always_vs_ctx": ["bagel__none", "bagel__bagel__ctx"]}.items():
    for rule in ("oracle", "agree", "either"):
        st = {n: status(n, rule) for n in ns}; common = sorted(set.intersection(*[set(v) for v in st.values()]))
        fr = np.array([st[ns[0]][p] for p in common], float)
        for n in ns[1:]:
            cl = np.array([st[n][p] for p in common], float); b01 = int(((fr == 1) & (cl == 0)).sum()); b10 = int(((fr == 0) & (cl == 1)).sum())
            nn = b01 + b10; k = min(b01, b10); p = min(1.0, 2 * sum(math.comb(nn, j) for j in range(k + 1)) / 2 ** nn) if nn else 1.0
            res[f"{g}:{rule}:{n}"] = {"n": len(common), "fresh": fr.mean(), "other": cl.mean(), "diff": fr.mean() - cl.mean(), "disc": [b01, b10], "p": p}
            print(f"{g:10s} {rule:6s} n={len(common)} fresh {fr.mean():.3f}  {n:24s} {cl.mean():.3f}  diff {fr.mean()-cl.mean():+.3f}  disc {b01} vs {b10}  p={p:.4f}")
json.dump(res, open(ROOT / "runs/ci/repair_strict.json", "w"), indent=1)
