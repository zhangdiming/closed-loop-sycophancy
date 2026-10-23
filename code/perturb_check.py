import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, ROOT
P = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
rel = {f["fid"]: f["args"].get("relation") for r in P.values() for f in r["facts"]}
rows = [r for r in load_rows(ROOT / "runs/judge") if r["judge"] in ("janus7b", "qwen3vl8b", "owlv2")]
out = {}
for L in ("human", "vlm"):
    o, _ = labels(rows, L)
    ans = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] != "janus7b" or r["gen"] != "janus7b": continue
        if r.get("context") and r.get("context_mode") not in (None, "assistant"): continue
        cond = ("closed" if r.get("context") else "fresh") + ("+" + r["perturb"] if r.get("perturb") else "")
        ans[cond][(r["image"], r["fid"])] = (r["answer"], r["type"])
    print(f"labels={L}")
    for cond in sorted(ans, key=lambda c: (c.split("+")[0] != "fresh", c)):
        view = cond.split("+")[1] if "+" in cond else None
        keep = lambda k, t: not ((view == "flip" and rel.get(k[1]) in ("left of", "right of")) or (view == "gray" and t == "color"))
        base = ans[cond.split("+")[0]]
        keys = [k for k, (a_, t) in ans[cond].items() if keep(k, t) and k in base]
        v = [k for k in keys if o.get(("janus7b",) + k) == "no"]; s_ = [k for k in keys if o.get(("janus7b",) + k) == "yes"]
        det = np.mean([ans[cond][k][0] == "no" for k in v]); fa = np.mean([ans[cond][k][0] == "no" for k in s_])
        det0 = np.mean([base[k][0] == "no" for k in v]); fa0 = np.mean([base[k][0] == "no" for k in s_])
        out[f"{L}:{cond}"] = {"det": det, "fa": fa, "det_unperturbed_same_facts": det0, "fa_unperturbed_same_facts": fa0, "n_viol": len(v), "n_sat": len(s_)}
        print(f"  {cond:16s} det {det:.3f} (unperturbed, same facts {det0:.3f})  FA {fa:.3f} ({fa0:.3f})  n={len(v)}/{len(s_)}")
json.dump(out, open(ROOT / "runs/ci/perturb_check.json", "w"), indent=1)
