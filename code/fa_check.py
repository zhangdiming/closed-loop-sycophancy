import json, glob, sys
oracle = {}
for f in glob.glob("runs/judge/qwen3vl8b__*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if not r.get("context"): oracle[(r["gen"], r["image"], r["fid"])] = r["answer"]
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
print(f"{'run':55s} {'n_sat':>6s} {'FA':>6s} {'n_viol':>6s} {'recall':>6s}")
for f in sorted(glob.glob(f"runs/judge/{pat}.jsonl")):
    if "qwen3vl8b__" in f or "owlv2__" in f: continue
    rows = [json.loads(l) for l in open(f)]
    sat = [r for r in rows if oracle.get((r["gen"], r["image"], r["fid"])) == "yes"]
    viol = [r for r in rows if oracle.get((r["gen"], r["image"], r["fid"])) == "no"]
    if not sat: continue
    print(f"{f.split('/')[-1]:55s} {len(sat):6d} {sum(r['answer']=='no' for r in sat)/len(sat):6.3f} {len(viol):6d} {sum(r['answer']=='no' for r in viol)/max(1,len(viol)):6.3f}")
