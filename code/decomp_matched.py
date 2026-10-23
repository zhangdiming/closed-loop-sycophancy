import json, glob, pathlib, collections, numpy as np, sys

ROOT=pathlib.Path(__file__).resolve().parent
rows=[]
for f in glob.glob(str(ROOT/"runs/judge/*.jsonl")): rows+=[json.loads(l) for l in open(f,encoding="utf-8")]
typ={(r["gen"],r["image"],r["fid"]):r["type"] for r in rows if r["judge"]=="qwen3vl8b"}
vlm={(r["gen"],r["image"],r["fid"]):r["answer"] for r in rows if r["judge"]=="qwen3vl8b" and not r.get("context")}
det={(r["gen"],r["image"],r["fid"]):r["answer"] for r in rows if r["judge"]=="owlv2"}
def labels(kind):
    o={}
    for k,v in vlm.items():
        t=typ[k]
        if kind=="vlm": o[k]=v
        elif t=="color": o[k]=v
        elif k in det and det[k]==v: o[k]=v
    if kind=="human":
        for g in sorted({k[0] for k in vlm}):
            lp,fp=ROOT/"data/human"/f"{g}_labels.json",ROOT/"runs/human"/f"{g}_flagged.json"
            if not (lp.exists() and fp.exists()): continue
            flagged=json.load(open(fp)); lab=json.load(open(lp))["labels"]
            for k in [k for k in list(o) if k[0]==g and typ[k]!="count" and o[k]=="no"]: o.pop(k)
            for i,f in enumerate(flagged):
                l=lab.get(str(i)); k=(g,f["image"],f["fid"])
                if l=="violated": o[k]="no"
                elif l=="satisfied": o[k]="yes"
    return o
def cond_of(r):
    if r.get("perturb"): return None
    return ("ctx" if r.get("context_mode") in (None,"assistant") else "ctx-"+r["context_mode"]) if r.get("context") else "fresh"
ans=collections.defaultdict(dict)
for r in rows:
    c=cond_of(r)
    if c is None or r["judge"] in ("qwen3vl8b","owlv2"): continue
    ans[(r["judge"],c,r["gen"])][(r["image"],r["fid"])]=r["answer"]
conds=["fresh","ctx-assistant_nocaption","ctx","ctx-user_prompt","ctx-caption_inline"]
for kind in ["human","vlm"]:
    o=labels(kind)
    for J,G in [("janus7b","janus7b"),("janus7b","showo2_7b"),("showo2_7b","showo2_7b"),("showo2_7b","janus7b")]:
        keys=set.intersection(*[set(ans[(J,c,G)]) for c in conds])
        viol=[k for k in keys if o.get((G,)+k)=="no"]; sat=[k for k in keys if o.get((G,)+k)=="yes"]
        det_=[np.mean([ans[(J,c,G)][k]=="no" for k in viol]) for c in conds]
        fa=[np.mean([ans[(J,c,G)][k]=="no" for k in sat]) for c in conds]
        print(kind,J,"on",G,"n_viol",len(viol),"n_sat",len(sat)," det:"," ".join(f"{x:.3f}" for x in det_)," FA:"," ".join(f"{x:.3f}" for x in fa))
