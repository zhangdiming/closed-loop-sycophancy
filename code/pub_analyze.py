import argparse, collections, glob, json, numpy as np
from bootstrap_ci import load_rows, labels, ROOT

def auroc(pos, neg):
    if not len(pos) or not len(neg): return float("nan")
    a = np.asarray(pos)[:, None]; b = np.asarray(neg)[None, :]
    return float(((a < b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judges", nargs="*", default=["bagel", "janus7b", "omnigen2"])
    ap.add_argument("--json"); a = ap.parse_args()
    facts = {r["pid"]: [f["fid"] for f in r["facts"]] for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl"))}
    allrows = load_rows(str(ROOT / "runs/judge")); res = {}
    lab, _ = labels([r for r in allrows if r["judge"] in ("qwen3vl8b", "owlv2")], "human")
    rng = np.random.default_rng(0)
    ef = ROOT / "runs/pub/entail_scores.json"; ent = json.load(open(ef)) if ef.exists() else {}
    for J in a.judges:
        rows = {}
        for f in glob.glob(str(ROOT / f"runs/pub/{J}.jsonl")) + glob.glob(str(ROOT / f"runs/pub/{J}_p[0-9].jsonl")):
            for l in open(f): r = json.loads(l); rows[(r["image"], r["protocol"])] = r
        pid_of = {im: r["pid"] for (im, _), r in rows.items()}
        truth = {}
        for im, pid in pid_of.items():
            ls = [lab.get((J, im, fid)) for fid in facts[pid]]
            if None in ls: continue
            truth[im] = int("no" in ls)
        ref = collections.defaultdict(dict)
        for r in allrows:
            if r["judge"] != J or r["gen"] != J or r.get("perturb") or r["image"] not in truth: continue
            c = "atomic_fresh" if not r.get("context") else ("atomic_cl" if r.get("context_mode") in (None, "assistant") else None)
            if c: ref[(r["image"], c)][r["fid"]] = r["p_yes"]
        score, flag = collections.defaultdict(dict), collections.defaultdict(dict); nkept = collections.defaultdict(list)
        sf = ROOT / f"runs/pub/entail_self_{J}.json"; sent = json.load(open(sf)) if sf.exists() else {}
        for (im, c), d in ref.items():
            if len(d) == len(facts[pid_of[im]]): score[c][im] = float(np.mean(list(d.values()))); flag[c][im] = int(min(d.values()) < 0.5)
        for (im, p), r in rows.items():
            if im not in truth: continue
            if p == "ug_out": score[p][im] = r["p_yes"]; flag[p][im] = int(r["p_yes"] < 0.5)
            elif p.startswith("dq"):
                if not r["p_yes"]: continue
                score[p][im] = float(np.mean(r["p_yes"])); flag[p][im] = int(min(r["p_yes"]) < 0.5)
                if ent:
                    kept = [py for q, py in zip(r["questions"], r["p_yes"]) if ent.get(pid_of[im], {}).get(q, 0) >= 0.5]
                    score[p + "_ent"][im] = float(np.mean(kept)) if kept else 1.0; flag[p + "_ent"][im] = int(bool(kept) and min(kept) < 0.5)
                    nkept[p].append(len(kept))
                if sent and p in ("dq_fresh", "dq_req", "dq_cl"):
                    kept = [py for q, py in zip(r["questions"], r["p_yes"]) if sent.get(pid_of[im], {}).get(q, 0) >= 0.5]
                    score[p + "_sent"][im] = float(np.mean(kept)) if kept else 1.0; flag[p + "_sent"][im] = int(bool(kept) and min(kept) < 0.5)
                    nkept[p + "_sent"].append(len(kept))
            elif p == "ov":
                v = r.get("answer"); v = {True: 1, False: 0, "true": 1, "false": 0}.get(v if not isinstance(v, str) else v.lower())
                if v is None: continue
                score[p][im] = v; flag[p][im] = 1 - v
            elif p == "uc":
                if r.get("score") is None: continue
                score[p][im] = r["score"]; flag[p][im] = int(r["score"] <= 6)
            elif p == "cotv":
                if r.get("final") not in ("yes", "no"): continue
                score[p][im] = float(r["final"] == "yes") if r.get("frac_yes") is None else r["frac_yes"]; flag[p][im] = int(r["final"] == "no")
        if not truth: print(J, "no rows yet"); continue
        pids = sorted({pid_of[im] for im in truth}); ix = {p: i for i, p in enumerate(pids)}
        W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=2000)
        out = {"n_images": len(truth), "n_flawed": sum(truth.values()), "protocols": {}}
        print(f"\n== {J}: {len(truth)} labelled own images, {sum(truth.values())} flawed")
        for p in ["atomic_fresh", "atomic_cl", "ug_out", "ov", "uc", "cotv", "dq_fresh", "dq_req", "dq_cl", "dq_ext", "dq_fresh_ent", "dq_req_ent", "dq_cl_ent", "dq_ext_ent", "dq_fresh_sent", "dq_req_sent", "dq_cl_sent"]:
            ims = [im for im in score[p]]
            if not ims: continue
            n_all = sum(1 for (im, pp) in rows if pp == p.replace("_sent", "").replace("_ent", "") and im in truth) if not p.startswith("atomic") else len(ims)
            g = np.array([ix[pid_of[im]] for im in ims]); t = np.array([truth[im] for im in ims]); s = np.array([score[p][im] for im in ims]); fl = np.array([flag[p][im] for im in ims])
            det = fl[t == 1].mean() if (t == 1).any() else float("nan"); fa = fl[t == 0].mean() if (t == 0).any() else float("nan")
            au = auroc(s[t == 1], s[t == 0])
            def det_at(sv, tv, fa=0.05):
                neg, pos = np.sort(sv[tv == 0]), sv[tv == 1]
                if len(np.unique(sv)) < 5 or not len(neg) or not len(pos): return float("nan")
                thr = neg[int(np.floor(fa * len(neg)))]
                return float((pos < thr).mean())
            d5 = det_at(s, t)
            bd, bf, ba, b5 = [], [], [], []
            for w in W:
                wi = w[g]; m1, m0 = (t == 1), (t == 0)
                bd.append((wi * fl * m1).sum() / max((wi * m1).sum(), 1)); bf.append((wi * fl * m0).sum() / max((wi * m0).sum(), 1))
                rep = np.repeat(np.arange(len(ims)), wi); ba.append(auroc(s[rep][t[rep] == 1], s[rep][t[rep] == 0])); b5.append(det_at(s[rep], t[rep]))
            ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
            out["protocols"][p] = {"n": len(ims), "parsed": len(ims) / n_all if n_all else None, "det": float(det), "det_ci": ci(bd),
                                   "fa": float(fa), "fa_ci": ci(bf), "auroc": au, "auroc_ci": ci(ba), "det_fa5": d5, "det_fa5_ci": ci(b5) if d5 == d5 else None,
                                   "mean_flawed": float(s[t == 1].mean()), "mean_correct": float(s[t == 0].mean())}
            q = out["protocols"][p]
            print(f"{p:13s} n={q['n']:4d} parsed={q['parsed']:.2f}  det {det:.3f} [{q['det_ci'][0]:.2f},{q['det_ci'][1]:.2f}]  "
                  f"FA {fa:.3f}  det@FA5 {d5:.2f}  AUROC {au:.3f} [{q['auroc_ci'][0]:.2f},{q['auroc_ci'][1]:.2f}]  mean score flawed/correct {q['mean_flawed']:.2f}/{q['mean_correct']:.2f}")
        out["questions_kept"] = {k: float(np.mean(v)) for k, v in nkept.items()}
        print("mean questions kept per image after the entailment filter:", out["questions_kept"])
        res[J] = out
    if a.json: json.dump(res, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
