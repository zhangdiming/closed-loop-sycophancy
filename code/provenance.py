import argparse, collections, json, pathlib, statistics, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import load_rows, labels, cond_of, ROOT, _repair_status


def kappa(a, b, cats=("violated", "satisfied", "ambiguous")):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats); return (po - pe) / (1 - pe)


def annotator_agreement():
    A, C = [], []
    for fp in sorted((ROOT / "data/human").glob("*_kappa3_labels.json")):
        g = fp.name.split("_kappa3")[0]; c = json.load(open(fp))["labels"]; t = json.load(open(ROOT / f"data/human/{g}_todo_labels.json"))["labels"]
        for i, l in c.items():
            if i in t: A.append(t[i]); C.append(l)
    rng = np.random.default_rng(0); idx = np.arange(len(A)); bs = []
    for _ in range(2000):
        s = rng.choice(idx, len(idx)); bs.append(kappa([A[i] for i in s], [C[i] for i in s]))
    return {"n_both": len(A), "agree": float(np.mean([x == y for x, y in zip(A, C)])), "kappa": kappa(A, C),
            "kappa_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def entailment():
    o = json.load(open(ROOT / "runs/pub/entail_scores.json")); q = json.load(open(ROOT / "runs/pub/entail_qwen25vl7b_text.json"))
    s = json.load(open(ROOT / "runs/pub/entail_self_omnigen2.json"))
    oq = [(o[p][k], q[p][k]) for p in q for k in q[p] if k in o.get(p, {})]; oq = np.array(oq) >= 0.5
    os_ = np.array([(o[p][k], s[p][k]) for p in s for k in s[p] if k in o.get(p, {})]) >= 0.5
    return {"oracle_questions": sum(len(v) for v in o.values()), "oracle_keep": float(np.mean([v >= 0.5 for d in o.values() for v in d.values()])),
            "qwen7b_questions": int(len(oq)), "qwen7b_agree_with_oracle": float((oq[:, 0] == oq[:, 1]).mean()),
            "qwen7b_keep": float(oq[:, 1].mean()), "oracle_keep_same_questions": float(oq[:, 0].mean()),
            "omnigen2_self_questions": int(len(os_)), "omnigen2_self_agree_with_oracle": float((os_[:, 0] == os_[:, 1]).mean())}


def repair_questions():
    from run_repair import decomp_questions
    n = [len(v) for v in decomp_questions("bagel").values()]
    return {"prompts": len(n), "mean": float(np.mean(n)), "min": int(min(n)), "max": int(max(n))}


def repair_start():
    out = {}
    for n in ("bagel__bagel__ctx", "omnigen2__omnigen2__ctx_dqt", "janus7b__janus7b__ctx", "showo2_7b__showo2_7b__ctx"):
        s = _repair_status(n); out[n] = {"all_ok": float(np.mean([q[0][0] for q in s.values()])), "fact_acc": float(np.mean([q[0][1] for q in s.values()]))}
    return out


def judge_agreement_and_coverage(rows):
    A = collections.defaultdict(dict); o, _ = labels(rows, "human")
    for r in rows:
        c = cond_of(r)
        if c is None or r.get("answer") is None: continue
        A[(r["judge"], c)][(r["gen"], r["image"], r["fid"])] = r["answer"]
    agree = {}
    for c in sorted({c for (j, c) in A if j == "qwen25vl3b_gsys"}):
        a, b = A[("qwen25vl3b_gsys", c)], A.get(("omnigen2", c), {}); k = [x for x in a if x in b]
        if k: agree[c] = {"n": len(k), "agree": float(np.mean([a[x] == b[x] for x in k]))}
    extra = {}
    for J in ("bagel", "blip3o", "showo2_7b"):
        f = {k for k in A[(J, "fresh")] if k[0] == "showo2_7b"}; cl = {k for k in A[(J, "ctx")] if k[0] == "showo2_7b"}
        ex = cl - f; extra[J] = {"extra_closed_loop_facts": len(ex), "violated_human": sum(o.get(k) == "no" for k in ex)}
    return agree, extra


def exhaustive_timing():
    d = json.load(open(ROOT / "data/human/exhaustive_labels_persion4.json"))["exh"]; ms = [v["ms"] for v in d["timing"].values()]
    return {"facts_timed": len(ms), "total_min": sum(ms) / 60000, "median_s": statistics.median(ms) / 1000}


def dpo_training_facts():
    P = [json.loads(l) for l in open(ROOT / "data/train_prompts.jsonl", encoding="utf-8")]
    return {"prompts": len(P), "facts": sum(len(p["facts"]) for p in P), "by_tag": dict(collections.Counter(p["tag"] for p in P))}


def image_tokens():
    out = {}
    for J in ("janus7b", "showo2_7b"):
        out[J] = sorted({json.loads(l)["n_img"] for l in open(ROOT / f"runs/probe/attention_{J}__janus7b.jsonl")})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = load_rows(ROOT / "runs/judge"); agree, extra = judge_agreement_and_coverage(rows)
    res = {"annotator_agreement": annotator_agreement(), "entailment": entailment(), "repair_questions": repair_questions(),
           "repair_start": repair_start(), "qwen3b_gsys_vs_omnigen2": agree, "coverage_exclusions": extra,
           "exhaustive_timing": exhaustive_timing(), "dpo_training": dpo_training_facts(), "image_tokens": image_tokens()}
    print(json.dumps(res, indent=1))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
