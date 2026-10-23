import argparse, json, pathlib, time
from mucg.adapters.base import load
from run_judge import perturb

ROOT = pathlib.Path(__file__).resolve().parent

def feedback_prompt(prompt, failed):
    notes = []
    for f in failed:
        a = f["args"]; t = f["type"]
        if t == "presence": notes.append(f"the image must clearly contain {a['class']}")
        elif t == "count": notes.append(f"there must be exactly {a['count']} {a['class']}(s), no more and no fewer")
        elif t == "color": notes.append(f"the {a['class']} must be {a['color']}")
        elif t == "position": notes.append(f"the {a['class']} must be {a['relation']} the {a['other']}")
    return prompt + ". " + "; ".join(notes) + "."

def decomp_questions(gen, entail="runs/pub/entail_scores.json"):
    dec = {}
    for fp in sorted((ROOT / "runs/pub").glob(f"decomp_{gen}_p[0-9].json")): dec.update(json.load(open(fp)))
    ent = json.load(open(ROOT / entail))
    return {pid: [{"fid": f"dq{i}", "question": f"{q} Please answer yes or no.", "q": q}
                  for i, q in enumerate(d["questions"]) if ent.get(pid, {}).get(q, 0) >= 0.5] for pid, d in dec.items()}

def feedback_prompt_q(prompt, failed):
    return prompt + ". Make sure the answer to each of these questions is yes: " + " ".join(f["q"] for f in failed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--judge", required=True)
    ap.add_argument("--context", action="store_true"); ap.add_argument("--perturb", default=None)
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tags", nargs="*"); ap.add_argument("--limit", type=int)
    ap.add_argument("--judge_prompt", default="current", choices=["current", "original"], help="closed-loop judge sees the current (feedback) prompt or always the original request")
    ap.add_argument("--no_threshold", type=float, default=None, help="a fact fails when p(no) >= this threshold instead of the decoded answer")
    ap.add_argument("--reuse_round0", default=None, help="run directory whose round-0 images (same seeds) are reused instead of regenerated")
    ap.add_argument("--questions", default="facts", choices=["facts", "decomp"], help="GenEval metadata facts, or the generator's own entailment-filtered decomposition")
    ap.add_argument("--entail", default="runs/pub/entail_scores.json", help="entailment scores for --questions decomp (oracle, or entail_text.py)")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "N"), help="process prompts I::N (parallel runs append to the same trace/manifests)")
    ap.add_argument("--protocol", default=None, choices=["unigen", "unicorn"], help="released holistic verifier instead of per-fact questions")
    ap.add_argument("--ug_threshold", type=float, default=0.5, help="UniGen rejects when p(yes) is below this (0.5 = its decoded verdict; review C158 matches it to ERASE's rejection rate)")
    ap.add_argument("--no_feedback", action="store_true", help="regenerate a rejected image from the original request with a new seed, as the holistic verifiers must (review C145)")
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts.jsonl")); ap.add_argument("--out", default=str(ROOT / "runs/repair"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    if a.tags: rows = [r for r in rows if r["tag"] in a.tags]
    if a.limit: rows = rows[: a.limit]
    if a.shard: rows = rows[a.shard[0]::a.shard[1]]
    name = f"{a.gen}__{a.judge}{'__ctx' if a.context else ''}{'__' + a.perturb if a.perturb else ''}{a.tag}"
    out = pathlib.Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
    for k in range(a.rounds + 1): (out / f"round{k}").mkdir(exist_ok=True)
    trace = out / "trace.jsonl"
    done = {(json.loads(l)["pid"], json.loads(l)["seed"]) for l in open(trace, encoding="utf-8")} if trace.exists() else set()
    dq = decomp_questions(a.gen, a.entail) if a.questions == "decomp" else None
    gen = load(a.gen); judge = None if a.judge == "none" else gen if a.judge == a.gen else load(a.judge)
    if a.protocol:
        import pub_protocols as pp
        B = pp.Backend.__new__(pp.Backend); B.name, B.A = a.judge, judge
    with open(trace, "a", encoding="utf-8") as tf:
        for r in rows:
            for s in a.seeds:
                if (r["pid"], s) in done: continue
                prompt, rec = r["prompt"], {"pid": r["pid"], "seed": s, "rounds": []}
                for k in range(a.rounds + 1):
                    t = time.time()
                    reuse = pathlib.Path(a.reuse_round0) / "round0" / f"{r['pid']}_s{s}.png" if (a.reuse_round0 and k == 0) else None
                    if reuse is not None and reuse.exists():
                        from PIL import Image as _I; img = _I.open(reuse).convert("RGB")
                    else:
                        img = gen.generate(prompt, seed=s + 100 * k)
                    name_k = f"{r['pid']}_s{s}.png"; img.save(out / f"round{k}" / name_k)
                    verdicts = []; qs = dq.get(r["pid"], []) if dq is not None else r["facts"]
                    if a.protocol:
                        if a.protocol == "unigen":
                            py = B.yn(img, pp.UG_OUT.format(prompt=r["prompt"])); ok = py is not None and py >= a.ug_threshold
                            verdicts = [{"fid": "unigen", "p_yes": py, "answer": "yes" if ok else "no"}]
                        else:
                            t_ = B.gen(img, pp.PROMPT_GALLERY[pp.UC_CAT[r["tag"]]].format(prompt=r["prompt"], category=r["tag"]), 400)
                            sc = pp.parse_json_field(t_, "score")
                            try: sc = float(sc)
                            except (TypeError, ValueError): sc = None
                            ok = sc is None or sc > 6
                            verdicts = [{"fid": "unicorn", "score": sc, "raw": t_[:1500], "answer": "yes" if ok else "no"}]
                        rec["rounds"].append({"k": k, "prompt": prompt, "image": f"round{k}/{r['pid']}_s{s}.png", "verdicts": verdicts,
                                              "n_failed": 0 if ok else 1, "sec": round(time.time() - t, 1)})
                        if ok: break
                        continue
                    for f in (qs if judge is not None else []):
                        jp = (r["prompt"] if a.judge_prompt == "original" else prompt) if a.context else None
                        v = judge.judge(perturb(img, a.perturb), f["question"], prior_prompt=jp)
                        if a.no_threshold is not None and v.get("p_no") is not None: v["answer"] = "no" if v["p_no"] >= a.no_threshold else "yes"
                        verdicts.append({"fid": f["fid"], "answer": v["answer"], "p_no": v.get("p_no")})
                    failed = [f for f, v in zip(qs, verdicts) if v["answer"] == "no"] if judge is not None else list(qs)
                    rec["rounds"].append({"k": k, "prompt": prompt, "image": f"round{k}/{name_k}", "verdicts": verdicts,
                                          "n_failed": len(failed), "sec": round(time.time() - t, 1)})
                    if not failed: break
                    prompt = r["prompt"] if a.no_feedback else (feedback_prompt_q(r["prompt"], failed) if dq is not None else feedback_prompt(r["prompt"], failed))
                for rd in rec["rounds"]:
                    with open(out / rd["image"].split("/")[0] / "manifest.jsonl", "a", encoding="utf-8") as mf:
                        mf.write(json.dumps({"pid": r["pid"], "seed": s, "image": rd["image"].split("/")[1], "prompt": r["prompt"],
                                             "model": a.gen, "round": rd["k"]}) + "\n")
                tf.write(json.dumps(rec) + "\n"); tf.flush()
    print(f"done -> {out}")

if __name__ == "__main__":
    main()
