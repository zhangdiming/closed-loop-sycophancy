import argparse, json, pathlib, time
from run_repair import feedback_prompt
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "plan"]); ap.add_argument("--name", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--judge"); ap.add_argument("--round", type=int, required=True); ap.add_argument("--context", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tags", nargs="*", default=["position", "color_attr", "counting"])
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    facts = {r["pid"]: r for r in rows}
    out = ROOT / "runs/repair" / a.name; out.mkdir(parents=True, exist_ok=True)
    sp = out / "state.json"
    state = json.load(open(sp)) if sp.exists() else {"active": {r["pid"]: r["prompt"] for r in rows if r["tag"] in a.tags}, "round": 0, "history": {}}
    k = a.round
    if a.cmd == "gen":
        from mucg.adapters.base import load
        rd = out / f"round{k}"; rd.mkdir(exist_ok=True)
        man = rd / "manifest.jsonl"
        done = {json.loads(l)["pid"] for l in open(man, encoding="utf-8")} if man.exists() else set()
        gen = load(a.gen)
        with open(man, "a", encoding="utf-8") as mf:
            for pid, prompt in state["active"].items():
                if pid in done: continue
                t = time.time(); img = gen.generate(prompt, seed=a.seed + 100 * k)
                name = f"{pid}_s{a.seed}.png"; img.save(rd / name)
                mf.write(json.dumps({"pid": pid, "seed": a.seed, "image": name, "prompt": facts[pid]["prompt"], "used_prompt": prompt,
                                     "model": a.gen, "round": k, "sec": round(time.time() - t, 1)}) + "\n"); mf.flush()
        print(f"round{k}: generated for {len(state['active'])} active prompts")
    else:
        jf = out / "judge" / f"{a.judge}__round{k}{'__ctx' if a.context else ''}.jsonl"
        verdict = {}
        for r in (json.loads(l) for l in open(jf, encoding="utf-8")): verdict.setdefault(r["pid"], {})[r["fid"]] = r["answer"]
        nxt = {}
        for pid, prompt in state["active"].items():
            v = verdict.get(pid, {})
            failed = [f for f in facts[pid]["facts"] if v.get(f["fid"]) == "no"]
            state["history"].setdefault(pid, []).append({"round": k, "prompt": prompt, "n_failed": len(failed)})
            if failed: nxt[pid] = feedback_prompt(facts[pid]["prompt"], failed)
        state["active"] = nxt; state["round"] = k + 1
        json.dump(state, open(sp, "w"), indent=1)
        print(f"round{k}: {len(verdict)} judged, {len(nxt)} prompts still failing -> round{k+1}")

if __name__ == "__main__":
    main()
