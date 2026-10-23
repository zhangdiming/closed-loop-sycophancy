import argparse, collections, glob, json, os, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="qwen3vl8b"); ap.add_argument("--runs", nargs="*")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--judge", action="store_true", help="run the oracle on all rounds first (needs a GPU)")
    a = ap.parse_args()
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    runs = a.runs or [p.name for p in sorted((ROOT / "runs/repair").glob("*")) if (p / "trace.jsonl").exists()]
    for name in runs:
        d = ROOT / "runs/repair" / name
        rounds = sorted(p.name for p in d.glob("round*") if (p / "manifest.jsonl").exists())
        if a.judge:
            for rd in rounds:
                subprocess.run([a.python, str(ROOT / "run_judge.py"), "--judge", a.oracle, "--gen", rd, "--gen_dir", str(d),
                                "--out", str(d / "oracle")], check=False)
        if (d / "trace.jsonl").exists():
            trace = [json.loads(l) for l in open(d / "trace.jsonl", encoding="utf-8")]
        else:
            jd = d / "judge"; per = collections.defaultdict(list)
            for rd in rounds:
                jf = sorted(jd.glob(f"*__{rd}*.jsonl"))
                verdict = collections.defaultdict(dict)
                for f in jf:
                    for r in (json.loads(l) for l in open(f, encoding="utf-8")): verdict[r["pid"]][r["fid"]] = r["answer"]
                for m in (json.loads(l) for l in open(d / rd / "manifest.jsonl", encoding="utf-8")):
                    v = verdict.get(m["pid"], {})
                    per[m["pid"]].append({"k": int(rd[5:]), "image": f"{rd}/{m['image']}", "n_failed": sum(x == "no" for x in v.values()),
                                          "judged": len(v) > 0})
            trace = [{"pid": pid, "seed": 0, "rounds": sorted(rs, key=lambda x: x["k"])} for pid, rs in per.items()]
            for t in trace:
                for r in t["rounds"]:
                    if not r.get("judged", True): r["n_failed"] = 1
        print(f"\n=== {name}: {len(trace)} prompts, rounds {rounds}")
        ora = {}
        for rd in rounds:
            f = d / "oracle" / f"{a.oracle}__{rd}.jsonl"
            if f.exists():
                for r in (json.loads(l) for l in open(f, encoding="utf-8")): ora[(rd, r["image"], r["fid"])] = r["answer"]
        print(f"{'round':8s} {'n_active':>8s} {'verifier_ok':>11s} {'oracle_ok(all)':>14s} {'fact_acc':>8s}  (oracle view of the image the loop holds after this round)")
        for k, rd in enumerate(rounds):
            active = [t for t in trace if len(t["rounds"]) > k]
            vok = [t["rounds"][k]["n_failed"] == 0 for t in active]
            okall, accs = [], []
            for t in trace:
                kk = min(k, len(t["rounds"]) - 1); rdk = f"round{kk}"; im = t["rounds"][kk]["image"].split("/")[1]
                labels = [ora.get((rdk, im, fct["fid"])) for fct in facts[t["pid"]]["facts"]]
                if any(l is None for l in labels): continue
                okall.append(all(l == "yes" for l in labels)); accs.append(sum(l == "yes" for l in labels) / len(labels))
            print(f"{rd:8s} {len(active):8d} {sum(vok)/max(1,len(vok)):11.3f} {sum(okall)/max(1,len(okall)):14.3f} {sum(accs)/max(1,len(accs)):8.3f}   (n_oracle={len(okall)})")

if __name__ == "__main__":
    main()
