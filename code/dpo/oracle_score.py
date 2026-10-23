import argparse, glob, json, pathlib, sys, PIL.Image
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mucg.adapters.base import load

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1); a = ap.parse_args()
    d = ROOT / a.dir; prompts = {r["pid"]: r for r in map(json.loads, open(ROOT / a.prompts, encoding="utf-8"))}
    ims = sorted(pathlib.Path(p).name for p in glob.glob(str(d / "*.png")))[a.part::a.nparts]
    of = d / (f"oracle_p{a.part}.jsonl" if a.nparts > 1 else "oracle.jsonl")
    done = {(r["image"], r["fid"]) for r in map(json.loads, open(of))} if of.exists() else set()
    O = load("qwen3vl8b"); f = open(of, "a")
    for n in ims:
        pr = prompts.get(n.rsplit("_s", 1)[0])
        if pr is None: continue
        im = None
        for fa in pr["facts"]:
            if (n, fa["fid"]) in done: continue
            im = im or PIL.Image.open(d / n).convert("RGB"); j = O.judge(im, fa["question"])
            f.write(json.dumps({"image": n, "fid": fa["fid"], "answer": j["answer"], "p_yes": j["p_yes"]}) + "\n")
        f.flush()
    print("done", a.dir, a.part)

if __name__ == "__main__":
    main()
