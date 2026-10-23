import argparse, json, os, pathlib, time
from mucg.adapters.base import load

ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts.jsonl")); ap.add_argument("--limit", type=int)
    ap.add_argument("--tags", nargs="*"); ap.add_argument("--out", default=str(ROOT / "runs/gen"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    if a.tags: rows = [r for r in rows if r["tag"] in a.tags]
    if a.limit: rows = rows[: a.limit]
    out = pathlib.Path(a.out) / a.model; out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.jsonl"
    done = set()
    if man.exists():
        done = {json.loads(l)["image"] for l in open(man, encoding="utf-8")}
    gen = load(a.model)
    t_all = time.time(); n = 0
    with open(man, "a", encoding="utf-8") as mf:
        for r in rows:
            todo = [s for s in a.seeds if f"{r['pid']}_s{s}.png" not in done]
            if not todo: continue
            t = time.time()
            if hasattr(gen, "generate_batch"):
                imgs = gen.generate_batch(r["prompt"], todo)
            else:
                imgs = [gen.generate(r["prompt"], seed=s) for s in todo]
            sec = round((time.time() - t) / len(todo), 2)
            for s, img in zip(todo, imgs):
                name = f"{r['pid']}_s{s}.png"
                img.save(out / name)
                mf.write(json.dumps({"pid": r["pid"], "seed": s, "image": name, "prompt": r["prompt"],
                                     "model": a.model, "sec": sec, "batch": todo,
                                     "size": list(img.size)}) + "\n"); n += 1
            mf.flush()
    print(f"{a.model}: generated {n} images in {time.time()-t_all:.0f}s -> {out}")

if __name__ == "__main__":
    main()
