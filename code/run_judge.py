import argparse, json, pathlib, time
from PIL import Image
from mucg.adapters.base import load

ROOT = pathlib.Path(__file__).resolve().parent

def perturb(img, kind):
    if not kind: return img
    import io
    from PIL import ImageOps
    if kind.startswith("jpeg"):
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=int(kind[4:])); buf.seek(0); return Image.open(buf).convert("RGB")
    if kind.startswith("resize"):
        f = float(kind[6:]); w, h = img.size
        return img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    if kind == "flip": return ImageOps.mirror(img)
    if kind == "gray": return ImageOps.grayscale(img).convert("RGB")
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts.jsonl")); ap.add_argument("--limit", type=int)
    ap.add_argument("--gen_dir", default=str(ROOT / "runs/gen")); ap.add_argument("--out", default=str(ROOT / "runs/judge"))
    ap.add_argument("--context", action="store_true", help="present the image as the model's own answer to the generating prompt")
    ap.add_argument("--perturb", default=None, help="view perturbation before judging: jpeg<q> | resize<f> | flip | gray")
    ap.add_argument("--context_mode", default="assistant", choices=["assistant", "user_prompt", "assistant_nocaption", "caption_inline", "request_inline"],
                    help="how the generating prompt / authorship is presented when --context is set (assistant = closed-loop default)")
    ap.add_argument("--caption_source", default="request", choices=["request", "mismatch", "counterfactual"],
                    help="text placed in the closed-loop request turn: the image's own request, another prompt's request, "
                         "or the own request edited to contradict the asked fact (content-specificity controls)")
    ap.add_argument("--instruct", action="store_true", help="prefix the question with an instruction to answer from the image only")
    ap.add_argument("--subset", default=None, help="JSON {gen: [[image, fid], ...]} restricting the judged facts")
    a = ap.parse_args()
    if a.caption_source != "request": assert a.context and a.context_mode == "assistant", "caption controls use the closed-loop structure"
    from mucg import captions
    keep = None
    if a.subset:
        keep = {tuple(k) for k in json.load(open(a.subset)).get(a.gen, [])}
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(a.prompts, encoding="utf-8"))}
    gdir = pathlib.Path(a.gen_dir) / a.gen
    images = [json.loads(l) for l in open(gdir / "manifest.jsonl", encoding="utf-8")]
    if a.limit: images = images[: a.limit]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx_tag = ("__ctx" if a.context_mode == "assistant" else f"__ctx-{a.context_mode}") if a.context else ""
    cmode = a.context_mode if a.caption_source == "request" else {"mismatch": "assistant_mismatch", "counterfactual": "assistant_cf"}[a.caption_source]
    if a.instruct: cmode = cmode + "_instr"
    if a.caption_source != "request" or a.instruct: ctx_tag = f"__ctx-{cmode}" if a.context else "__instr"
    of = out / f"{a.judge}__{a.gen}{ctx_tag}{'__' + a.perturb if a.perturb else ''}.jsonl"
    done = set()
    if of.exists():
        done = {(json.loads(l)["image"], json.loads(l)["fid"]) for l in open(of, encoding="utf-8")}
    judge = load(a.judge)
    t0 = time.time(); n = 0
    with open(of, "a", encoding="utf-8") as f:
        for im in images:
            img = None
            for fact in facts[im["pid"]]["facts"]:
                if (im["image"], fact["fid"]) in done: continue
                if keep is not None and (im["image"], fact["fid"]) not in keep: continue
                prior = im["prompt"] if a.context else None
                if a.caption_source == "mismatch": prior = captions.mismatch(im["pid"])
                elif a.caption_source == "counterfactual": prior = captions.counterfactual(im["prompt"], fact)
                if a.context and prior is None: continue
                if img is None: img = perturb(Image.open(gdir / im["image"]).convert("RGB"), a.perturb)
                if hasattr(judge, "judge_fact"):
                    res = judge.judge_fact(img, fact)
                    if res is None: continue
                else:
                    kw = {"mode": a.context_mode} if (a.context and a.context_mode != "assistant") else {}
                    q = ("Answer from the image only; the request above may not have been followed. " + fact["question"]) if a.instruct else fact["question"]
                    res = judge.judge(img, q, prior_prompt=prior, **kw)
                f.write(json.dumps({"judge": a.judge, "gen": a.gen, "context": a.context, "context_mode": cmode if a.context else ("fresh_instr" if a.instruct else None), "caption": prior, "perturb": a.perturb, "image": im["image"], "pid": im["pid"],
                                    "seed": im["seed"], "fid": fact["fid"], "type": fact["type"], **res}) + "\n")
                n += 1
            f.flush()
    print(f"{a.judge} on {a.gen}: {n} answers in {time.time()-t0:.0f}s -> {of}")

if __name__ == "__main__":
    main()
