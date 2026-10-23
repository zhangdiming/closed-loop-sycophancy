import argparse, json, pathlib, random, re, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent
COT = ("Think step by step about the parts of the image that matter for this question, "
       "then end with \"Answer: yes\" or \"Answer: no\".")
OWN = {"bagel": "bagel", "janus7b": "janus7b", "omnigen2": "omnigen2"}


def items_for(judge):
    sub = json.load(open(ROOT / f"runs/mask/subset_{judge}.json"))
    rng = random.Random(0); out = []
    for gen, its in sub.items():
        if gen not in (OWN[judge], "janus7b", "showo2_7b"): continue
        viol = [x for x in its if x[2] == "violated"]; sat = [x for x in its if x[2] == "satisfied"]
        rng.shuffle(sat); out += [(gen, *x) for x in sat[:100]]
        if gen == OWN[judge]: out += [(gen, *x) for x in viol]
        else: out += [(gen, *x) for x in viol]
    own = [x for x in out if x[0] == OWN[judge] or x[3] == "satisfied"]
    oth = [x for x in out if x[0] != OWN[judge] and x[3] == "violated"]
    rng.shuffle(oth)
    return sorted(own + oth[:200])


def parse(text):
    t = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*\"?(yes|no)\b", t, re.I)
    if m: return m[-1].lower()
    m = re.findall(r"\b(yes|no)\b", t, re.I)
    if "</think>" in text and m: return m[0].lower()
    return m[-1].lower() if m else None


class Gen:
    def __init__(self, judge):
        from mucg.adapters.base import load
        self.j = judge; self.A = load(judge)

    @torch.no_grad()
    def __call__(self, img, question, prior, n):
        A, j = self.A, self.j
        if j == "bagel":
            from data.data_utils import pil_img2rgb
            import importlib, sys
            inf = A.inf; VLM_THINK = sys.modules[type(inf).__module__].VLM_THINK_SYSTEM_PROMPT
            ctx = inf.init_gen_context()
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                ctx = inf.update_context_text(VLM_THINK, ctx)
                if prior is not None: ctx = inf.update_context_text(prior, ctx)
                ctx = inf.update_context_image(inf.vae_transform.resize_transform(pil_img2rgb(img)), ctx, vae=False)
                ctx = inf.update_context_text(question, ctx)
                return inf.gen_text(ctx, max_length=n, do_sample=False)
        q = re.sub(r"\s*Answer yes or no\.?\s*$", "", question).strip() + " " + COT
        if j == "janus7b":
            if prior is None:
                conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
            else:
                conv = [{"role": "<|User|>", "content": prior}, {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [img]},
                        {"role": "<|User|>", "content": q}, {"role": "<|Assistant|>", "content": ""}]
            inp = A.proc(conversations=conv, images=[img], force_batchify=True).to(A.device)
            emb = A.model.prepare_inputs_embeds(**inp)
            out = A.model.language_model.generate(inputs_embeds=emb, attention_mask=inp.attention_mask, pad_token_id=A.tok.eos_token_id,
                                                  bos_token_id=A.tok.bos_token_id, eos_token_id=A.tok.eos_token_id,
                                                  max_new_tokens=n, do_sample=False, use_cache=True)
            return A.tok.decode(out[0], skip_special_tokens=True)
        if j == "omnigen2":
            from mucg.adapters.omnigen2 import SYS, IMG
            if prior is None: text = A.chat._apply_chat_template(q, [img])
            else:
                text = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{prior}<|im_end|>\n"
                        f"<|im_start|>assistant\n{IMG}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
            inp = A.chat.prepare_inputs_for_text_generation(text, [img], A.pipe.mllm.device)
            out = A.pipe.mllm.generate(**inp, max_new_tokens=n, do_sample=False)
            return A.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        raise ValueError(j)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True, choices=list(OWN))
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    items = items_for(a.judge)[a.part::a.nparts]
    if a.limit: items = items[: a.limit]
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    man = {g: {r["image"]: r for r in (json.loads(l) for l in open(ROOT / "runs/gen" / g / "manifest.jsonl"))} for g in {x[0] for x in items}}
    out = ROOT / "runs/reason"; out.mkdir(parents=True, exist_ok=True)
    of = out / (f"{a.judge}.jsonl" if a.part == 0 else f"{a.judge}_p{a.part}.jsonl")
    done = {(r["gen"], r["image"], r["fid"], r["ctx"]) for r in (json.loads(l) for l in open(of))} if of.exists() else set()
    f = open(of, "a"); G = Gen(a.judge)
    for gen, image, fid, label in items:
        im = man[gen][image]; prior = im["prompt"]
        q = next(x["question"] for x in facts[im["pid"]]["facts"] if x["fid"] == fid)
        img = None
        for ctx in ("fresh", "closed"):
            if (gen, image, fid, ctx) in done: continue
            if img is None: img = Image.open(ROOT / "runs/gen" / gen / image).convert("RGB")
            t = G(img, q, prior if ctx == "closed" else None, a.max_new)
            f.write(json.dumps({"judge": a.judge, "gen": gen, "image": image, "pid": im["pid"], "fid": fid, "label": label,
                                "ctx": ctx, "text": t, "verdict": parse(t)}) + "\n"); f.flush()
    print("done", of)


if __name__ == "__main__":
    main()
