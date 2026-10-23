import argparse, json, pathlib, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", required=True); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1); a = ap.parse_args()
    from mucg.adapters.omnigen2 import OmniGen2Adapter, SYS, IMG
    A = OmniGen2Adapter(); mllm, tok = A.pipe.mllm, A.tok
    print("attn:", mllm.config._attn_implementation)
    layers = mllm.model.layers if hasattr(mllm.model, "layers") else mllm.model.language_model.layers
    L = len(layers); quart = [set(range(q * L // 4, (q + 1) * L // 4)) for q in range(4)]
    state = {"mask": None, "layers": set()}
    def pre_hook(idx):
        def h(mod, args, kwargs):
            if state["mask"] is not None and idx in state["layers"]: kwargs["attention_mask"] = state["mask"]
            return args, kwargs
        return h
    for i, ly in enumerate(layers): ly.register_forward_pre_hook(pre_hook(i), with_kwargs=True)

    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    sub = json.load(open(a.subset))
    out = ROOT / "runs/mask"; out.mkdir(parents=True, exist_ok=True); of = out / ("mask_omnigen2.jsonl" if a.part == 0 else f"mask_omnigen2_p{a.part}.jsonl")
    done = set()
    if of.exists(): done = {(r["gen"], r["image"], r["fid"]) for r in (json.loads(l) for l in open(of))}
    f = open(of, "a")
    man = {g: {r["image"]: r for r in (json.loads(l) for l in open(ROOT / "runs/gen" / g / "manifest.jsonl"))} for g in sub}
    head = f"<|im_start|>system\n{SYS}<|im_end|>\n"
    for gen, items in sub.items():
        for image, fid, label in items[a.part::a.nparts]:
            if (gen, image, fid) in done: continue
            im = man[gen][image]; prior = im["prompt"]
            q = next(x["question"] for x in facts[im["pid"]]["facts"] if x["fid"] == fid)
            img = Image.open(ROOT / "runs/gen" / gen / image).convert("RGB")
            rows = []
            for ctx in ("fresh", "closed", "inline"):
                if ctx == "fresh": text = A.chat._apply_chat_template(q, [img])
                elif ctx == "closed":
                    text = (f"{head}<|im_start|>user\n{prior}<|im_end|>\n"
                            f"<|im_start|>assistant\n{IMG}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
                else: text = f"{head}<|im_start|>user\n{IMG}Caption: {prior}\n{q}<|im_end|>\n<|im_start|>assistant\n"
                inp = A.chat.prepare_inputs_for_text_generation(text, [img], mllm.device)
                ids = inp["input_ids"][0].tolist(); n = len(ids)
                img_idx = [i for i, t in enumerate(ids) if t == mllm.config.image_token_id]
                pieces = [tok.decode([t]) for t in ids]; s, starts = "", []
                for pc in pieces: starts.append(len(s)); s += pc
                def span(x):
                    pos = s.find(x)
                    if pos < 0: return []
                    end = pos + len(x)
                    return [i for i, st in enumerate(starts) if st < end and st + len(pieces[i]) > pos]
                cap = span(prior) if ctx != "fresh" else []
                if ctx != "fresh": assert cap, prior
                dt = torch.bfloat16; neg = torch.finfo(dt).min
                base = torch.full((n, n), neg, dtype=dt, device=mllm.device).triu(1)[None, None]
                conds = {"none": None}
                if ctx == "closed":
                    last_img, cap_end = max(img_idx), max(cap)
                    conds.update({"all": list(range(cap_end + 1, n)), "answer": [n - 1], "post_image": list(range(last_img + 1, n)),
                                  "image_only": list(range(cap_end + 1, last_img + 1))})
                elif ctx == "inline":
                    conds.update({"all": list(range(max(cap) + 1, n)), "answer": [n - 1]})
                for cname, qpos in conds.items():
                    lsets = [("all", set(range(L)))]
                    if cname == "all": lsets += [(f"Q{k+1}", quart[k]) for k in range(4)] + [(f"noQ{k+1}", set(range(L)) - quart[k]) for k in range(4)]
                    for lname, lset in lsets:
                        if qpos is not None:
                            mk = base.clone(); qi = torch.tensor(qpos, device=mllm.device); ki = torch.tensor(cap, device=mllm.device)
                            mk[0, 0, qi[:, None], ki[None, :]] = neg
                            state["mask"], state["layers"] = mk, lset
                        with torch.inference_mode(): o = mllm(**inp)
                        state["mask"] = None
                        pr = torch.softmax(o.logits[0, -1].float(), -1)
                        py, pn = pr[A.yes_ids].sum().item(), pr[A.no_ids].sum().item()
                        rows.append({"judge": "omnigen2", "gen": gen, "image": image, "pid": im["pid"], "fid": fid, "label": label,
                                     "ctx": ctx, "mask": cname, "layers": lname, "p_yes": py / (py + pn + 1e-12), "mass": py + pn})
                        if ctx == "fresh": break
            for r in rows: f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", of)

if __name__ == "__main__":
    main()
