import argparse, json, pathlib, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="janus7b"); ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--modes", nargs="*", default=["fresh", "assistant", "assistant_nocaption", "caption_inline", "user_prompt"])
    ap.add_argument("--only_violated", action="store_true", help="restrict to facts the oracle marked violated")
    ap.add_argument("--judge", default="janus7b", choices=["janus7b", "janus1b"])
    a = ap.parse_args()
    from mucg.adapters.janus import JanusAdapter
    J = JanusAdapter(repo={"janus7b": "deepseek-ai/Janus-Pro-7B", "janus1b": "deepseek-ai/Janus-Pro-1B"}[a.judge])
    m, proc, tok = J.model, J.proc, J.tok
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    gdir = ROOT / "runs/gen" / a.gen
    images = [json.loads(l) for l in open(gdir / "manifest.jsonl", encoding="utf-8")]
    viol = set()
    of = ROOT / "runs/judge" / f"qwen3vl8b__{a.gen}.jsonl"
    if a.only_violated and of.exists():
        viol = {(r["image"], r["fid"]) for r in (json.loads(l) for l in open(of, encoding="utf-8")) if r["answer"] == "no"}
    out = ROOT / "runs/probe"; out.mkdir(exist_ok=True)
    f = open(out / f"attention_{a.judge}__{a.gen}.jsonl", "w", encoding="utf-8")
    n = 0
    img_tag_id = tok.encode(proc.image_tag, add_special_tokens=False)[-1] if hasattr(proc, "image_tag") else None
    for im in images:
        for fact in facts[im["pid"]]["facts"]:
            if a.only_violated and (im["image"], fact["fid"]) not in viol: continue
            if n >= a.limit: break
            img = Image.open(gdir / im["image"]).convert("RGB")
            for mode in a.modes:
                prior = None if mode == "fresh" else im["prompt"]
                mm = "assistant" if mode == "fresh" else mode
                q = fact["question"]
                if prior is None:
                    conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
                elif mm == "assistant":
                    conv = [{"role": "<|User|>", "content": prior}, {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [img]},
                            {"role": "<|User|>", "content": q}, {"role": "<|Assistant|>", "content": ""}]
                elif mm == "user_prompt":
                    conv = [{"role": "<|User|>", "content": f"{prior}\n<image_placeholder>\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
                elif mm == "assistant_nocaption":
                    conv = [{"role": "<|User|>", "content": "Generate an image."}, {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [img]},
                            {"role": "<|User|>", "content": q}, {"role": "<|Assistant|>", "content": ""}]
                else:
                    conv = [{"role": "<|User|>", "content": f"<image_placeholder>\nCaption: {prior}\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
                inp = proc(conversations=conv, images=[img], force_batchify=True).to(J.device)
                ids = inp.input_ids[0].tolist()
                is_img = inp.images_seq_mask[0].bool().tolist() if hasattr(inp, "images_seq_mask") else [False] * len(ids)
                pieces = [tok.decode([t]) for t in ids]
                text, starts = "", []
                for pc in pieces: starts.append(len(text)); text += pc
                def span(sub):
                    if not sub: return set()
                    pos = text.find(sub)
                    if pos < 0: pos = text.find(sub.strip())
                    if pos < 0: return set()
                    end = pos + len(sub)
                    return {i for i, st in enumerate(starts) if st < end and st + len(pieces[i]) > pos}
                cap = span(prior) if prior is not None else set()
                qs = span(q)
                with torch.inference_mode():
                    emb = m.prepare_inputs_embeds(**inp)
                    o = m.language_model(inputs_embeds=emb, attention_mask=inp.attention_mask, use_cache=False, output_attentions=True)
                    per_layer = torch.stack([x[0, :, -1, :].float().mean(0) for x in o.attentions])
                    att = per_layer.mean(0)
                    probs = torch.softmax(o.logits[0, -1].float(), -1)
                p_yes = probs[J.yes_ids].sum().item(); p_no = probs[J.no_ids].sum().item()
                mass = lambda idx: float(att[list(idx)].sum()) if idx else 0.0
                img_idx = [i for i, b in enumerate(is_img) if b]
                layer_img = [float(per_layer[l, img_idx].sum()) for l in range(per_layer.shape[0])] if img_idx else []
                layer_cap = [float(per_layer[l, list(cap)].sum()) for l in range(per_layer.shape[0])] if cap else []
                f.write(json.dumps({"image": im["image"], "pid": im["pid"], "fid": fact["fid"], "type": fact["type"], "mode": mode,
                                    "att_image": mass([i for i, b in enumerate(is_img) if b]), "att_caption": mass(cap), "att_question": mass(qs),
                                    "n_img": sum(is_img), "n_cap": len(cap), "p_yes": p_yes / (p_yes + p_no + 1e-12),
                                    "layer_img": layer_img, "layer_cap": layer_cap}) + "\n")
            n += 1; f.flush()
    print("wrote", n, "image-facts")

if __name__ == "__main__":
    main()
