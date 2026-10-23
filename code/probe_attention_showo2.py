import argparse, json, pathlib, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="showo2_7b"); ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--modes", nargs="*", default=["fresh", "assistant_nocaption", "assistant", "user_prompt", "caption_inline"])
    ap.add_argument("--only_violated", action="store_true"); ap.add_argument("--size", default="7B", choices=["7B", "1.5B"])
    a = ap.parse_args()
    from mucg.adapters.showo2 import ShowO2Adapter, omni_attn_mask_naive
    J = ShowO2Adapter(size=a.size); m, tok, dev = J.model, J.tok, J.device
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    gdir = ROOT / "runs/gen" / a.gen
    images = [json.loads(l) for l in open(gdir / "manifest.jsonl", encoding="utf-8")]
    viol = set(); of = ROOT / "runs/judge" / f"qwen3vl8b__{a.gen}.jsonl"
    if a.only_violated and of.exists():
        viol = {(r["image"], r["fid"]) for r in (json.loads(l) for l in open(of, encoding="utf-8")) if r["answer"] == "no"}
    out = ROOT / "runs/probe"; out.mkdir(exist_ok=True)
    f = open(out / f"attention_showo2_{a.size.lower()}__{a.gen}.jsonl", "w", encoding="utf-8"); n = 0
    from datasets.utils import image_transform
    for im in images:
        for fact in facts[im["pid"]]["facts"]:
            if a.only_violated and (im["image"], fact["fid"]) not in viol: continue
            if n >= a.limit: break
            img = Image.open(gdir / im["image"]).convert("RGB"); q = fact["question"]
            for mode in a.modes:
                prior = None if mode == "fresh" else im["prompt"]; mm = "assistant" if mode == "fresh" else mode
                with torch.no_grad():
                    x = image_transform(img, resolution=J.res).to(dev).unsqueeze(0)
                    lat = J.vae.sample(x.to(J.dt).unsqueeze(2)).squeeze(2).to(J.dt)
                    und = m.image_embedder_und(lat); gen = m.image_embedder_gen(lat)
                    und = m.und_trans(und + m.position_embedding(m.image_position_ids))["last_hidden_state"]
                    img_emb = m.fusion_proj(torch.cat([und, gen], -1))
                    sys_ids = tok("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False)["input_ids"]
                    role_a = tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
                    role_b = tok("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
                    q_ids = tok(q, add_special_tokens=False).input_ids; cap_ids = []
                    if prior is not None and mm in ("assistant", "assistant_nocaption"):
                        cap_ids = tok(prior if mm == "assistant" else "Generate an image.", add_special_tokens=False).input_ids
                        role_a = role_a + cap_ids + tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
                        role_b = tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"] + q_ids + role_b; q_tail = []
                        cap_pos = (len(sys_ids) + 1 + len(tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]), len(cap_ids))
                    elif prior is not None and mm == "user_prompt":
                        cap_ids = tok(prior + "\n", add_special_tokens=False)["input_ids"]; role_a = role_a + cap_ids; q_tail = q_ids
                        cap_pos = (len(sys_ids) + 1 + len(tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]), len(cap_ids))
                    elif prior is not None and mm == "caption_inline":
                        cap_ids = tok(f"Caption: {prior}\n", add_special_tokens=False)["input_ids"]; q_tail = cap_ids + q_ids; cap_pos = None
                    else:
                        q_tail = q_ids; cap_pos = None
                    ta = torch.tensor([J.sid["bos_id"]] + sys_ids + role_a, device=dev)[None]
                    tb = torch.tensor([J.sid["boi_id"], J.sid["eoi_id"]] + q_tail + role_b, device=dev)[None]
                    ea, eb = m.showo.model.embed_tokens(ta), m.showo.model.embed_tokens(tb)
                    te = m.time_embed(torch.Tensor([[1.0]]).to(dev), ea.dtype)
                    if hasattr(m, "time_embed_proj"): te = m.time_embed_proj(te)
                    emb = torch.cat([ea, eb[:, :1], te, img_emb, eb[:, 1:]], 1).to(J.dt)
                    mp = torch.tensor([ta.shape[1] + 2, J.n_mmu])[None, None, :].to(dev)
                    mask = omni_attn_mask_naive(B=1, LEN=emb.size(1), modalities=mp, device=dev, inverted=True).to(emb.dtype)
                    o = m.showo(inputs_embeds=emb, attention_mask=mask, output_attentions=True)
                    per_layer = torch.stack([x[0, :, -1, :].float().mean(0) for x in o.attentions]); att = per_layer.mean(0)
                    probs = torch.softmax(o.logits[0, -1].float(), -1)
                p_yes = probs[J.yes_ids].sum().item(); p_no = probs[J.no_ids].sum().item()
                n_a = ta.shape[1]; img_idx = list(range(n_a + 2, n_a + 2 + J.n_mmu))
                if cap_pos is not None: cap_idx = list(range(cap_pos[0], cap_pos[0] + cap_pos[1]))
                elif mm == "caption_inline" and prior is not None: cap_idx = list(range(n_a + 2 + J.n_mmu + 1, n_a + 2 + J.n_mmu + 1 + len(cap_ids)))
                else: cap_idx = []
                q_start = emb.size(1) - len(role_b) - len(q_ids); q_idx = list(range(q_start, q_start + len(q_ids)))
                ms = lambda idx: float(att[idx].sum()) if idx else 0.0
                f.write(json.dumps({"image": im["image"], "pid": im["pid"], "fid": fact["fid"], "type": fact["type"], "mode": mode,
                                    "att_image": ms(img_idx), "att_caption": ms(cap_idx), "att_question": ms(q_idx), "n_img": len(img_idx), "n_cap": len(cap_idx),
                                    "p_yes": p_yes / (p_yes + p_no + 1e-12),
                                    "layer_img": [float(per_layer[l, img_idx].sum()) for l in range(per_layer.shape[0])],
                                    "layer_cap": [float(per_layer[l, cap_idx].sum()) for l in range(per_layer.shape[0])] if cap_idx else []}) + "\n")
            n += 1; f.flush()
    print("wrote", n, "image-facts")

if __name__ == "__main__":
    main()
