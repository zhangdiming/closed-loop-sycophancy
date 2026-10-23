import argparse, json, pathlib, sys, numpy as np, torch, PIL.Image
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mucg.adapters.base import load

@torch.inference_mode()
def gen_tokens(A, prompt, seeds):
    torch.manual_seed(seeds[0]); torch.cuda.manual_seed_all(seeds[0])
    n = len(seeds); m, proc = A.model, A.proc
    conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=proc.sft_format, system_prompt="")
    ids = torch.LongTensor(A.tok.encode(sft + proc.image_start_tag))
    tokens = torch.zeros((n * 2, len(ids)), dtype=torch.int, device=A.device)
    for i in range(n * 2):
        tokens[i] = ids
        if i % 2: tokens[i, 1:-1] = proc.pad_id
    emb = m.language_model.get_input_embeddings()(tokens)
    ntok = (A.img_size // 16) ** 2; gen = torch.zeros((n, ntok), dtype=torch.int, device=A.device); out = None
    for i in range(ntok):
        out = m.language_model.model(inputs_embeds=emb, use_cache=True, past_key_values=out.past_key_values if i else None)
        logits = m.gen_head(out.last_hidden_state[:, -1, :])
        logits = logits[1::2] + A.cfg * (logits[0::2] - logits[1::2])
        nxt = torch.multinomial(torch.softmax(logits / A.temp, -1), 1); gen[:, i] = nxt.squeeze(-1)
        emb = m.prepare_gen_img_embeds(torch.cat([nxt, nxt], 1).view(-1)).unsqueeze(1)
    dec = m.gen_vision_model.decode_code(gen, shape=[n, 8, A.img_size // 16, A.img_size // 16])
    dec = np.clip((dec.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return [PIL.Image.fromarray(d) for d in dec], gen.cpu().numpy().astype(np.int32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--judge", action="store_true"); ap.add_argument("--ckpt")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(ROOT / a.prompts, encoding="utf-8")][a.part::a.nparts]
    A = load("janus1b")
    if a.ckpt:
        sd = torch.load(ROOT / a.ckpt, map_location="cpu"); A.model.language_model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})
    jf = out / f"judge_p{a.part}.jsonl"
    done = {(r["image"], r["fid"], r["ctx"]) for r in map(json.loads, open(jf))} if (a.judge and jf.exists()) else set()
    f = open(jf, "a") if a.judge else None
    for r in rows:
        names = [f"{r['pid']}_s{s}.png" for s in range(a.seeds)]
        if not all((out / n).exists() for n in names):
            ims, toks = gen_tokens(A, r["prompt"], list(range(a.seeds)))
            for n, im, t in zip(names, ims, toks): im.save(out / n); np.save(out / n.replace(".png", ".npy"), t)
        if not a.judge: continue
        for n in names:
            im = None
            for fa in r["facts"]:
                for ctx in ("fresh", "closed"):
                    if (n, fa["fid"], ctx) in done: continue
                    im = im or PIL.Image.open(out / n).convert("RGB")
                    j = A.judge(im, fa["question"], r["prompt"] if ctx == "closed" else None)
                    f.write(json.dumps({"pid": r["pid"], "image": n, "fid": fa["fid"], "ctx": ctx, "answer": j["answer"], "p_yes": j["p_yes"]}) + "\n")
            f.flush()
    print("done", a.out, a.part)

if __name__ == "__main__":
    main()
