import argparse, json, pathlib, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="janus7b", choices=["janus7b", "janus1b"]); ap.add_argument("--subset", required=True); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    from mucg.adapters.janus import JanusAdapter
    J = JanusAdapter(repo={"janus7b": "deepseek-ai/Janus-Pro-7B", "janus1b": "deepseek-ai/Janus-Pro-1B"}[a.judge])
    m, proc, tok = J.model, J.proc, J.tok
    layers = m.language_model.model.layers; L = len(layers)
    quart = [set(range(q * L // 4, (q + 1) * L // 4)) for q in range(4)]
    state = {"mask": None, "layers": set()}

    def pre_hook(idx):
        def h(mod, args, kwargs):
            if state["mask"] is not None and idx in state["layers"]: kwargs["attention_mask"] = state["mask"]
            return args, kwargs
        return h
    for i, ly in enumerate(layers): ly.register_forward_pre_hook(pre_hook(i), with_kwargs=True)

    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    sub = json.load(open(a.subset))
    out = ROOT / "runs/mask"; out.mkdir(parents=True, exist_ok=True)
    of = out / (f"mask_{a.judge}.jsonl" if a.part == 0 else f"mask_{a.judge}_p{a.part}.jsonl")
    done = set()
    if of.exists(): done = {(r["gen"], r["image"], r["fid"]) for r in (json.loads(l) for l in open(of))}
    f = open(of, "a")
    man = {}
    for gen, items in sub.items():
        man[gen] = {r["image"]: r for r in (json.loads(l) for l in open(ROOT / "runs/gen" / gen / "manifest.jsonl"))}
    for gen, items in sub.items():
        for image, fid, label in items[a.part::a.nparts]:
            if (gen, image, fid) in done: continue
            im = man[gen][image]; prior = im["prompt"]
            q = next(x["question"] for x in facts[im["pid"]]["facts"] if x["fid"] == fid)
            img = Image.open(ROOT / "runs/gen" / gen / image).convert("RGB")
            rows = []
            for ctx in ("fresh", "closed", "inline"):
                if ctx == "fresh":
                    conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
                elif ctx == "closed":
                    conv = [{"role": "<|User|>", "content": prior}, {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [img]},
                            {"role": "<|User|>", "content": q}, {"role": "<|Assistant|>", "content": ""}]
                else:
                    conv = [{"role": "<|User|>", "content": f"<image_placeholder>\nCaption: {prior}\n{q}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
                inp = proc(conversations=conv, images=[img], force_batchify=True).to(J.device)
                ids = inp.input_ids[0].tolist(); n = len(ids)
                is_img = inp.images_seq_mask[0].bool().tolist()
                pieces = [tok.decode([t]) for t in ids]; text, starts = "", []
                for pc in pieces: starts.append(len(text)); text += pc
                def span(s):
                    pos = text.find(s)
                    if pos < 0: pos = text.find(s.strip())
                    if pos < 0: return []
                    end = pos + len(s)
                    return [i for i, st in enumerate(starts) if st < end and st + len(pieces[i]) > pos]
                img_idx = [i for i, b in enumerate(is_img) if b]
                cap = span(prior) if ctx != "fresh" else []
                if ctx != "fresh": assert cap, (prior, text[:300])
                emb = m.prepare_inputs_embeds(**inp)
                dt = emb.dtype; neg = torch.finfo(dt).min
                base = torch.full((n, n), neg, dtype=dt, device=emb.device).triu(1)[None, None]
                conds = {"none": None}
                if ctx == "closed":
                    last_img, cap_end = max(img_idx), max(cap)
                    conds.update({"all": list(range(cap_end + 1, n)), "answer": [n - 1], "post_image": list(range(last_img + 1, n)),
                                  "image_only": list(range(cap_end + 1, last_img + 1))})
                elif ctx == "inline":
                    conds.update({"all": list(range(max(cap) + 1, n)), "answer": [n - 1]})
                for cname, qpos in conds.items():
                    for lname, lset in ([("all", set(range(L)))] + ([(f"Q{k+1}", quart[k]) for k in range(4)] + [(f"noQ{k+1}", set(range(L)) - quart[k]) for k in range(4)] if cname == "all" else [])):
                        if qpos is None:
                            state["mask"] = None
                            if lname != "all": continue
                        else:
                            mk = base.clone(); qi = torch.tensor(qpos, device=emb.device); ki = torch.tensor(cap, device=emb.device)
                            mk[0, 0, qi[:, None], ki[None, :]] = neg
                            state["mask"], state["layers"] = mk, lset
                        with torch.inference_mode():
                            o = m.language_model(inputs_embeds=emb, attention_mask=base if qpos is not None else inp.attention_mask, use_cache=False)
                        state["mask"] = None
                        pr = torch.softmax(o.logits[0, -1].float(), -1)
                        py, pn = pr[J.yes_ids].sum().item(), pr[J.no_ids].sum().item()
                        rows.append({"judge": a.judge, "gen": gen, "image": image, "pid": im["pid"], "fid": fid, "label": label,
                                     "ctx": ctx, "mask": cname, "layers": lname, "p_yes": py / (py + pn + 1e-12), "mass": py + pn})
                        if ctx == "fresh": break
            for r in rows: f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", of)

if __name__ == "__main__":
    main()
