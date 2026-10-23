import argparse, json, pathlib, torch
import torch.nn.functional as Fn
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent

STATE = {"layer": None, "layers": set(), "keys": None, "queries": None}


def install(nav, model):
    orig = nav.flash_attn_varlen_func

    def masked_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, **kw):
        if STATE["keys"] is None or STATE["layer"] not in STATE["layers"]:
            return orig(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=causal, **kw)
        Lq, Lk = q.shape[0], k.shape[0]
        qpos = torch.arange(Lk - Lq, Lk, device=q.device)
        sel = STATE["queries"](qpos)
        if not bool(sel.any()):
            return orig(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=causal, **kw)
        kpos = torch.arange(Lk, device=q.device)
        allow = (kpos[None, :] <= qpos[:, None]) if causal else torch.ones(Lq, Lk, dtype=torch.bool, device=q.device)
        blk = torch.zeros(Lk, dtype=torch.bool, device=q.device); blk[STATE["keys"][STATE["keys"] < Lk]] = True
        allow = allow & ~(sel[:, None] & blk[None, :])
        rep = q.shape[1] // k.shape[1]
        qq, kk, vv = q.transpose(0, 1)[None], k.repeat_interleave(rep, 1).transpose(0, 1)[None], v.repeat_interleave(rep, 1).transpose(0, 1)[None]
        o = Fn.scaled_dot_product_attention(qq, kk, vv, attn_mask=allow[None, None])
        return o[0].transpose(0, 1).contiguous()

    nav.flash_attn_varlen_func = masked_flash
    layers = model.language_model.model.layers
    for i, ly in enumerate(layers):
        def h(mod, args, kwargs, i=i):
            STATE["layer"] = i
        ly.self_attn.register_forward_pre_hook(h, with_kwargs=True)
    return len(layers)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", required=True)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from mucg.adapters.bagel import BagelAdapter, _sb
    from data.data_utils import pil_img2rgb
    import modeling.bagel.qwen2_navit as nav
    A = BagelAdapter(); inf = A.inf; tok = inf.tokenizer
    L = install(nav, inf.model); quart = [set(range(q * L // 4, (q + 1) * L // 4)) for q in range(4)]
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    sub = json.load(open(a.subset))
    out = ROOT / "runs/mask"; out.mkdir(parents=True, exist_ok=True)
    of = out / ("mask_bagel.jsonl" if a.part == 0 else f"mask_bagel_p{a.part}.jsonl")
    done = {(r["gen"], r["image"], r["fid"]) for r in (json.loads(l) for l in open(of))} if of.exists() else set()
    f = open(of, "a")
    man = {g: {r["image"]: r for r in (json.loads(l) for l in open(ROOT / "runs/gen" / g / "manifest.jsonl"))} for g in sub}

    def text_span(segment_text, sub_text, seg_start):
        ids = tok.encode(segment_text); pieces = [tok.decode([t]) for t in ids]; s, starts = "", []
        for pc in pieces: starts.append(len(s)); s += pc
        p0 = s.find(sub_text); assert p0 >= 0, (segment_text, sub_text); p1 = p0 + len(sub_text)
        return [seg_start + 1 + i for i, st in enumerate(starts) if st < p1 and st + len(pieces[i]) > p0]

    def run(img_t, q, prior, ctx_kind, keys=None, queries=None, layers=frozenset()):
        STATE.update(keys=keys, queries=queries, layers=set(layers))
        try:
            ctx = inf.init_gen_context(); seg = {}
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                if ctx_kind == "closed":
                    ctx = inf.update_context_text(prior, ctx); seg["req_end"] = ctx["kv_lens"][0]
                seg["img_start"] = ctx["kv_lens"][0]
                ctx = inf.update_context_image(img_t, ctx, vae=False); seg["img_end"] = ctx["kv_lens"][0]
                text = f"Caption: {prior}\n{q}" if ctx_kind == "inline" else q
                ctx = inf.update_context_text(text, ctx); seg["answer"] = ctx["kv_lens"][0]
                logits = _sb.first_token_logits(inf, ctx)
        finally:
            STATE.update(keys=None, queries=None, layers=set())
        pr = torch.softmax(logits.float(), -1)
        py, pn = pr[A.yes_ids].sum().item(), pr[A.no_ids].sum().item()
        return py / (py + pn + 1e-12), py + pn, seg

    n_done = 0
    for gen, items in sub.items():
        for image, fid, label in items[a.part::a.nparts]:
            if (gen, image, fid) in done: continue
            if a.limit and n_done >= a.limit: break
            im = man[gen][image]; prior = im["prompt"]
            q = next(x["question"] for x in facts[im["pid"]]["facts"] if x["fid"] == fid)
            img_t = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(ROOT / "runs/gen" / gen / image).convert("RGB")))
            rows = []
            def add(ctx, mask, layers, p, m):
                rows.append({"judge": "bagel", "gen": gen, "image": image, "pid": im["pid"], "fid": fid, "label": label,
                             "ctx": ctx, "mask": mask, "layers": layers, "p_yes": p, "mass": m})
            p, m, _ = run(img_t, q, prior, "fresh"); add("fresh", "none", "all", p, m)
            p, m, seg = run(img_t, q, prior, "closed"); add("closed", "none", "all", p, m)
            keys = torch.arange(1, seg["req_end"] - 1, device="cuda")
            s1, s2, s3 = seg["img_start"], seg["img_end"], seg["answer"]
            conds = {"all": lambda x: x >= s1, "post_image": lambda x: x >= s2,
                     "image_only": lambda x: (x >= s1) & (x < s2), "answer": lambda x: x == s3}
            for cname, qf in conds.items():
                lsets = [("all", set(range(L)))] + ([(f"Q{k+1}", quart[k]) for k in range(4)] if cname == "all" else [])
                for lname, lset in lsets:
                    p, m, _ = run(img_t, q, prior, "closed", keys, qf, lset); add("closed", cname, lname, p, m)
            p, m, seg = run(img_t, q, prior, "inline"); add("inline", "none", "all", p, m)
            cap = text_span(f"Caption: {prior}\n{q}", prior, seg["img_end"])
            keys = torch.tensor(cap, device="cuda"); c1, s3 = max(cap), seg["answer"]
            for cname, qf in {"all": lambda x: x > c1, "answer": lambda x: x == s3}.items():
                p, m, _ = run(img_t, q, prior, "inline", keys, qf, set(range(L))); add("inline", cname, "all", p, m)
            for r in rows: f.write(json.dumps(r) + "\n")
            f.flush(); n_done += 1
    print("done", of)


if __name__ == "__main__":
    main()
