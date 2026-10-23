import argparse, json, pathlib, sys, time
from copy import deepcopy
import torch
from PIL import Image
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from mucg.adapters.bagel import BagelAdapter, _sb
from mucg.adapters.base import yesno_from_text
from data.data_utils import pil_img2rgb

PREFIX = {"none": "", "think": "<think>\n\n</think>\n\n", "think1": "<think> </think> ", "answer": "Answer:"}
CONDS = ["vit_fresh", "vit_cl", "vae_fresh", "vae_cl"]
GENS = ["bagel", "janus7b", "showo2_7b"]


@torch.no_grad()
def forced_logits(inf, gen_context, prefix_ids):
    model = inf.model
    gc = deepcopy(gen_context)
    gi = model.prepare_start_tokens(gc["kv_lens"], gc["ropes"], inf.new_token_ids)
    curr = gi["packed_start_tokens"]; kvl = gi["key_values_lens"]; kvi = gi["packed_key_value_indexes"]
    pos = gi["packed_query_position_ids"]; past = gc["past_key_values"]
    extra = {"mode": "und"} if model.use_moe else {}
    seq = list(prefix_ids)
    while True:
        emb = model.language_model.model.embed_tokens(curr)
        qi = torch.cumsum(kvl, dim=0) + torch.arange(0, len(kvl), device=kvl.device, dtype=kvl.dtype)
        up = list(kvi.split(kvl.tolist(), dim=0))
        for i in range(len(up)): up[i] += i
        kvi = torch.cat(up, dim=0)
        out = model.language_model.forward_inference(packed_query_sequence=emb, query_lens=torch.ones_like(curr),
                                                     packed_query_position_ids=pos, packed_query_indexes=qi, past_key_values=past,
                                                     key_values_lens=kvl, packed_key_value_indexes=kvi, update_past_key_values=True,
                                                     is_causal=True, **extra)
        past = out.past_key_values
        if not seq: return model.language_model.lm_head(out.packed_query_sequence)[0].float()
        curr = torch.tensor([seq.pop(0)], device=curr.device, dtype=curr.dtype)
        up = list(kvi.split(kvl.tolist(), dim=0))
        for i in range(len(up)): up[i] = torch.cat([up[i], torch.tensor([up[i][-1] + 1], device=up[i].device)], dim=0)
        kvi = torch.cat(up, dim=0); kvl = kvl + 1; pos = pos + 1


def read(ad, logits):
    pr = torch.softmax(logits, -1); py, pn = pr[ad.yes_ids].sum().item(), pr[ad.no_ids].sum().item()
    raw = ad.inf.tokenizer.decode([int(pr.argmax())])
    return {"p_yes": py / (py + pn) if py + pn else None, "mass": py + pn, "answer": yesno_from_text(raw), "raw": raw}


@torch.no_grad()
def image_contexts(ad, image, prompt):
    inf = ad.inf; img = inf.vae_transform.resize_transform(pil_img2rgb(image)); out = {}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for cl in (False, True):
            for vae in (False, True):
                ctx = inf.init_gen_context()
                if cl: ctx = inf.update_context_text(prompt, ctx)
                out[f"{'vae' if vae else 'vit'}_{'cl' if cl else 'fresh'}"] = inf.update_context_image(img, ctx, vae=vae)
    return out


@torch.no_grad()
def ask(ad, ctx, q, prefix_ids):
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        c = ad.inf.update_context_text(q, deepcopy(ctx)); return read(ad, forced_logits(ad.inf, c, prefix_ids))


def facts():
    P = {r["pid"]: r for r in (json.loads(l) for l in open(CODE / "data/prompts.jsonl", encoding="utf-8"))}
    items = {}
    for g in GENS:
        for l in open(CODE / f"runs/judge/bagel_vae__{g}.jsonl"):
            r = json.loads(l)
            if r.get("perturb") or r.get("context"): continue
            q = next(f["question"] for f in P[r["pid"]]["facts"] if f["fid"] == r["fid"])
            items.setdefault((g, r["image"]), {"pid": r["pid"], "prompt": P[r["pid"]]["prompt"], "facts": {}})["facts"][r["fid"]] = q
    return items


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="think"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--smoke", type=int); a = ap.parse_args()
    items = facts(); keys = sorted(items); ad = BagelAdapter(); tok = ad.inf.tokenizer
    if a.smoke:
        stats = {k: [] for k in PREFIX}; n = 0
        for key in keys[::max(1, len(keys) // a.smoke)]:
            it = items[key]; ctx = image_contexts(ad, Image.open(CODE / f"runs/gen/{key[0]}" / key[1]), it["prompt"])
            for fid, q in list(it["facts"].items())[:1]:
                for k, s in PREFIX.items():
                    r = ask(ad, ctx["vae_fresh"], q, tok.encode(s) if s else []); stats[k].append((r["mass"], r["raw"]))
                n += 1
            if n >= a.smoke: break
        for k, v in stats.items():
            print(k, repr(PREFIX[k]), "mean mass", round(sum(m for m, _ in v) / len(v), 3), "raw", [x for _, x in v][:12], flush=True)
        return
    out = CODE / "runs/vae_format"; out.mkdir(parents=True, exist_ok=True); fo = out / f"bagel_{a.prefix}.jsonl"
    done = {(json.loads(l)["gen"], json.loads(l)["image"]) for l in open(fo)} if fo.exists() else set()
    i, n = map(int, a.shard.split("/")); keys = keys[i::n]
    pre = tok.encode(PREFIX[a.prefix]) if PREFIX[a.prefix] else []
    print(f"{len(keys)} images in shard {a.shard}; prefix {PREFIX[a.prefix]!r} -> {pre}", flush=True)
    for key in keys:
        if key in done: continue
        t = time.time(); it = items[key]; g, image = key
        ctx = image_contexts(ad, Image.open(CODE / f"runs/gen/{g}" / image), it["prompt"]); rows = []
        for fid, q in it["facts"].items():
            for c in CONDS:
                r = ask(ad, ctx[c], q, pre)
                rows.append({"gen": g, "image": image, "pid": it["pid"], "fid": fid, "cond": c, **r})
        with open(fo, "a") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        del ctx; torch.cuda.empty_cache()
        print(g, image, len(it["facts"]), f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
