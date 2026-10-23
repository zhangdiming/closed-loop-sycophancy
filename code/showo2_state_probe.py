import argparse, collections, json, pathlib, random, sys, time
import numpy as np, torch
from PIL import Image
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from mucg.adapters.base import load, yesno_from_text

CONDS = ["main_fresh", "main_cl", "state_fresh", "state_cl"]


def select(model, n_sat, seed=0):
    from bootstrap_ci import load_rows, labels, ROOT
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    per = {}
    for (g, image, fid), l in o.items():
        if g == model: per.setdefault(image, {})[fid] = l
    viol = sorted(i for i, d in per.items() if "no" in d.values())
    sat = sorted(i for i, d in per.items() if d and all(v == "yes" for v in d.values()))
    random.Random(seed).shuffle(sat)
    return viol + sat[:n_sat], per


@torch.no_grad()
def regenerate(ad, prompt, seeds):
    from mucg.adapters.showo2 import prepare_gen_input, omni_attn_mask_naive, denorm
    n = len(seeds)
    btt, btt_null, bmp, bmp_null = prepare_gen_input([prompt] * n, ad.tok, ad.n_t2i, ad.bos_id, ad.eos_id, ad.boi_id,
                                                     ad.eoi_id, ad.pad_id, ad.img_pad_id, ad.max_text_len, ad.device)
    zs = [torch.randn((1, ad.latent_dim, ad.lh * ad.patch, ad.lw * ad.patch),
                      generator=torch.Generator(device=ad.device).manual_seed(int(s)), device=ad.device) for s in seeds]
    z = torch.cat(zs, 0).to(ad.dt); z = torch.cat([z, z], 0)
    text_tokens = torch.cat([btt, btt_null], 0); mp = torch.cat([bmp, bmp_null], 0)
    mask = omni_attn_mask_naive(text_tokens.size(0), ad.max_seq_len, mp, ad.device).to(ad.dt)
    kw = dict(text_tokens=text_tokens, attention_mask=mask, modality_positions=mp, output_hidden_states=True,
              max_seq_len=ad.max_seq_len, guidance_scale=ad.cfg_scale)
    samples = ad.sample_fn(z, ad.model.t2i_generate, **kw)[-1]
    samples = torch.chunk(samples, 2)[0]
    images = ad.vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
    return [Image.fromarray(im) for im in denorm(images)], samples.to(ad.dt)


@torch.no_grad()
def judge_latent(ad, lat, question, prior_prompt=None):
    from mucg.adapters.showo2 import omni_attn_mask_naive
    m, tok, dev = ad.model, ad.tok, ad.device
    und = m.image_embedder_und(lat); gen = m.image_embedder_gen(lat)
    und = m.und_trans(und + m.position_embedding(m.image_position_ids))["last_hidden_state"]
    img_emb = m.fusion_proj(torch.cat([und, gen], -1))
    sys_ids = tok("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False)["input_ids"]
    role_a = tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = tok("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q_ids = tok(question, add_special_tokens=False).input_ids
    if prior_prompt is not None:
        role_a = role_a + tok(prior_prompt, add_special_tokens=False).input_ids + tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        role_b = tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"] + q_ids + role_b
        q_ids = []
    ta = torch.tensor([ad.sid["bos_id"]] + sys_ids + role_a, device=dev)[None]
    tb = torch.tensor([ad.sid["boi_id"], ad.sid["eoi_id"]] + q_ids + role_b, device=dev)[None]
    ea, eb = m.showo.model.embed_tokens(ta), m.showo.model.embed_tokens(tb)
    te = m.time_embed(torch.Tensor([[1.0]]).to(dev), ea.dtype)
    if hasattr(m, "time_embed_proj"): te = m.time_embed_proj(te)
    emb = torch.cat([ea, eb[:, :1], te, img_emb, eb[:, 1:]], 1).to(ad.dt)
    mp = torch.tensor([ta.shape[1] + 2, ad.n_mmu])[None, None, :].to(dev)
    mask = omni_attn_mask_naive(B=1, LEN=emb.size(1), modalities=mp, device=dev, inverted=True).to(emb.dtype)
    probs = torch.softmax(m.showo(inputs_embeds=emb, attention_mask=mask)["logits"][0, -1].float(), -1)
    py, pn = probs[ad.yes_ids].sum().item(), probs[ad.no_ids].sum().item()
    raw = tok.decode([int(probs.argmax())])
    return {"p_yes": py / (py + pn) if py + pn else None, "mass": py + pn, "answer": yesno_from_text(raw), "raw": raw}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="showo2_1.5b"); ap.add_argument("--sat", type=int, default=150)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = CODE / "runs/state_showo2"; out.mkdir(parents=True, exist_ok=True)
    fo, fr = out / f"{a.model}_state.jsonl", out / f"{a.model}_regen.jsonl"
    done = {json.loads(l)["image"] for l in open(fr)} if fr.exists() else set()
    imgs, per = select(a.model, a.sat)
    man = {json.loads(l)["image"]: json.loads(l) for l in open(CODE / f"runs/gen/{a.model}/manifest.jsonl")}
    P = {r["pid"]: r for r in (json.loads(l) for l in open(CODE / "data/prompts.jsonl", encoding="utf-8"))}
    by_pid = collections.defaultdict(list)
    for im in imgs: by_pid[man[im]["pid"]].append(im)
    pids = sorted(by_pid); i, n = map(int, a.shard.split("/")); pids = pids[i::n]
    if a.limit: pids = pids[: a.limit]
    print(f"{sum(len(by_pid[p]) for p in pids)} images / {len(pids)} prompts in shard {a.shard}", flush=True)
    ad = load(a.model)
    for pid in pids:
        todo = [im for im in by_pid[pid] if im not in done]
        if not todo: continue
        r = P[pid]
        for batch in sorted({tuple(man[im]["batch"]) for im in todo}):
          t = time.time(); batch = list(batch)
          imgs_b, lat_b = regenerate(ad, r["prompt"], batch)
          for im in [x for x in todo if man[x]["batch"] == batch]:
              k = batch.index(man[im]["seed"]); img, lat = imgs_b[k], lat_b[k:k + 1]
              sv = Image.open(CODE / f"runs/gen/{a.model}" / im).convert("RGB")
              diff = np.abs(np.asarray(img, dtype=np.int16) - np.asarray(sv, dtype=np.int16))
              facts = [f for f in r["facts"] if f["fid"] in per.get(im, {})]
              with open(fo, "a") as f:
                  for fact in facts:
                      q = fact["question"]
                      res = {"main_fresh": ad.judge(sv, q), "main_cl": ad.judge(sv, q, prior_prompt=r["prompt"]),
                             "state_fresh": judge_latent(ad, lat, q), "state_cl": judge_latent(ad, lat, q, prior_prompt=r["prompt"])}
                      if a.check:
                          from datasets.utils import image_transform
                          x = image_transform(sv, resolution=ad.res).to(ad.device).unsqueeze(0)
                          lz = ad.vae.sample(x.to(ad.dt).unsqueeze(2), deterministic=True).squeeze(2).to(ad.dt)
                          print("check", round(judge_latent(ad, lz, q)["p_yes"], 4), round(res["main_fresh"]["p_yes"], 4),
                                "latent L2 run-vs-reenc", round(float((lat.float() - lz.float()).norm() / lz.float().norm()), 4), flush=True)
                      for c in CONDS:
                          f.write(json.dumps({"image": im, "pid": pid, "fid": fact["fid"], "type": fact["type"], "cond": c,
                                              "label": per[im][fact["fid"]], "p_yes": res[c]["p_yes"], "mass": res[c]["mass"],
                                              "answer": res[c]["answer"], "raw": res[c]["raw"]}) + "\n")
              with open(fr, "a") as f:
                  f.write(json.dumps({"image": im, "mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()),
                                      "n_facts": len(facts), "sec": round(time.time() - t, 1)}) + "\n")
              print(im, f"diff {diff.mean():.3f} max {diff.max()} facts {len(facts)} {time.time() - t:.1f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
