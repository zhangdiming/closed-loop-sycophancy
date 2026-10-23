import argparse, json, pathlib, random, sys, time
from copy import deepcopy
import numpy as np, torch
from PIL import Image
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from mucg.adapters.bagel import BagelAdapter, _sb
from data.data_utils import pil_img2rgb

CONDS = ["vit_fresh", "vit_cl", "vae_fresh", "vae_cl", "state_fresh", "state_cl", "statevae_fresh", "statevae_cl"]


class ExactLatentVAE:
    def __init__(self, vae, latent): self.vae, self.latent = vae, latent
    def encode(self, x): return self.latent.to(device=x.device if x.is_cuda else "cuda")
    def decode(self, z): return self.vae.decode(z)


def select(n_sat, seed=0):
    from bootstrap_ci import load_rows, labels, ROOT
    rows = load_rows(ROOT / "runs/judge"); o, _ = labels(rows, "human")
    per = {}
    for (g, image, fid), l in o.items():
        if g == "bagel": per.setdefault(image, {})[fid] = l
    viol = sorted(i for i, d in per.items() if "no" in d.values())
    sat = sorted(i for i, d in per.items() if d and all(v == "yes" for v in d.values()))
    random.Random(seed).shuffle(sat)
    return viol + sat[:n_sat], per


@torch.no_grad()
def regenerate(ad, prompt, seed):
    inf = ad.inf; _sb.set_seed(seed)
    gen = inf.init_gen_context(); cfg_img = deepcopy(gen)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        cfg_text = deepcopy(gen)
        gen = inf.update_context_text(prompt, gen); cfg_img = inf.update_context_text(prompt, cfg_img)
        keep = deepcopy(gen)
        shape = (ad.size, ad.size); lat = {}
        orig = inf.decode_image
        def grab(latent, image_shape): lat["z"] = latent; return orig(latent, image_shape)
        inf.decode_image = grab
        try:
            img = inf.gen_image(shape, gen, cfg_text_precontext=cfg_text, cfg_img_precontext=cfg_img, cfg_text_scale=ad.cfg_text,
                                cfg_img_scale=1.0, cfg_interval=[0.4, 1.0], timestep_shift=3.0, num_timesteps=ad.steps,
                                cfg_renorm_min=0.0, cfg_renorm_type="global")
        finally:
            inf.decode_image = orig
    m = inf.model; h = w = ad.size // m.latent_downsample; p = m.latent_patch_size
    z = lat["z"].reshape(1, h, w, p, p, m.latent_channel)
    z = torch.einsum("nhwpqc->nchpwq", z).reshape(1, m.latent_channel, h * p, w * p)
    return img, z, keep


@torch.no_grad()
def contexts(ad, img, z, gen_ctx, prompt):
    inf = ad.inf; im = inf.vae_transform.resize_transform(pil_img2rgb(img)); out = {}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        def base(cl): return inf.update_context_text(prompt, inf.init_gen_context()) if cl else inf.init_gen_context()
        for cl, suf in ((False, "fresh"), (True, "cl")):
            out[f"vit_{suf}"] = inf.update_context_image(im, base(cl), vae=False)
            out[f"vae_{suf}"] = inf.update_context_image(im, base(cl), vae=True)
            real = inf.vae_model; inf.vae_model = ExactLatentVAE(real, z)
            try:
                b = deepcopy(gen_ctx) if cl else base(False)
                out[f"state_{suf}"] = inf.update_context_image(im, b, vae=True, vit=True)
                b = deepcopy(gen_ctx) if cl else base(False)
                out[f"statevae_{suf}"] = inf.update_context_image(im, b, vae=True, vit=False)
            finally:
                inf.vae_model = real
    return out


@torch.no_grad()
def ask(ad, ctx, q):
    inf = ad.inf
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        c = inf.update_context_text(q, deepcopy(ctx)); logits = _sb.first_token_logits(inf, c)
    pr = torch.softmax(logits.float(), -1); py, pn = pr[ad.yes_ids].sum().item(), pr[ad.no_ids].sum().item()
    return py / (py + pn) if py + pn else None, py + pn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); ap.add_argument("--sat", type=int, default=150)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    out = CODE / "runs/state"; out.mkdir(parents=True, exist_ok=True)
    fo, fr = out / "bagel_state.jsonl", out / "regen.jsonl"
    done = {json.loads(l)["image"] for l in open(fr)} if fr.exists() else set()
    imgs, per = select(a.sat)
    i, n = map(int, a.shard.split("/")); imgs = imgs[i::n]
    if a.limit: imgs = imgs[: a.limit]
    print(f"{len(imgs)} images in shard {a.shard}", flush=True)
    P = {r["pid"]: r for r in (json.loads(l) for l in open(CODE / "data/prompts.jsonl", encoding="utf-8"))}
    ad = BagelAdapter()
    for image in imgs:
        if image in done: continue
        t = time.time(); pid, seed = image[:-4].rsplit("_s", 1); seed = int(seed); r = P[pid]
        img, z, gctx = regenerate(ad, r["prompt"], seed)
        saved = np.asarray(Image.open(CODE / "runs/gen/bagel" / image).convert("RGB"), dtype=np.int16)
        diff = np.abs(np.asarray(img.convert("RGB"), dtype=np.int16) - saved)
        ctxs = contexts(ad, img, z, gctx, r["prompt"])
        with open(fo, "a") as f:
            for fact in r["facts"]:
                if fact["fid"] not in per.get(image, {}): continue
                for c in CONDS:
                    py, mass = ask(ad, ctxs[c], fact["question"])
                    f.write(json.dumps({"image": image, "pid": pid, "fid": fact["fid"], "type": fact["type"], "cond": c,
                                        "label": per[image][fact["fid"]], "p_yes": py, "mass": mass}) + "\n")
        with open(fr, "a") as f:
            f.write(json.dumps({"image": image, "mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()),
                                "frac_identical": float((diff.max(-1) == 0).mean()), "sec": round(time.time() - t, 1)}) + "\n")
        del ctxs, gctx; torch.cuda.empty_cache()
        print(image, f"diff {diff.mean():.3f} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
