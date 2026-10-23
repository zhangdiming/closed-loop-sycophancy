import argparse, collections, json, pathlib, sys
import numpy as np, torch
from PIL import Image
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from mucg.adapters.base import load
import showo2_state_probe as S


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="showo2_1.5b"); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    imgs, per = S.select(a.model, 150)
    man = {json.loads(l)["image"]: json.loads(l) for l in open(CODE / f"runs/gen/{a.model}/manifest.jsonl")}
    P = {r["pid"]: r for r in (json.loads(l) for l in open(CODE / "data/prompts.jsonl", encoding="utf-8"))}
    viol = [im for im in imgs if "no" in per[im].values()]
    pick = viol[:: max(1, len(viol) // a.n)][: a.n]
    ad = load(a.model); fo = CODE / f"runs/state_showo2/{a.model}_latent_dist.jsonl"
    from datasets.utils import image_transform
    rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0))
    for im in pick:
        m = man[im]; r = P[m["pid"]]; batch = m["batch"]
        imgs_b, lat_b = S.regenerate(ad, r["prompt"], batch); k = batch.index(m["seed"]); z_run = lat_b[k:k + 1]
        sv = Image.open(CODE / f"runs/gen/{a.model}" / im).convert("RGB")
        x = image_transform(sv, resolution=ad.res).to(ad.device).unsqueeze(0).to(ad.dt).unsqueeze(2)
        z_det = ad.vae.sample(x, deterministic=True).squeeze(2).to(ad.dt)
        z_s1 = ad.vae.sample(x).squeeze(2).to(ad.dt); z_s2 = ad.vae.sample(x).squeeze(2).to(ad.dt)
        flips = collections.Counter()
        for f in r["facts"]:
            if f["fid"] not in per.get(im, {}): continue
            for cl in (False, True):
                a1 = S.judge_latent(ad, z_s1, f["question"], r["prompt"] if cl else None)
                a2 = S.judge_latent(ad, z_s2, f["question"], r["prompt"] if cl else None)
                c = "cl" if cl else "fresh"
                flips[f"n_{c}"] += 1
                flips[f"pyes_{c}"] += (a1["p_yes"] < 0.5) != (a2["p_yes"] < 0.5)
                flips[f"dec_{c}"] += (a1["answer"] == "no") != (a2["answer"] == "no")
        row = {"image": im, "rel_run_det": rel(z_run, z_det), "rel_s1_det": rel(z_s1, z_det), "rel_s1_s2": rel(z_s1, z_s2),
               "rel_run_s1": rel(z_run, z_s1), "cos_run_det": cos(z_run, z_det), "cos_s1_s2": cos(z_s1, z_s2), **flips}
        with open(fo, "a") as fh: fh.write(json.dumps(row) + "\n")
        print(im, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k != "image"}, flush=True)


if __name__ == "__main__":
    main()
