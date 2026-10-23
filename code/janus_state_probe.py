import argparse, collections, json, pathlib, random, sys, time
import numpy as np, torch
from PIL import Image
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from mucg.adapters.base import load, yesno_from_text

CONDS = ["vit_fresh", "vit_cl", "state_fresh", "state_cl", "reenc_fresh", "reenc_cl",
         "statevit_fresh", "statevit_cl", "reencvit_fresh", "reencvit_cl"]
PH = "<image_placeholder>"


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


@torch.inference_mode()
def regenerate(ad, prompt, seeds):
    torch.manual_seed(seeds[0]); torch.cuda.manual_seed_all(seeds[0])
    n = len(seeds); m, proc, tok = ad.model, ad.proc, ad.tok
    conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=proc.sft_format, system_prompt="")
    ids = torch.LongTensor(tok.encode(sft + proc.image_start_tag))
    tokens = torch.zeros((n * 2, len(ids)), dtype=torch.int, device=ad.device)
    for i in range(n * 2):
        tokens[i] = ids
        if i % 2: tokens[i, 1:-1] = proc.pad_id
    emb = m.language_model.get_input_embeddings()(tokens)
    ntok = (ad.img_size // 16) ** 2
    gen = torch.zeros((n, ntok), dtype=torch.int, device=ad.device)
    out = None
    for i in range(ntok):
        out = m.language_model.model(inputs_embeds=emb, use_cache=True, past_key_values=out.past_key_values if i else None)
        logits = m.gen_head(out.last_hidden_state[:, -1, :])
        logits = logits[1::2] + ad.cfg * (logits[0::2] - logits[1::2])
        nxt = torch.multinomial(torch.softmax(logits / ad.temp, -1), 1)
        gen[:, i] = nxt.squeeze(-1)
        nxt = torch.cat([nxt, nxt], 1).view(-1)
        emb = m.prepare_gen_img_embeds(nxt).unsqueeze(1)
    dec = m.gen_vision_model.decode_code(gen, shape=[n, 8, ad.img_size // 16, ad.img_size // 16])
    dec = np.clip((dec.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return [Image.fromarray(d) for d in dec], gen.long(), ids


@torch.inference_mode()
def reencode(ad, img):
    x = torch.tensor(np.asarray(img.convert("RGB").resize((ad.img_size, ad.img_size)), dtype=np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1)[None].to(device=ad.device, dtype=torch.bfloat16)
    _, _, info = ad.model.gen_vision_model.encode(x)
    return info[2].view(1, -1).long()


class Assembler:
    def __init__(self, ad):
        self.ad, self.m, self.tok, self.proc = ad, ad.model, ad.tok, ad.proc
        self.emb = self.m.language_model.get_input_embeddings()
        v = self.tok.vocab
        self.img_id, self.boi, self.eoi = v[PH], v[self.proc.image_start_tag], v[self.proc.image_end_tag]

    def text_ids(self, s, bos):
        return torch.LongTensor(self.tok.encode(s, add_special_tokens=bos)).to(self.ad.device)

    def slot(self, kind, x):
        if kind == "gen": return self.m.prepare_gen_img_embeds(x.view(-1).to(self.ad.device))
        pv = self.proc.image_processor([x.convert("RGB")], return_tensors="pt").pixel_values.to(self.ad.device, torch.bfloat16)
        return self.m.aligner(self.m.vision_model(pv))[0]

    def expand(self, ids, slots):
        parts, start, k = [], 0, 0
        pos = (ids == self.img_id).nonzero().flatten().tolist()
        assert len(pos) == len(slots), (len(pos), len(slots))
        bo, eo = self.emb(torch.tensor([self.boi], device=self.ad.device)), self.emb(torch.tensor([self.eoi], device=self.ad.device))
        for p in pos:
            parts.append(self.emb(ids[start:p])); parts += [bo, slots[k], eo]; start = p + 1; k += 1
        parts.append(self.emb(ids[start:]))
        return torch.cat(parts, 0)

    def sft(self, conv):
        return self.proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=self.proc.sft_format,
                                                                   system_prompt=self.proc.system_prompt)

    def fresh(self, q, slots):
        s = self.sft([{"role": "<|User|>", "content": "\n".join([PH] * len(slots) + [q])}, {"role": "<|Assistant|>", "content": ""}])
        return self.expand(self.text_ids(s, True), slots)

    def run(self, prefix_ids, gen_ids, q, vit=None):
        user = f"{PH}\n{q}" if vit is not None else q
        s = self.sft([{"role": "<|User|>", "content": "R"}, {"role": "<|Assistant|>", "content": "@@IMG@@"},
                      {"role": "<|User|>", "content": user}, {"role": "<|Assistant|>", "content": ""}])
        post = s.split("@@IMG@@", 1)[1]
        head = torch.cat([self.emb(prefix_ids.to(self.ad.device)), self.slot("gen", gen_ids),
                          self.emb(torch.tensor([self.eoi], device=self.ad.device))], 0)
        tail = self.expand(self.text_ids(post, False), [self.slot("vit", vit)] if vit is not None else [])
        return torch.cat([head, tail], 0)


@torch.inference_mode()
def readout(ad, emb):
    out = ad.model.language_model(inputs_embeds=emb[None], use_cache=False)
    pr = torch.softmax(out.logits[0, -1].float(), -1)
    py, pn = pr[ad.yes_ids].sum().item(), pr[ad.no_ids].sum().item()
    raw = ad.tok.decode([int(pr.argmax())], skip_special_tokens=True)
    return {"p_yes": py / (py + pn) if py + pn else None, "mass": py + pn, "answer": yesno_from_text(raw), "raw": raw}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="janus7b"); ap.add_argument("--sat", type=int, default=150)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = CODE / "runs/state_janus"; out.mkdir(parents=True, exist_ok=True)
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
    ad = load(a.model); asm = Assembler(ad)
    for pid in pids:
        todo = [im for im in by_pid[pid] if im not in done]
        if not todo: continue
        t = time.time(); r = P[pid]; batch = man[todo[0]]["batch"]
        assert all(man[im]["batch"] == batch for im in todo)
        imgs_b, gen_b, prefix = regenerate(ad, r["prompt"], batch)
        for im in todo:
            k = batch.index(man[im]["seed"]); img, g = imgs_b[k], gen_b[k:k + 1]
            saved = np.asarray(Image.open(CODE / f"runs/gen/{a.model}" / im).convert("RGB"), dtype=np.int16)
            diff = np.abs(np.asarray(img, dtype=np.int16) - saved)
            ge = reencode(ad, Image.open(CODE / f"runs/gen/{a.model}" / im))
            agree = float((ge == g).float().mean())
            facts = [f for f in r["facts"] if f["fid"] in per.get(im, {})]
            sv = Image.open(CODE / f"runs/gen/{a.model}" / im).convert("RGB")
            with open(fo, "a") as f:
                for fact in facts:
                    q = fact["question"]; res = {}
                    res["vit_fresh"] = ad.judge(sv, q)
                    res["vit_cl"] = ad.judge(sv, q, prior_prompt=r["prompt"])
                    for name, ids in (("state", g), ("reenc", ge)):
                        res[f"{name}_fresh"] = readout(ad, asm.fresh(q, [asm.slot("gen", ids)]))
                        res[f"{name}_cl"] = readout(ad, asm.run(prefix, ids, q))
                        res[f"{name}vit_fresh"] = readout(ad, asm.fresh(q, [asm.slot("gen", ids), asm.slot("vit", sv)]))
                        res[f"{name}vit_cl"] = readout(ad, asm.run(prefix, ids, q, vit=sv))
                    if a.check:
                        chk = readout(ad, asm.fresh(q, [asm.slot("vit", sv)]))
                        print("check vit_fresh", round(chk["p_yes"], 4), round(res["vit_fresh"]["p_yes"], 4), flush=True)
                    for c in CONDS:
                        f.write(json.dumps({"image": im, "pid": pid, "fid": fact["fid"], "type": fact["type"], "cond": c,
                                            "label": per[im][fact["fid"]], "p_yes": res[c]["p_yes"], "mass": res[c]["mass"],
                                            "answer": res[c]["answer"], "raw": res[c]["raw"]}) + "\n")
            with open(fr, "a") as f:
                f.write(json.dumps({"image": im, "mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()),
                                    "token_agree_reenc": agree, "n_facts": len(facts), "sec": round(time.time() - t, 1)}) + "\n")
            print(im, f"diff {diff.mean():.3f} max {diff.max()} reenc-agree {agree:.3f} facts {len(facts)} {time.time() - t:.1f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
