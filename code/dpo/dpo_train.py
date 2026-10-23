import argparse, json, math, pathlib, random, sys, numpy as np, torch, torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mucg.adapters.base import load

def prompt_ids(A, prompt):
    conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft = A.proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=A.proc.sft_format, system_prompt="")
    return torch.LongTensor(A.tok.encode(sft + A.proc.image_start_tag))

def seq_logp(m, ids, toks):
    n = toks.shape[0]; L = ids.shape[0]
    te = m.language_model.get_input_embeddings()(ids.unsqueeze(0).expand(n, -1))
    ie = m.prepare_gen_img_embeds(toks[:, :-1].reshape(-1)).view(n, toks.shape[1] - 1, -1)
    h = m.language_model.model(inputs_embeds=torch.cat([te, ie.to(te.dtype)], 1)).last_hidden_state[:, L - 1:]
    logits = m.gen_head(h).float()
    return torch.gather(F.log_softmax(logits, -1), 2, toks.unsqueeze(-1)).squeeze(-1).sum(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--dir", default="runs/dpo/train"); ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    pairs = json.load(open(ROOT / a.pairs)); out = ROOT / f"runs/dpo/ckpt_{a.name}"; out.mkdir(parents=True, exist_ok=True)
    A = load("janus1b"); m = A.model; dev = A.device
    ref = load("janus1b").model; ref.requires_grad_(False)
    m.requires_grad_(False); m.language_model.float(); m.language_model.requires_grad_(True); m.language_model.train()
    m.language_model.model.gradient_checkpointing_enable() if hasattr(m.language_model.model, "gradient_checkpointing_enable") else None
    opt = torch.optim.AdamW([p for p in m.language_model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(pairs) * a.epochs / a.accum); warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    tok = lambda n: torch.from_numpy(np.load(ROOT / a.dir / n.replace(".png", ".npy")).astype(np.int64)).to(dev)
    log = open(out / "log.jsonl", "w"); k = 0; acc = []; margins = []
    for ep in range(a.epochs):
        random.shuffle(pairs)
        for p in pairs:
            ids = prompt_ids(A, p["prompt"]).to(dev); toks = torch.stack([tok(p["chosen"]), tok(p["rejected"])])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): r = seq_logp(ref, ids, toks)
            with torch.autocast("cuda", dtype=torch.bfloat16): pi = seq_logp(m, ids, toks)
            z = a.beta * ((pi[0] - r[0]) - (pi[1] - r[1])); loss = -F.logsigmoid(z) / a.accum; loss.backward()
            acc.append(float(z > 0)); margins.append(float(z)); k += 1
            if k % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(m.language_model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                s = k // a.accum
                if s % 10 == 0 or s == steps:
                    rec = {"step": s, "epoch": ep, "reward_acc": float(np.mean(acc[-80:])), "margin": float(np.mean(margins[-80:]))}
                    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save({k_: v.to(torch.bfloat16) for k_, v in m.language_model.state_dict().items()}, out / "lm.pt")
    print("saved", out)

if __name__ == "__main__":
    main()
