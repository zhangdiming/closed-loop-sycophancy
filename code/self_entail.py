import argparse, glob, json, pathlib, torch
from entail_filter import ENTAIL
ROOT = pathlib.Path(__file__).resolve().parent

def text_pyes(name, A, text):
    if name.startswith("janus"):
        conv = [{"role": "<|User|>", "content": text}, {"role": "<|Assistant|>", "content": ""}]
        sft = A.proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=A.proc.sft_format, system_prompt=A.proc.system_prompt)
        ids = torch.tensor([A.tok.encode(sft)], device=A.device)
        logits = A.model.language_model(inputs_embeds=A.model.language_model.get_input_embeddings()(ids)).logits[0, -1]
    elif name == "bagel":
        from mucg.adapters.bagel import _sb
        inf = A.inf; ctx = inf.init_gen_context()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            ctx = inf.update_context_text(text, ctx); logits = _sb.first_token_logits(inf, ctx)
    elif name == "omnigen2":
        t = A.chat._apply_chat_template(text, None)
        inp = A.chat.prepare_inputs_for_text_generation(t, None, A.pipe.mllm.device)
        logits = A.pipe.mllm(**inp).logits[0, -1]
    pr = torch.softmax(logits.float(), -1)
    py, pn = pr[A.yes_ids].sum().item(), pr[A.no_ids].sum().item()
    return py / (py + pn + 1e-12), py + pn

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True, choices=["bagel", "janus7b", "omnigen2"]); a = ap.parse_args()
    from mucg.adapters.base import load
    A = load(a.judge)
    prompts = {r["pid"]: r["prompt"] for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    dec = {}
    for f in sorted(glob.glob(str(ROOT / f"runs/pub/decomp_{a.judge}_p*.json"))): dec.update(json.load(open(f)))
    out = ROOT / f"runs/pub/entail_self_{a.judge}.json"; res = json.load(open(out)) if out.exists() else {}
    masses = []
    for i, (pid, v) in enumerate(sorted(dec.items())):
        for q in v["questions"]:
            if q in res.get(pid, {}): continue
            p, m = text_pyes(a.judge, A, ENTAIL.format(prompt=prompts[pid], q=q)); res.setdefault(pid, {})[q] = p; masses.append(m)
        if i % 50 == 0: json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"), indent=0)
    print("done", out, sum(len(v) for v in res.values()), "questions; mean yes/no mass", sum(masses) / max(len(masses), 1))

if __name__ == "__main__":
    main()
