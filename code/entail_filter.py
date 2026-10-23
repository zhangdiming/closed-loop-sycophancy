import glob, json, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parent
ENTAIL = ("Text-to-image prompt: \"{prompt}\"\nQuestion about a generated image: \"{q}\"\n"
          "Suppose the image shows exactly what the prompt states and nothing contradicts it. Is the answer to the question "
          "then necessarily yes, because the prompt itself states it? Answer no if the prompt does not state it, or if "
          "the answer could be no without violating the prompt. Answer yes or no.")

def main():
    from mucg.adapters.qwen3vl import Qwen3VLAdapter
    A = Qwen3VLAdapter()
    prompts = {r["pid"]: r["prompt"] for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    pairs = set()
    for f in glob.glob(str(ROOT / "runs/pub/decomp_*.json")):
        for pid, v in json.load(open(f)).items():
            for q in v["questions"]: pairs.add((pid, q))
    out = ROOT / "runs/pub/entail_scores.json"; res = json.load(open(out)) if out.exists() else {}
    todo = sorted(p for p in pairs if q_missing(res, p))
    print(len(pairs), "pairs,", len(todo), "to score")
    for i, (pid, q) in enumerate(todo):
        msgs = [{"role": "user", "content": [{"type": "text", "text": ENTAIL.format(prompt=prompts[pid], q=q)}]}]
        inp = A.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(A.model.device)
        with torch.no_grad(): pr = torch.softmax(A.model(**inp).logits[0, -1].float(), -1)
        py, pn = pr[A.yes_ids].sum().item(), pr[A.no_ids].sum().item()
        res.setdefault(pid, {})[q] = py / (py + pn + 1e-12)
        if i % 500 == 0: json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"), indent=0); print("done", out)

def q_missing(res, p):
    return p[1] not in res.get(p[0], {})

if __name__ == "__main__":
    main()
