import json, pathlib, sys, torch
from pub_protocols import DECOMP, TAGS, parse_questions
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    from mucg.adapters.qwen3vl import Qwen3VLAdapter
    A = Qwen3VLAdapter()
    prompts = [r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8")) if r["tag"] in TAGS]
    part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    prompts = prompts[::-1][part::nparts] if nparts > 1 else prompts
    out = ROOT / ("runs/pub/decomp_qwen3vl.json" if nparts == 1 else f"runs/pub/decomp_qwen3vl_p{part}.json")
    dec = json.load(open(out)) if out.exists() else {}
    for r in prompts:
        if r["pid"] in dec: continue
        msgs = [{"role": "user", "content": [{"type": "text", "text": DECOMP.format(prompt=r["prompt"])}]}]
        inp = A.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(A.model.device)
        with torch.no_grad(): o = A.model.generate(**inp, max_new_tokens=200, do_sample=False)
        raw = A.processor.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        dec[r["pid"]] = {"raw": raw, "questions": parse_questions(raw)}
        json.dump(dec, open(out, "w"), indent=1)
    print("done", out, len(dec))

if __name__ == "__main__":
    main()
