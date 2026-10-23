import argparse, glob, json, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from entail_filter import ENTAIL
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--out", default="runs/pub/entail_qwen25vl7b_text.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if "VL" in a.model:
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    else: cls = AutoModelForCausalLM
    m = cls.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})
    prompts = {r["pid"]: r["prompt"] for r in map(json.loads, open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    pairs = set()
    for f in glob.glob(str(ROOT / "runs/pub/decomp_*_p[0-9].json")):
        if "qwen3vl" in f: continue
        for pid, v in json.load(open(f)).items():
            for q in v["questions"]: pairs.add((pid, q))
    out = ROOT / a.out; res = json.load(open(out)) if out.exists() else {}
    todo = sorted(p for p in pairs if p[1] not in res.get(p[0], {})); print(len(pairs), "pairs,", len(todo), "to score")
    for i, (pid, q) in enumerate(todo):
        text = tok.apply_chat_template([{"role": "user", "content": ENTAIL.format(prompt=prompts[pid], q=q)}], tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(m.device)
        with torch.no_grad(): pr = torch.softmax(m(**inp).logits[0, -1].float(), -1)
        py, pn = pr[yes].sum().item(), pr[no].sum().item(); res.setdefault(pid, {})[q] = py / (py + pn + 1e-12)
        if i % 500 == 0: json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"), indent=0); print("done", out)

if __name__ == "__main__":
    main()
