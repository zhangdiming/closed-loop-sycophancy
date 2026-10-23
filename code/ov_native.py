import argparse, json, pathlib, re, torch
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent
SYS_PROMPT = (" You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning "
              "process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here")
def question(prompt):
    return f"""This image was generated from the prompt: {prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    Respond strictly in the following JSON format: """ + """

    {
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
    }
    """
TAGS = ("position", "color_attr", "counting"); GENS = ("bagel", "janus7b", "omnigen2")

def images():
    prompts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    out = []
    for g in GENS:
        for r in (json.loads(l) for l in open(ROOT / "runs/gen" / g / "manifest.jsonl")):
            if prompts[r["pid"]]["tag"] in TAGS and r["seed"] in (0, 1): out.append((g, r, prompts[r["pid"]]))
    return sorted(out, key=lambda x: (x[0], x[1]["image"]))

def parse(t):
    body = t.split("</think>")[-1]
    m = re.search(r'"answer"\s*:\s*"?(true|false)', body, re.I) or re.search(r"\b(true|false)\b", body, re.I)
    return m.group(1).lower() if m else None

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1); a = ap.parse_args()
    ims = images()
    sub = {}
    for g, r, pr in ims: sub.setdefault(g, []).extend([[r["image"], f["fid"]] for f in pr["facts"]])
    json.dump(sub, open(ROOT / "runs/pub/subset_ov.json", "w"))
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("comin/OmniVerifier-7B")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("comin/OmniVerifier-7B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
    of = ROOT / ("runs/pub/ov_native.jsonl" if a.part == 0 else f"runs/pub/ov_native_p{a.part}.jsonl")
    done = {(json.loads(l)["gen"], json.loads(l)["image"]) for l in open(of)} if of.exists() else set()
    f = open(of, "a")
    for g, r, pr in ims[a.part::a.nparts]:
        if (g, r["image"]) in done: continue
        img = Image.open(ROOT / "runs/gen" / g / r["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": SYS_PROMPT + question(r["prompt"])}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
        t = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        f.write(json.dumps({"judge": "omniverifier7b", "gen": g, "image": r["image"], "pid": r["pid"], "tag": pr["tag"], "raw": t, "answer": parse(t)}) + "\n"); f.flush()
    print("done", of)

if __name__ == "__main__":
    main()
