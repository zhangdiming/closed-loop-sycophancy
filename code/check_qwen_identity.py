import glob, json, os, torch
from safetensors import safe_open
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
qs = glob.glob(f"{P}/hf-cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*")[0]; ex = f"{P}/models/qwen25vl7b_from_blip3o"
qm = json.load(open(f"{qs}/model.safetensors.index.json"))["weight_map"]; em = json.load(open(f"{ex}/model.safetensors.index.json"))["weight_map"]
bad, n = [], 0
for k in sorted(qm):
    with safe_open(f"{qs}/{qm[k]}", "pt") as a, safe_open(f"{ex}/{em[k]}", "pt") as b:
        x, y = a.get_tensor(k), b.get_tensor(k)
    if k in ("model.embed_tokens.weight", "lm_head.weight"): x, y = x[:151668], y[:151668]
    n += 1
    if x.shape != y.shape or not torch.equal(x, y): bad.append((k, tuple(x.shape), float((x.float() - y.float()).abs().max()) if x.shape == y.shape else None))
print(n, "tensors compared;", len(bad), "differ"); print(bad[:10])
