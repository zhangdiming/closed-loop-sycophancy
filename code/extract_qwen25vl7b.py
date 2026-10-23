import glob, json, os, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
bs = glob.glob(f"{P}/hf-cache/models--BLIP3o--BLIP3o-Model-8B/snapshots/*")[0]
qs = glob.glob(f"{P}/hf-cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*")[0]
out = f"{P}/models/qwen25vl7b_from_blip3o"; os.makedirs(out, exist_ok=True)
qmap = json.load(open(f"{qs}/model.safetensors.index.json"))["weight_map"]
bmap = json.load(open(f"{bs}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{qs}/config.json")); V = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
for f in os.listdir(qs):
    if f.endswith((".json", ".txt")) and f != "model.safetensors.index.json": shutil.copy(os.path.realpath(f"{qs}/{f}"), f"{out}/{f}")
shards = {}
for k in qmap: shards.setdefault(bmap[k], []).append(k)
new_map = {}
for i, (sf, keys) in enumerate(sorted(shards.items())):
    t = {}
    with safe_open(f"{bs}/{sf}", "pt") as fh:
        for k in keys:
            x = fh.get_tensor(k)
            if k in ("model.embed_tokens.weight", "lm_head.weight") and x.shape[0] != V:
                print(k, tuple(x.shape), "->", V)
                x = torch.cat([x, x.new_zeros(V - x.shape[0], x.shape[1])]) if x.shape[0] < V else x[:V]
            t[k] = x.contiguous()
    name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"; save_file(t, f"{out}/{name}", metadata={"format": "pt"})
    new_map.update({k: name for k in keys}); print(name, len(keys))
json.dump({"metadata": {}, "weight_map": new_map}, open(f"{out}/model.safetensors.index.json", "w"))
print("done", out, len(new_map))
