import importlib

REGISTRY = {
    "janus7b": ("mucg.adapters.janus", "JanusAdapter", {"repo": "deepseek-ai/Janus-Pro-7B"}),
    "janus1b": ("mucg.adapters.janus", "JanusAdapter", {"repo": "deepseek-ai/Janus-Pro-1B"}),
    "showo2_7b": ("mucg.adapters.showo2", "ShowO2Adapter", {"size": "7B"}),
    "showo2_1.5b": ("mucg.adapters.showo2", "ShowO2Adapter", {"size": "1.5B"}),
    "bagel": ("mucg.adapters.bagel", "BagelAdapter", {}),
    "bagel_vae": ("mucg.adapters.bagel", "BagelAdapter", {"vae_ctx": True}),
    "blip3o": ("mucg.adapters.blip3o", "Blip3oAdapter", {}),
    "omnigen2": ("mucg.adapters.omnigen2", "OmniGen2Adapter", {}),
    "lumina": ("mucg.adapters.lumina", "LuminaAdapter", {}),
    "qwen3vl8b": ("mucg.adapters.qwen3vl", "Qwen3VLAdapter", {"repo": "Qwen/Qwen3-VL-8B-Instruct"}),
    "qwen25vl7b": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "Qwen/Qwen2.5-VL-7B-Instruct"}),
    "qwen25vl3b": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "weights": "OmniGen2/OmniGen2", "subfolder": "mllm"}),
    "qwen25vl3b_gsys": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "weights": "OmniGen2/OmniGen2", "subfolder": "mllm",
                        "system": "You are a helpful assistant that generates high-quality images based on user instructions."}),
    "internvl3_2b": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "OpenGVLab/InternVL3-2B-hf", "auto": True}),
    "internvl3_8b": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "OpenGVLab/InternVL3-8B-hf", "auto": True}),
    "omniverifier7b": ("mucg.adapters.qwen25vl", "Qwen25VLAdapter", {"repo": "comin/OmniVerifier-7B"}),
    "owlv2": ("mucg.adapters.owlv2", "Owlv2Oracle", {}),
}

def load(name, **overrides):
    mod, cls, kw = REGISTRY[name]
    kw = {**kw, **overrides}
    return getattr(importlib.import_module(mod), cls)(**kw)

def yesno_from_text(text):
    t = text.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"): return "no"
    return t[:20]
