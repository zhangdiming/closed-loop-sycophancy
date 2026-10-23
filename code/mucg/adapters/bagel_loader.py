import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy

P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
REPO = os.path.join(P, "repos", "Bagel")
sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

T2I_PROMPT = "a red cup to the left of a blue book on a wooden table"
QUESTION = "Is the red cup to the left of the blue book? Answer yes or no."
EDIT_PROMPT = "make the cup green"


def find_snapshot():
    pat = os.path.join(P, "hf-cache", "models--ByteDance-Seed--BAGEL-7B-MoT", "snapshots", "*")
    need = ["llm_config.json", "vit_config.json", "ema.safetensors", "ae.safetensors"]
    for s in sorted(glob.glob(pat)):
        if all(os.path.exists(os.path.join(s, f)) for f in need):
            return s
    raise FileNotFoundError(f"no complete BAGEL-7B-MoT snapshot under {pat}")


def nvsmi_used_mib():
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", idx],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception as e:
        return -1


class Step:
    def __init__(self, name, results):
        self.name, self.results = name, results

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        print(f"\n===== [{self.name}] start", flush=True)
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        dt = time.time() - self.t0
        rec = dict(
            wall_s=round(dt, 2),
            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
            peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 3),
            nvsmi_used_MiB_at_end=nvsmi_used_mib(),
        )
        self.results[self.name] = rec
        print(f"===== [{self.name}] done: {json.dumps(rec)}", flush=True)
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, max_memory_gib):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    assert torch.cuda.device_count() == 1, f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}"
    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{max_memory_gib}GiB"},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )
    same_device_modules = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
                           "vae2llm", "llm2vae", "connector", "vit_pos_embed"]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device if k in device_map else "cuda:0"
    placements = sorted({str(v) for v in device_map.values()})
    print("device_map placements:", placements, "| entries:", len(device_map), flush=True)
    offloaded = {k: v for k, v in device_map.items() if str(v) not in ("0", "cuda:0")}
    assert not offloaded, f"some modules would be CPU/disk offloaded: {offloaded}"

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=os.path.join(P, "envs", "bagel_offload_unused"),
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    dev = torch.device("cuda:0")
    vae_model = vae_model.to(dev).eval()
    _enc, _dec = vae_model.encode, vae_model.decode
    vae_model.encode = lambda x: _enc(x.to(dev))
    vae_model.decode = lambda z: _dec(z.to(dev))

    n_params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"Bagel params: {n_params/1e9:.2f}B, dtypes={dtypes}; VAE dtype={next(vae_model.parameters()).dtype}", flush=True)

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids,
    )
    return inferencer


@torch.no_grad()
def first_token_logits(inf, gen_context):
    model = inf.model
    gc = deepcopy(gen_context)
    gi = model.prepare_start_tokens(gc["kv_lens"], gc["ropes"], inf.new_token_ids)
    curr_tokens = gi["packed_start_tokens"]
    key_values_lens = gi["key_values_lens"]
    packed_key_value_indexes = gi["packed_key_value_indexes"]
    packed_query_position_ids = gi["packed_query_position_ids"]

    packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
    query_lens = torch.ones_like(curr_tokens)
    packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
        0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype)
    uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for i in range(len(uppacked)):
        uppacked[i] += i
    packed_key_value_indexes = torch.cat(uppacked, dim=0)
    extra = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=query_lens,
        packed_query_position_ids=packed_query_position_ids,
        packed_query_indexes=packed_query_indexes,
        past_key_values=gc["past_key_values"],
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes,
        update_past_key_values=True,
        is_causal=True,
        **extra,
    )
    logits = model.language_model.lm_head(output.packed_query_sequence)
    return logits[0].float()


@torch.no_grad()
def yes_no_probs(inf, image, question):
    tok = inf.tokenizer
    gen_context = inf.init_gen_context()
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        img = inf.vae_transform.resize_transform(pil_img2rgb(image))
        gen_context = inf.update_context_image(img, gen_context, vae=False)
        gen_context = inf.update_context_text(question, gen_context)
        logits = first_token_logits(inf, gen_context)
    probs = torch.softmax(logits, dim=-1)

    def single_id(s):
        ids = tok.encode(s)
        return ids[0] if len(ids) == 1 else None

    variants = {}
    for s in ["Yes", "No", " Yes", " No", "yes", "no", "YES", "NO"]:
        i = single_id(s)
        variants[s] = dict(token_id=i, prob=(float(probs[i]) if i is not None else None),
                           n_tokens=len(tok.encode(s)))
    p_yes, p_no = variants["Yes"]["prob"], variants["No"]["prob"]
    top = torch.topk(probs, 10)
    top10 = [(tok.decode([int(i)]), int(i), round(float(p), 6)) for p, i in zip(top.values, top.indices)]
    out = dict(
        P_Yes=p_yes, P_No=p_no,
        P_Yes_normalized=(p_yes / (p_yes + p_no) if p_yes is not None and p_no is not None else None),
        logit_Yes=float(logits[variants["Yes"]["token_id"]]), logit_No=float(logits[variants["No"]["token_id"]]),
        variants=variants, top10=top10, argmax_token=top10[0][0],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None, help="snapshot dir; default: auto-detect in hf-cache")
    ap.add_argument("--out_dir", default=os.path.join(P, "envs"))
    ap.add_argument("--t2i_size", type=int, default=1024, help="square T2I resolution (repo default 1024)")
    ap.add_argument("--num_timesteps", type=int, default=50)
    ap.add_argument("--max_memory_gib", type=int, default=44)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_edit", action="store_true")
    args = ap.parse_args()

    model_path = args.model_path or find_snapshot()
    print("model_path:", model_path)
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0),
          "| CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    import transformers, flash_attn
    print("transformers", transformers.__version__, "| flash_attn", flash_attn.__version__, flush=True)

    results = dict(model_path=model_path, t2i_size=args.t2i_size, num_timesteps=args.num_timesteps,
                   seed=args.seed, placement="single GPU cuda:0, bf16, no CPU offload")
    t2i_path = os.path.join(args.out_dir, "smoke_bagel_t2i.png")
    edit_path = os.path.join(args.out_dir, "smoke_bagel_edit.png")

    with Step("0_load", results):
        inf = load_model(model_path, args.max_memory_gib)
    results["0_load"]["model_alloc_GiB_after_load"] = round(torch.cuda.memory_allocated() / 2**30, 3)

    with Step("1_t2i", results):
        set_seed(args.seed)
        out = inf(text=T2I_PROMPT, think=False,
                  cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0],
                  timestep_shift=3.0, num_timesteps=args.num_timesteps,
                  cfg_renorm_min=0.0, cfg_renorm_type="global",
                  image_shapes=(args.t2i_size, args.t2i_size))
        t2i_img = out["image"]
        t2i_img.save(t2i_path)
    results["1_t2i"].update(prompt=T2I_PROMPT, image=t2i_path, size=list(t2i_img.size))
    print("saved", t2i_path, t2i_img.size, flush=True)

    with Step("2_understand", results):
        set_seed(args.seed)
        out = inf(image=t2i_img, text=QUESTION, think=False, understanding_output=True,
                  do_sample=False, text_temperature=0.3, max_think_token_n=64)
        answer = out["text"]
    results["2_understand"].update(question=QUESTION, answer=answer)
    print(f"QUESTION: {QUESTION}\nANSWER: {answer!r}", flush=True)

    with Step("3_yes_no_probs", results):
        yn = yes_no_probs(inf, t2i_img, QUESTION)
    results["3_yes_no_probs"].update(yn)
    print(f"P(Yes)={yn['P_Yes']:.6f}  P(No)={yn['P_No']:.6f}  P(Yes|Yes,No)={yn['P_Yes_normalized']:.4f}")
    print("top10 first-token candidates:", yn["top10"], flush=True)

    if not args.skip_edit:
        with Step("4_edit", results):
            set_seed(args.seed)
            out = inf(image=t2i_img, text=EDIT_PROMPT, think=False,
                      cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
                      timestep_shift=3.0, num_timesteps=args.num_timesteps,
                      cfg_renorm_min=0.0, cfg_renorm_type="text_channel")
            edit_img = out["image"]
            edit_img.save(edit_path)
        results["4_edit"].update(prompt=EDIT_PROMPT, image=edit_path, size=list(edit_img.size))
        print("saved", edit_path, edit_img.size, flush=True)

    results["overall_peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    res_path = os.path.join(args.out_dir, "smoke_bagel_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nRESULTS:", json.dumps(results, indent=2, default=str))
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
