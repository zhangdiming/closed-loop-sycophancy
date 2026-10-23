import os, sys, glob, importlib.util, torch
from PIL import Image
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
os.environ.setdefault("HF_HUB_CACHE", f"{P}/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer, AutoProcessor, CLIPImageProcessor
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
from blip3o.model import blip3oQwenForCausalLM
from blip3o.conversation import conv_templates
from qwen_vl_utils import process_vision_info
from .base import yesno_from_text

class Blip3oAdapter:
    can_generate = True

    def __init__(self, device="cuda", guidance=3.0, steps=50):
        snap = sorted(glob.glob(f"{P}/hf-cache/models--BLIP3o--BLIP3o-Model-8B/snapshots/*"), key=os.path.getmtime)[-1]
        self.dev = torch.device(device); dt = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(snap, use_fast=False)
        self.model = blip3oQwenForCausalLM.from_pretrained(snap, low_cpu_mem_usage=True, torch_dtype=dt).to(self.dev).eval()
        if len(self.tok) > self.model.get_input_embeddings().weight.shape[0]: self.model.resize_token_embeddings(len(self.tok))
        dp = os.path.join(snap, "diffusion-decoder")
        spec = importlib.util.spec_from_file_location("pipeline_llava_gen", os.path.join(dp, "pipeline_llava_gen.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.pipe = mod.EmuVisualGenerationPipeline(
            tokenizer=self.tok, multimodal_encoder=self.model,
            scheduler=EulerDiscreteScheduler.from_pretrained(dp, subfolder="scheduler"),
            unet=UNet2DConditionModel.from_pretrained(dp, subfolder="unet", variant="bf16", torch_dtype=dt),
            vae=AutoencoderKL.from_pretrained(dp, subfolder="vae", variant="bf16", torch_dtype=dt),
            feature_extractor=CLIPImageProcessor.from_pretrained(dp, subfolder="feature_extractor"), safety_checker=None)
        self.pipe.vae.to(self.dev); self.pipe.unet.to(self.dev)
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.guidance, self.steps = guidance, steps
        one = lambda s: self.proc.tokenizer.encode(s, add_special_tokens=False)
        self.yes_ids = [one(s)[0] for s in ("Yes", " Yes", "yes", " yes") if len(one(s)) == 1]
        self.no_ids = [one(s)[0] for s in ("No", " No", "no", " no") if len(one(s)) == 1]

    @torch.inference_mode()
    def generate(self, prompt, seed=0):
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], f"Please generate image based on the following caption: {prompt}")
        conv.append_message(conv.roles[1], None)
        torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
        return self.pipe([conv.get_prompt()], guidance_scale=self.guidance, num_inference_steps=self.steps).image.convert("RGB")

    @torch.inference_mode()
    def judge(self, image, question, prior_prompt=None):
        img = image.convert("RGB")
        if prior_prompt is None:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": f"Please generate image based on the following caption: {prior_prompt}"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "Here is the generated image."}]},
                        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        im, vid = process_vision_info(messages)
        inputs = self.proc(text=[text], images=im, videos=vid, padding=True, return_tensors="pt").to(self.dev)
        probs = torch.softmax(self.model(**inputs).logits[0, -1].float(), -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        txt = self.proc.tokenizer.decode([int(probs.argmax())]); z = p_yes + p_no
        return {"answer": yesno_from_text(txt), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": txt}
