import os, sys, glob, torch
from PIL import Image
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg")); REPO = f"{P}/repos/OmniGen2"
os.environ.setdefault("HF_HUB_CACHE", f"{P}/hf-cache")
if REPO not in sys.path: sys.path.insert(0, REPO)
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from .base import yesno_from_text
NEG = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, "
       "(extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar")
SYS = "You are a helpful assistant that generates high-quality images based on user instructions."
IMG = "<img1>: <|vision_start|><|image_pad|><|vision_end|>"

class OmniGen2Adapter:
    can_generate = True

    def __init__(self, device="cuda", size=512, steps=50, text_guidance=4.0):
        snap = os.path.dirname(glob.glob(f"{P}/hf-cache/models--OmniGen2--OmniGen2/snapshots/*/model_index.json")[0])
        dt = torch.bfloat16
        self.pipe = OmniGen2Pipeline.from_pretrained(snap, torch_dtype=dt, trust_remote_code=True)
        self.pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(snap, subfolder="transformer", torch_dtype=dt)
        self.pipe = self.pipe.to(device)
        self.chat = OmniGen2ChatPipeline(transformer=self.pipe.transformer, vae=self.pipe.vae, scheduler=self.pipe.scheduler,
                                         mllm=self.pipe.mllm, processor=self.pipe.processor)
        self.device, self.size, self.steps, self.tg = device, size, steps, text_guidance
        tok = self.pipe.processor.tokenizer
        one = lambda s: tok.encode(s, add_special_tokens=False)
        self.yes_ids = [one(s)[0] for s in ("Yes", " Yes", "yes", " yes") if len(one(s)) == 1]
        self.no_ids = [one(s)[0] for s in ("No", " No", "no", " no") if len(one(s)) == 1]
        self.tok = tok

    @torch.no_grad()
    def generate(self, prompt, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(prompt=prompt, input_images=None, width=self.size, height=self.size, num_inference_steps=self.steps,
                        max_sequence_length=1024, text_guidance_scale=self.tg, image_guidance_scale=1.0, cfg_range=(0.0, 1.0),
                        negative_prompt=NEG, num_images_per_prompt=1, generator=gen, output_type="pil")
        return out.images[0]

    @torch.no_grad()
    def judge(self, image, question, prior_prompt=None, mode="assistant"):
        img = image.convert("RGB")
        head = f"<|im_start|>system\n{SYS}<|im_end|>\n"
        if prior_prompt is None:
            text = self.chat._apply_chat_template(question, [img])
        elif mode in ("assistant", "assistant_nocaption"):
            req = prior_prompt if mode == "assistant" else "Generate an image."
            text = (f"{head}<|im_start|>user\n{req}<|im_end|>\n"
                    f"<|im_start|>assistant\n{IMG}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
        elif mode == "user_prompt":
            text = f"{head}<|im_start|>user\n{prior_prompt}\n{IMG}{question}<|im_end|>\n<|im_start|>assistant\n"
        elif mode == "caption_inline":
            text = f"{head}<|im_start|>user\n{IMG}Caption: {prior_prompt}\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else: raise ValueError(mode)
        inputs = self.chat.prepare_inputs_for_text_generation(text, [img], self.pipe.mllm.device)
        probs = torch.softmax(self.pipe.mllm(**inputs).logits[0, -1].float(), -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        txt = self.tok.decode([int(probs.argmax())]); z = p_yes + p_no
        return {"answer": yesno_from_text(txt), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": txt}
