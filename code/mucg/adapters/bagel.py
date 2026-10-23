import os, sys, glob, importlib.util, torch
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_spec = importlib.util.spec_from_file_location("bagel_loader", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bagel_loader.py"))
_sb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sb)
from data.data_utils import pil_img2rgb
from .base import yesno_from_text

class BagelAdapter:
    can_generate = True

    def __init__(self, device="cuda", size=512, steps=50, cfg_text=4.0, max_memory_gib=40, vae_ctx=False):
        self.vae_ctx = vae_ctx
        snap = sorted(glob.glob(f"{P}/hf-cache/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/*"))[-1]
        self.inf = _sb.load_model(snap, max_memory_gib)
        self.size, self.steps, self.cfg_text = size, steps, cfg_text
        tok = self.inf.tokenizer
        one = lambda s: tok.encode(s)
        self.yes_ids = [one(s)[0] for s in ("Yes", " Yes", "yes", " yes") if len(one(s)) == 1]
        self.no_ids = [one(s)[0] for s in ("No", " No", "no", " no") if len(one(s)) == 1]

    def generate(self, prompt, seed=0):
        _sb.set_seed(seed)
        out = self.inf(text=prompt, think=False, cfg_text_scale=self.cfg_text, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0],
                       timestep_shift=3.0, num_timesteps=self.steps, cfg_renorm_min=0.0, cfg_renorm_type="global",
                       image_shapes=(self.size, self.size))
        return out["image"]

    @torch.no_grad()
    def judge(self, image, question, prior_prompt=None, mode="assistant"):
        inf = self.inf
        ctx = inf.init_gen_context()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if prior_prompt is not None and mode in ("assistant", "user_prompt"):
                ctx = inf.update_context_text(prior_prompt, ctx)
            elif prior_prompt is not None and mode == "assistant_nocaption":
                ctx = inf.update_context_text("Generate an image.", ctx)
            elif prior_prompt is not None and mode not in ("caption_inline", "request_inline"):
                raise ValueError(mode)
            img = inf.vae_transform.resize_transform(pil_img2rgb(image))
            ctx = inf.update_context_image(img, ctx, vae=self.vae_ctx)
            lab = {"caption_inline": "Caption", "request_inline": "Request"}.get(mode)
            q = f"{lab}: {prior_prompt}\n{question}" if (prior_prompt is not None and lab) else question
            ctx = inf.update_context_text(q, ctx)
            logits = _sb.first_token_logits(inf, ctx)
        probs = torch.softmax(logits, -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        text = inf.tokenizer.decode([int(probs.argmax())])
        z = p_yes + p_no
        return {"answer": yesno_from_text(text), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": text}
