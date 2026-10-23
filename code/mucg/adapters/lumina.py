import os, sys, glob, tempfile, torch, torch.nn.functional as F
from PIL import Image
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg")); REPO = f"{P}/repos/Lumina-DiMOO"
if REPO not in sys.path: sys.path.insert(0, REPO)
from transformers import AutoTokenizer
from diffusers import VQModel
from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_breaks,
                               generate_crop_size_list, var_center_crop)
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
from .base import yesno_from_text
MASK, NL, BOA, EOA, BOI, EOI = (SPECIAL_TOKENS[k] for k in ("mask_token", "newline_token", "answer_start", "answer_end", "boi", "eoi"))
SYS_MMU = ("You are a multimodal model that can process both text and images. Answer the following question based on the "
           "provided images. Analyze each image and combine relevant details to answer.")

class LuminaAdapter:
    can_generate = True

    def __init__(self, device="cuda", size=512, steps=32, cfg=4.0, gen_length=8, judge_steps=1):
        torch.set_grad_enabled(False)
        self.ckpt = sorted(glob.glob(f"{P}/hf-cache/models--Alpha-VLLM--Lumina-DiMOO/snapshots/*"))[-1]
        self.tok = AutoTokenizer.from_pretrained(self.ckpt, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(self.ckpt, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.vq = VQModel.from_pretrained(self.ckpt, subfolder="vqvae").to(device)
        self.vae_scale = 2 ** (len(self.vq.config.block_out_channels) - 1)
        self.device, self.size, self.steps, self.cfg, self.gen_length, self.judge_steps = device, size, steps, cfg, gen_length, judge_steps
        self.crops = generate_crop_size_list((1024 // 32) ** 2, 32)
        one = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.yes_ids = [one(s)[0] for s in ("Yes", " Yes", "yes", " yes") if len(one(s)) == 1]
        self.no_ids = [one(s)[0] for s in ("No", " No", "no", " no") if len(one(s)) == 1]
        self.tmp = tempfile.mkdtemp()

    def generate(self, prompt, seed=0):
        setup_seed(int(seed) + 1)
        H = W = self.size
        seq_len, newline_every, gh, gw = calculate_vq_params(H, W)
        inp, unc = generate_text_to_image_prompt(prompt, create_prompt_templates())
        con = self.tok(inp)["input_ids"]; uncon = self.tok(unc)["input_ids"]
        img_mask = add_break_line([MASK] * seq_len, gh, gw, new_number=NL)
        ids = torch.tensor(con + [BOA, BOI] + img_mask + [EOI, EOA], device=self.device).unsqueeze(0)
        unc_ids = torch.tensor(uncon, device=self.device).unsqueeze(0)
        vq_tokens = generate_image(self.model, ids, seq_len=seq_len, newline_every=newline_every, timesteps=self.steps,
                                   temperature=1.0, cfg_scale=self.cfg, uncon_ids=unc_ids, code_start=len(con) + 2, use_cache=False)
        out = decode_vq_to_image(vq_tokens, f"{self.tmp}/g.png", vae_ckpt=self.ckpt, image_height=H, image_width=W, vqvae=self.vq)
        return out.convert("RGB")

    def _img_tokens(self, image):
        im = var_center_crop(image.convert("RGB"), crop_size_list=self.crops)
        iw, ih = im.size
        _, _, gh, gw = calculate_vq_params(ih, iw, self.vae_scale)
        return add_break_line(encode_img_with_breaks(im, vqvae=self.vq), gh, gw, new_number=NL)

    def judge(self, image, question, prior_prompt=None):
        tok = self.tok
        img_tok = self._img_tokens(image)
        if prior_prompt is None:
            ids = tok("<system>" + SYS_MMU + "</system>" + "<user>" + question + "</user>")["input_ids"]
            seq = ids[:-1] + img_tok + ids[-1:]
        else:
            head = tok("<system>" + SYS_MMU + "</system>" + "<user>" + prior_prompt + "</user>")["input_ids"]
            q = tok("<user>" + question + "</user>", add_special_tokens=False)["input_ids"]
            seq = head + [BOA, BOI] + img_tok + [EOI, EOA] + q
        code_start = len(seq) + 1
        x = torch.tensor(seq + [BOA] + [MASK] * self.gen_length + [EOA], device=self.device).unsqueeze(0)
        probs = None
        n_steps = self.judge_steps
        lo, hi = code_start, code_start + self.gen_length
        per_step = max(1, self.gen_length // n_steps)
        for step in range(n_steps):
            mask_index = (x == MASK)
            logits = self.model(x, infer=True).logits
            if step == 0:
                probs = F.softmax(logits[0, code_start].float(), -1)
            x0 = torch.argmax(logits, dim=-1)
            p_blk = F.softmax(logits[:, lo:hi].to(torch.float64), dim=-1)
            conf = torch.full(x0.shape, -float("inf"), dtype=torch.float64, device=x0.device)
            conf[:, lo:hi] = torch.gather(p_blk, -1, x0[:, lo:hi].unsqueeze(-1)).squeeze(-1)
            conf = torch.where(mask_index, conf, -float("inf"))
            k = min(per_step, int(mask_index[0, lo:hi].sum()))
            if k <= 0: break
            _, sel = torch.topk(conf[0], k=k)
            x[0, sel] = x0[0, sel]
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        text = "yes" if p_yes >= p_no else "no"
        z = p_yes + p_no
        return {"answer": yesno_from_text(text), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": text}
