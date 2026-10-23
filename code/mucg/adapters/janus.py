import os, numpy as np, torch, PIL.Image
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg"))
os.environ.setdefault("HF_HUB_CACHE", f"{P}/hf-cache")
import janus
assert "repos/Janus" in os.path.abspath(janus.__file__), janus.__file__
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from .base import yesno_from_text

class JanusAdapter:
    can_generate = True

    def __init__(self, repo="deepseek-ai/Janus-Pro-7B", device="cuda", img_size=384, cfg_weight=5.0, temperature=1.0):
        self.proc: VLChatProcessor = VLChatProcessor.from_pretrained(repo)
        self.tok = self.proc.tokenizer
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.bfloat16).to(torch.bfloat16).to(device).eval()
        self.device, self.img_size, self.cfg, self.temp = device, img_size, cfg_weight, temperature
        one = lambda ws: [self.tok.encode(w, add_special_tokens=False) for w in ws]
        self.yes_ids = [i[0] for i in one(["Yes", " Yes", "yes", " yes"]) if len(i) == 1]
        self.no_ids = [i[0] for i in one(["No", " No", "no", " no"]) if len(i) == 1]

    @torch.inference_mode()
    def generate_batch(self, prompt, seeds):
        torch.manual_seed(seeds[0]); torch.cuda.manual_seed_all(seeds[0])
        n = len(seeds); m, proc = self.model, self.proc
        conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
        sft = proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=proc.sft_format, system_prompt="")
        ids = torch.LongTensor(self.tok.encode(sft + proc.image_start_tag))
        tokens = torch.zeros((n * 2, len(ids)), dtype=torch.int, device=self.device)
        for i in range(n * 2):
            tokens[i] = ids
            if i % 2: tokens[i, 1:-1] = proc.pad_id
        emb = m.language_model.get_input_embeddings()(tokens)
        ntok = (self.img_size // 16) ** 2
        gen = torch.zeros((n, ntok), dtype=torch.int, device=self.device)
        out = None
        for i in range(ntok):
            out = m.language_model.model(inputs_embeds=emb, use_cache=True, past_key_values=out.past_key_values if i else None)
            logits = m.gen_head(out.last_hidden_state[:, -1, :])
            logits = logits[1::2] + self.cfg * (logits[0::2] - logits[1::2])
            nxt = torch.multinomial(torch.softmax(logits / self.temp, -1), 1)
            gen[:, i] = nxt.squeeze(-1)
            nxt = torch.cat([nxt, nxt], 1).view(-1)
            emb = m.prepare_gen_img_embeds(nxt).unsqueeze(1)
        dec = m.gen_vision_model.decode_code(gen, shape=[n, 8, self.img_size // 16, self.img_size // 16])
        dec = np.clip((dec.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [PIL.Image.fromarray(d) for d in dec]

    def generate(self, prompt, seed=0):
        return self.generate_batch(prompt, [seed])[0]

    @torch.inference_mode()
    def judge(self, image, question, prior_prompt=None, mode="assistant"):
        if prior_prompt is None:
            conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{question}", "images": [image]},
                    {"role": "<|Assistant|>", "content": ""}]
        elif mode == "assistant":
            conv = [{"role": "<|User|>", "content": prior_prompt},
                    {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [image]},
                    {"role": "<|User|>", "content": question},
                    {"role": "<|Assistant|>", "content": ""}]
        elif mode == "user_prompt":
            conv = [{"role": "<|User|>", "content": f"{prior_prompt}\n<image_placeholder>\n{question}", "images": [image]},
                    {"role": "<|Assistant|>", "content": ""}]
        elif mode == "assistant_nocaption":
            conv = [{"role": "<|User|>", "content": "Generate an image."},
                    {"role": "<|Assistant|>", "content": "<image_placeholder>", "images": [image]},
                    {"role": "<|User|>", "content": question},
                    {"role": "<|Assistant|>", "content": ""}]
        elif mode in ("caption_inline", "request_inline"):
            lab = "Caption" if mode == "caption_inline" else "Request"
            conv = [{"role": "<|User|>", "content": f"<image_placeholder>\n{lab}: {prior_prompt}\n{question}", "images": [image]},
                    {"role": "<|Assistant|>", "content": ""}]
        else: raise ValueError(mode)
        pil = [image.convert("RGB")]
        inp = self.proc(conversations=conv, images=pil, force_batchify=True).to(self.device)
        emb = self.model.prepare_inputs_embeds(**inp)
        out = self.model.language_model(inputs_embeds=emb, attention_mask=inp.attention_mask, use_cache=False)
        probs = torch.softmax(out.logits[0, -1].float(), -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        top = int(probs.argmax())
        text = self.tok.decode([top], skip_special_tokens=True)
        z = p_yes + p_no
        return {"answer": yesno_from_text(text), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None,
                "mass": z, "raw": text}
