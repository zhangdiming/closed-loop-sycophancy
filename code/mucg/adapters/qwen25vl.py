import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .base import yesno_from_text

class Qwen25VLAdapter:
    can_generate = False

    def __init__(self, repo="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", dtype=torch.bfloat16, system=None, auto=False, weights=None, subfolder=""):
        self.system = system
        self.processor = AutoProcessor.from_pretrained(repo)
        cls = Qwen2_5_VLForConditionalGeneration
        if auto:
            from transformers import AutoModelForImageTextToText as cls
        self.model = cls.from_pretrained(weights or repo, subfolder=subfolder, torch_dtype=dtype, device_map=device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})

    @torch.no_grad()
    def judge(self, image, question, prior_prompt=None, mode="assistant"):
        img = {"type": "image", "image": image.convert("RGB")}; q = {"type": "text", "text": question}
        if prior_prompt is None:
            msgs = [{"role": "user", "content": [img, q]}]
        elif mode in ("assistant", "assistant_nocaption"):
            req = prior_prompt if mode == "assistant" else "Generate an image."
            msgs = [{"role": "user", "content": [{"type": "text", "text": req}]}, {"role": "assistant", "content": [img]},
                    {"role": "user", "content": [q]}]
        elif mode == "user_prompt":
            msgs = [{"role": "user", "content": [{"type": "text", "text": prior_prompt}, img, q]}]
        elif mode == "caption_inline":
            msgs = [{"role": "user", "content": [img, {"type": "text", "text": f"Caption: {prior_prompt}\n{question}"}]}]
        else: raise ValueError(mode)
        if self.system is not None: msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]}] + msgs
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        probs = torch.softmax(self.model(**inputs).logits[0, -1].float(), -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item(); z = p_yes + p_no
        txt = self.processor.tokenizer.decode([int(probs.argmax())])
        return {"answer": yesno_from_text(txt), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": txt}
