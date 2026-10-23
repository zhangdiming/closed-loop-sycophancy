import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from .base import yesno_from_text

class Qwen3VLAdapter:
    can_generate = False

    def __init__(self, repo="Qwen/Qwen3-VL-8B-Instruct", device="cuda", dtype=torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype, device_map=device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})
        self.device = device

    @torch.no_grad()
    def judge(self, image: Image.Image, question: str, prior_prompt=None):
        if prior_prompt is None:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": f"Generate an image: {prior_prompt}"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "Here is the generated image."}]},
                        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        logits = out.scores[0][0].float()
        probs = torch.softmax(logits, -1)
        p_yes = probs[self.yes_ids].sum().item(); p_no = probs[self.no_ids].sum().item()
        z = p_yes + p_no
        text = self.processor.batch_decode(out.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"answer": yesno_from_text(text), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None,
                "mass": z, "raw": text}
