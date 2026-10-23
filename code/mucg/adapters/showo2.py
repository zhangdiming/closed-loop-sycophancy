import os, sys, torch
from PIL import Image
P = os.environ.get("MUCG_HOME", os.path.expanduser("~/mucg")); REPO = f"{P}/repos/Show-o/show-o2"
os.environ.setdefault("HF_HUB_CACHE", f"{P}/hf-cache")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
if REPO not in sys.path: sys.path.insert(0, REPO)
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download, snapshot_download
import models.modeling_showo2_qwen2_5 as showo2_mod
from models.modeling_siglip import SiglipModel as _RealSiglip
from transformers.models.siglip.configuration_siglip import SiglipConfig

class _SiglipFromConfig:
    @staticmethod
    def from_pretrained(path, *a, **k): return _RealSiglip(SiglipConfig.from_pretrained(path))
showo2_mod.SiglipModel = _SiglipFromConfig
from models import Showo2Qwen2_5, omni_attn_mask_naive, WanVAE
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_hyper_params, path_to_llm_name, denorm
from datasets.utils import image_transform
from transport import Sampler, create_transport
from .base import yesno_from_text

CONFIGS = {"7B": f"{REPO}/configs/showo2_7b_demo_432x432.yaml", "1.5B": f"{REPO}/configs/showo2_1.5b_demo_432x432.yaml"}

class ShowO2Adapter:
    can_generate = True

    def __init__(self, size="7B", device="cuda", guidance_scale=5.0, steps=50):
        self.device, self.dt, self.cfg_scale, self.steps = torch.device(device), torch.bfloat16, guidance_scale, steps
        cfg = OmegaConf.load(CONFIGS[size])
        cfg.model.vae_model.pretrained_model_path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
        self.vae = WanVAE(vae_pth=cfg.model.vae_model.pretrained_model_path, dtype=self.dt, device=self.device)
        self.tok, self.sid = get_text_tokenizer(cfg.model.showo.llm_model_path, add_showo_tokens=True,
                                                return_showo_token_ids=True, llm_name=path_to_llm_name[cfg.model.showo.llm_model_path])
        cfg.model.showo.llm_vocab_size = len(self.tok)
        local = snapshot_download(cfg.model.showo.pretrained_model_path, local_files_only=True)
        self.model = Showo2Qwen2_5.from_pretrained(local, use_safetensors=False, torch_dtype=self.dt,
                                                   load_from_showo=True).to(self.device).to(self.dt).eval()
        if cfg.model.showo.add_time_embeds:
            cfg.dataset.preprocessing.num_t2i_image_tokens += 1
            cfg.dataset.preprocessing.num_mmu_image_tokens += 1
            cfg.dataset.preprocessing.num_video_tokens += 1
        self.cfg = cfg
        self.hp = get_hyper_params(cfg, self.tok, self.sid)
        (self.n_t2i, self.n_mmu, _, self.max_seq_len, self.max_text_len, self.latent_dim, self.patch, self.lw, self.lh,
         self.pad_id, self.bos_id, self.eos_id, self.boi_id, self.eoi_id, *_rest) = self.hp
        self.img_pad_id = self.hp[16]
        tr = create_transport(path_type=cfg.transport.path_type, prediction=cfg.transport.prediction, loss_weight=cfg.transport.loss_weight,
                              train_eps=cfg.transport.train_eps, sample_eps=cfg.transport.sample_eps, snr_type=cfg.transport.snr_type,
                              do_shift=cfg.transport.do_shift, seq_len=self.n_t2i)
        self.sample_fn = Sampler(tr).sample_ode(sampling_method=cfg.transport.sampling_method, num_steps=steps, atol=cfg.transport.atol,
                                                rtol=cfg.transport.rtol, reverse=cfg.transport.reverse,
                                                time_shifting_factor=cfg.transport.time_shifting_factor)
        self.res = cfg.dataset.preprocessing.resolution
        one = lambda w: self.tok.encode(w, add_special_tokens=False)
        self.yes_ids = [one(w)[0] for w in ("Yes", " Yes", "yes", " yes") if len(one(w)) == 1]
        self.no_ids = [one(w)[0] for w in ("No", " No", "no", " no") if len(one(w)) == 1]

    @torch.no_grad()
    def generate_batch(self, prompt, seeds):
        n = len(seeds)
        btt, btt_null, bmp, bmp_null = prepare_gen_input([prompt] * n, self.tok, self.n_t2i, self.bos_id, self.eos_id, self.boi_id,
                                                         self.eoi_id, self.pad_id, self.img_pad_id, self.max_text_len, self.device)
        zs = [torch.randn((1, self.latent_dim, self.lh * self.patch, self.lw * self.patch),
                          generator=torch.Generator(device=self.device).manual_seed(int(s)), device=self.device) for s in seeds]
        z = torch.cat(zs, 0).to(self.dt); z = torch.cat([z, z], 0)
        text_tokens = torch.cat([btt, btt_null], 0); mp = torch.cat([bmp, bmp_null], 0)
        mask = omni_attn_mask_naive(text_tokens.size(0), self.max_seq_len, mp, self.device).to(self.dt)
        kw = dict(text_tokens=text_tokens, attention_mask=mask, modality_positions=mp, output_hidden_states=True,
                  max_seq_len=self.max_seq_len, guidance_scale=self.cfg_scale)
        samples = self.sample_fn(z, self.model.t2i_generate, **kw)[-1]
        samples = torch.chunk(samples, 2)[0]
        images = self.vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
        return [Image.fromarray(im) for im in denorm(images)]

    def generate(self, prompt, seed=0):
        return self.generate_batch(prompt, [seed])[0]

    @torch.no_grad()
    def judge(self, image, question, prior_prompt=None, mode="assistant"):
        m, tok, dev = self.model, self.tok, self.device
        img = image_transform(image.convert("RGB"), resolution=self.res).to(dev).unsqueeze(0)
        lat = self.vae.sample(img.to(self.dt).unsqueeze(2)).squeeze(2).to(self.dt)
        und = m.image_embedder_und(lat); gen = m.image_embedder_gen(lat)
        und = m.und_trans(und + m.position_embedding(m.image_position_ids))["last_hidden_state"]
        img_emb = m.fusion_proj(torch.cat([und, gen], -1))
        sys_ids = tok("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False)["input_ids"]
        role_a = tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        role_b = tok("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        q_ids = tok(question, add_special_tokens=False).input_ids
        if prior_prompt is not None and mode in ("assistant", "assistant_nocaption"):
            prior_ids = tok(prior_prompt if mode == "assistant" else "Generate an image.", add_special_tokens=False).input_ids
            role_a = role_a + prior_ids + tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
            role_b = tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"] + q_ids + role_b
            q_ids = []
        elif prior_prompt is not None and mode == "user_prompt":
            role_a = role_a + tok(prior_prompt + "\n", add_special_tokens=False)["input_ids"]
        elif prior_prompt is not None and mode in ("caption_inline", "request_inline"):
            q_ids = tok(f"{'Caption' if mode == 'caption_inline' else 'Request'}: {prior_prompt}\n", add_special_tokens=False)["input_ids"] + q_ids
        ta = torch.tensor([self.sid["bos_id"]] + sys_ids + role_a, device=dev)[None]
        tb = torch.tensor([self.sid["boi_id"], self.sid["eoi_id"]] + q_ids + role_b, device=dev)[None]
        ea, eb = m.showo.model.embed_tokens(ta), m.showo.model.embed_tokens(tb)
        te = m.time_embed(torch.Tensor([[1.0]]).to(dev), ea.dtype)
        if hasattr(m, "time_embed_proj"): te = m.time_embed_proj(te)
        emb = torch.cat([ea, eb[:, :1], te, img_emb, eb[:, 1:]], 1).to(self.dt)
        mp = torch.tensor([ta.shape[1] + 2, self.n_mmu])[None, None, :].to(dev)
        mask = omni_attn_mask_naive(B=1, LEN=emb.size(1), modalities=mp, device=dev, inverted=True).to(emb.dtype)
        logits = m.showo(inputs_embeds=emb, attention_mask=mask)["logits"][0, -1].float()
        probs = torch.softmax(logits, -1)
        p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
        text = tok.decode([int(probs.argmax())])
        z = p_yes + p_no
        return {"answer": yesno_from_text(text), "p_yes": p_yes / z if z else None, "p_no": p_no / z if z else None, "mass": z, "raw": text}
