# Closed-Loop Sycophancy

Code, labels, and analysis outputs for

> **Agreeable Rather Than Blind: Closed-Loop Sycophancy in Unified Multimodal Models**
> Diming Zhang. Submitted to the International Journal of Computer Vision (special issue on Multimodal LLMs for Unified
> Comprehension and Generation).

Unified multimodal models often verify their own generations with the generation request still in context. The paper
shows that, for images read back through the understanding encoder, the request rather than self-blindness causes most of
the failures this closed loop adds, and that request-free verification with entailed atomic questions (ERASE) avoids most
of them.

## Repository layout

| Path | Contents |
|---|---|
| `code/run_generate.py`, `code/run_judge.py` | Generate images for the GenEval prompts; ask every fact question of a judge in a chosen context (fresh, closed loop, and the request variants and controls) |
| `code/run_repair.py`, `code/run_repair_stage.py`, `code/score_repair.py` | Generate–verify–regenerate repair loops and their oracle/detector scoring |
| `code/mucg/` | Model adapters (BAGEL, Janus-Pro, Show-o2, BLIP3-o, OmniGen2, Lumina-DiMOO, Qwen2.5-VL / Qwen3-VL / InternVL3 / OmniVerifier judges, OWLv2 detector) with one `generate` / `judge` interface |
| `code/bootstrap_ci.py` | Label definitions and the prompt-clustered bootstrap behind the difference-in-differences terms (`did`, `stdid`, `matched`, `decomp`, `repair`, ...) |
| `code/*_probe.py`, `code/mask_probe*.py`, `code/probe_attention*.py` | Generation-state probes, attention masking, attention probe, reasoning probe |
| `code/pub_protocols.py`, `code/ov_native.py`, `code/entail_*.py`, `code/self_entail.py`, `code/ext_decomp.py` | Published verification protocols, the trained OmniVerifier-7B, question decomposition and entailment filtering |
| `code/*_analyze.py`, `code/pool_*.py`, `code/nolabel_fit.py`, `code/multiplicity.py`, `code/pooled_selfspec.py`, ... | Analyses that turn judge outputs into the JSON files in `paper/data/ci/` |
| `code/make_*_kit.py`, `code/score_audit.py`, `code/label_summary.py` | Blind labelling tool (inspection passes, audit, exhaustive labelling) and label summaries |
| `code/dpo/` | Self-reward DPO experiment with Janus-Pro-1B |
| `code/data/prompts.jsonl` | The 553 GenEval prompts and their 1,326 atomic yes/no facts (also `geneval_metadata.jsonl`, `prompts_dpg.jsonl`, `train_prompts.jsonl`) |
| `code/data/human/`, `code/data/audit/` | Human labels: inspection passes, second and third annotators (anonymised), blind audit, exhaustive labels |
| `code/data/pub_prompts/` | Verification prompts copied verbatim from the released UniCorn and OmniVerifier code, for `pub_protocols.py` |
| `paper/scripts/make_tables.py`, `paper/scripts/make_figures.py` | Regenerate every table body and figure of the paper from `paper/data/ci/` |
| `paper/data/ci/` | Bootstrap and analysis outputs behind every reported number; `SOURCES.md` maps values typed into the text to their files |

Comments and docstrings have been removed from the code; `python <script>.py -h` lists each script's arguments.

## Reproducing the tables and figures

This needs only Python 3.10+, NumPy, and Matplotlib:

```bash
python paper/scripts/make_tables.py    # writes paper/tables/tab_*.tex
python paper/scripts/make_figures.py   # writes paper/figures/Fig1-Fig5.pdf and FigS1.pdf
```

The outputs are identical to the tables and figures of the submitted manuscript.

## Re-running the analyses

The analyses read the per-fact judge outputs in `code/runs/judge/` (about 215 MB), the repair-loop traces in
`code/runs/repair/`, and the generated images in `code/runs/gen/`. These are too large for this repository and are
released separately (link to be added). With them in place, for example:

```bash
cd code
python bootstrap_ci.py did --labels human --json ../paper/data/ci/did_human.json
python bootstrap_ci.py matched --labels human --json ../paper/data/ci/matched_human.json
python pooled_selfspec.py --json ../paper/data/ci/pooled_selfspec.json
python multiplicity.py ../paper/data/ci --json ../paper/data/ci/multiplicity.json
```

`paper/data/ci/SOURCES.md` lists the script that produced each file.

## Re-running the experiments

Each model runs in its own environment (Python 3.10, PyTorch 2.8 with CUDA 12.8; one 46 GB GPU per model). Clone the
official repositories into `$MUCG_HOME/repos/` and install their requirements:

| Model | Repository (`$MUCG_HOME/repos/...`) | Checkpoint |
|---|---|---|
| BAGEL-7B-MoT | `Bagel` (ByteDance-Seed/Bagel) | `ByteDance-Seed/BAGEL-7B-MoT` |
| Janus-Pro-1B / 7B | `Janus` (deepseek-ai/Janus) | `deepseek-ai/Janus-Pro-1B`, `deepseek-ai/Janus-Pro-7B` |
| Show-o2-1.5B / 7B | `Show-o` (showlab/Show-o) | per the Show-o2 configs |
| BLIP3-o-8B | BLIP3o (JiuhaiChen/BLIP3o) | `BLIP3o/BLIP3o-Model-8B` |
| OmniGen2 | `OmniGen2` (VectorSpaceLab/OmniGen2) | `OmniGen2/OmniGen2` |
| Lumina-DiMOO | `Lumina-DiMOO` (Alpha-VLLM/Lumina-DiMOO) | `Alpha-VLLM/Lumina-DiMOO` |
| Oracle and control judges | Hugging Face Transformers | `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen2.5-VL-3B/7B-Instruct`, `OpenGVLab/InternVL3-2B/8B-hf`, `comin/OmniVerifier-7B`, `google/owlv2-large-patch14-ensemble` |

`MUCG_HOME` (default `~/mucg`) holds `repos/` and the Hugging Face cache `hf-cache/`. Typical entry points:

```bash
cd code
python run_generate.py --model bagel --seeds 0 1
python run_judge.py --judge bagel --gen janus7b                                # fresh context
python run_judge.py --judge bagel --gen janus7b --context                      # closed loop
python run_judge.py --judge janus7b --gen showo2_7b --context --context_mode request_inline
python run_repair.py --gen bagel --judge bagel --rounds 3 --tags position color_attr counting [--context] [--questions decomp]
```

## Third-party material

The GenEval prompts and facts are derived from GenEval (Ghosh et al., 2023), the DPG-Bench prompts from ELLA (Hu et al.,
2024); obtain the original DPG-Bench file from the ELLA repository. The prompts in `code/data/pub_prompts/` are copied from
the released code of UniCorn and OmniVerifier and remain under their original licences.

## Licence

- **Code** (`code/**/*.py`, `paper/scripts/`): MIT License, see [`LICENSE`](LICENSE).
- **Data and analysis outputs** (`code/data/human/`, `code/data/audit/`, `code/data/train_prompts.jsonl`,
  `paper/data/ci/`, `paper/figures/assets/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see
  [`LICENSE-DATA`](LICENSE-DATA).
- **Third-party material** keeps its original licence: the GenEval prompts underlying `code/data/prompts.jsonl` and
  `code/data/geneval_metadata.jsonl`, the DPG-Bench prompts underlying `code/data/prompts_dpg.jsonl`, and the
  verification prompts in `code/data/pub_prompts/`.

## Citation

To be added on publication.
