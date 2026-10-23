# Sources of values typed directly in main.tex (integrity audit v43)

| Manuscript item | Source file in data/ci | Produced by |
|---|---|---|
| Table reward (counted-verdict preference pairs) | reward_quality.txt | code/reward_quality.py |
| Table dpg and Sect. DPG prose (closed-loop and self-specific terms) | dpg_did_vlm.json / .txt | code/bootstrap_ci.py did --labels vlm --judge_dir runs/judge_dpg |
| Tables attention and layers | attention_summary.txt | code/summarize_probe.py on runs/probe/attention_*.jsonl |
| Fig. 1 example (fresh 0.10, generic 0.16, closed loop 0.65, inline 0.25) | fig1_example.json (all contexts, read by scripts/make_figures.py) | runs/judge/bagel__bagel[__ctx].jsonl, ge_0416_s1, fact ge_0416_f2 |
| Calibration and matched-UniGen thresholds | thresholds.json | round-0 verdicts in runs/repair/*/trace.jsonl |
| Table models (backbone sizes, resolutions, s/image) | -- | configuration from model cards and run logs, not a measured result |
| DPG prompt/question counts and the 48 checked flags | -- | runs/judge_dpg and the DPG flag sheet (not exported) |
| "245 first images ... 206", questions per prompt, first 445 prompts, annotator-C substitution, null-slope 0.35 | -- | in-session analysis outputs not exported; values unchanged since v35 |
| Table bagelvae, last two rows; Appendix C.1 format-controlled paragraph | vae_format.json | code/vae_format.py --prefix answer (runs/vae_format/bagel_answer.jsonl), code/vae_format_analyze.py |
| Table genstate, Appendix C.3, Sect. 4.1 "The generation state" | state_janus7b.json, state_janus1b.json, state_showo2_1.5b.json, state_showo2_7b.json | code/janus_state_probe.py, code/showo2_state_probe.py (runs/state_janus, runs/state_showo2), code/gstate_analyze.py |
| Sect. 4.4/4.5 label-free correction false alarms (0.007 to 0.036 vs 0.004 to 0.024) | pool_extras.json | code/pool_extras.py (correction: fa_fresh, fa_corrected) |
| Sect. 6.2 verifier-free loop per round (0.886 after one round) | always_baseline.json | always_by_round |
| Appendix app:fa (false alarms and re-rendered views), human-verified re-rendered views | perturb_check.json | code/perturb_check.py (human:closed+*) |
| Sect. 4.2 "Own images in every variant", Appendix Table varself | variant_selfspec.json | code/variant_selfspec.py |
| Appendix A.2 readout agreement on violated facts (97.1 to 100%, up to 0.03) | readout_viol.json | code/readout_viol.py |
| Table C3 decoded rows, Appendix C.3 decoded results | state_showo2_1.5b.json, state_showo2_7b.json (rules.decoded) | code/gstate_analyze.py |
| Appendix Table exceptions (tab:exceptions) | variant_selfspec.json, pool_extras.json (observed_ci95), state_showo2_7b.json, stdid_human.txt, matched_human.txt, exhaustive.json | values copied from the cited tables; code/variant_selfspec.py (duplicate-own-rows bug fixed 2026-10-22: estimates unchanged, n halved) |
| Appendix C.3 latent distances and verdict changes (run vs re-encoded; two sampled re-encodings) | latent_dist.json; verdict changes from runs/state_showo2/*_state.jsonl | code/latent_dist.py |
| Table status (per-judge self-blindness tests), Sect. 4.1 matched fresh self term | did_human (tab_did_human.tex marks), stdid_human.json, matched_human.json (self_fresh, selfspec ci90), pool_extras.json, variant_selfspec.json, state_probe.json, state_showo2_*.json | scripts/make_tables.py (status block) |
| Table C3 / B3 one-sided marks for the generation state | state_showo2_*.json (ci90 added 2026-10-23) | code/gstate_analyze.py |
| Table dpg BAGEL row (own images only, 0.68 -> 0.08) | dpg_did_vlm.json (bagel: own_fresh, own_ctx) | code/bootstrap_ci.py did --labels vlm --judge_dir runs/judge_dpg --partial (2026-10-23; Janus-Pro-7B and Show-o2-7B entries unchanged) |
| Suppl. app:dpg (DPG-Bench) "false-alarm rates stay below 0.05" | dpg_fa_vlm.json (max 0.048) | code/dpg_fa.py |
| Suppl. app:dpo (DPO) order-averaged difference 0.011 (0.000 to 0.021) | dpo_eval.json (fresh_2s-closed_2s) | code/dpo/eval_analyze.py |
| Appendix C.3 "72% of the tokens agree" (Janus-Pro VQ re-encoding) | janus_token_agree.json (pooled mean 0.718) | runs/state_janus/*_regen.jsonl (token_agree_reenc) |
| Appendix C.3 run-vs-re-encoded verdict changes (0.4 to 1.8%) | showo2_verdict_changes.json | runs/state_showo2/*_state.jsonl (main_* vs state_* verdicts) |
| Appendix C.3 "150" satisfied images per model | -- | code/showo2_state_probe.py and janus_state_probe.py default --sat 150 |
| Main-text Appendices A-E (text-only summaries, v61) | values copied from the supplementary sections and tables they cite (instr_*, reqinline, dpo_eval, reason_probe, dpg_did_vlm, dpg_fa_vlm, attention_summary.txt, reward_quality.txt, bestofn) | v60/v61 restructure (2026-10-23); seeds means 0.918/0.832 from repair_seeds_oracle.json (pooled) |

Layout (v62, 2026-10-23): table rows in paper/tables/tab_*.tex (scripts/make_tables.py), figures in paper/figures/ (scripts/make_figures.py; FigPool.pdf renamed Fig4.pdf, former Fig4.pdf renamed FigS1.pdf; source images in figures/assets/). Packages: python build/make_packages.py.
| Sect. 4.1 / B.1 / S2.1 cross-judge average self-specific term (+0.032; less affected -0.003) | pooled_selfspec.json | code/pooled_selfspec.py (one prompt bootstrap shared by all judges, B=2000) |
| Sect. 4.1 / S2.1 / S2.9 multiplicity over 341 secondary own-image terms (41 at p<0.05; Holm 4, BH 5) | multiplicity.json | code/multiplicity.py data/ci (normal approximation to each 95% bootstrap interval) |
| A.5 / S1.6 pooled three-seed test p<1e-6 | repair_seeds_oracle.json (pooled.p_cluster_signflip) | code/seeds_analyze.py (prompt-level sign-flip permutation, 1e6 draws) |
