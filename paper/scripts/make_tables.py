import json, re, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CI = os.path.join(D, "data", "ci"); T = os.path.join(D, "tables"); os.makedirs(T, exist_ok=True)
ld = lambda f: json.load(open(os.path.join(CI, f)))
NAME = {"qwen25vl7b": "Qwen2.5-VL-7B$^{\\dagger}$", "omniverifier7b": "OmniVerifier-7B$^{\\ddagger}$", "bagel": "BAGEL-7B-MoT", "omnigen2": "OmniGen2", "janus1b": "Janus-Pro-1B", "showo2_1.5b": "Show-o2-1.5B",
        "qwen25vl3b": "Qwen2.5-VL-3B$^{\\dagger}$", "qwen25vl3b_gsys": "\\quad OmniGen2 prompt$^{\\dagger}$", "internvl3_2b": "InternVL3-2B$^{\\dagger}$", "internvl3_8b": "InternVL3-8B$^{\\dagger}$",
        "janus7b": "Janus-Pro-7B", "lumina": "Lumina-DiMOO", "blip3o": "BLIP3-o-8B", "showo2_7b": "Show-o2-7B"}
sg = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.2f}"
ci = lambda d: f"[{sg(d['ci95'][0])}, {sg(d['ci95'][1])}]"
sg3 = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"


def did_rows(L, common=False):
    d = ld(f"did_{'common_' if common else ''}{L}.json")
    order = sorted((j for j in d if j in NAME), key=lambda j: d[j]["framing"]["est"])
    rows = []
    for j in order:
        r = d[j]; n = r["_n"]
        lo90 = r["selfspec"]["ci90"][0]; hi90 = r["selfspec"]["ci90"][1]
        mark = "\\textsuperscript{eq}" if (lo90 > -0.10 and hi90 < 0.10) else ("\\textsuperscript{ni}" if lo90 > -0.10 else "")
        rows.append(f"{NAME[j]} & {r['own_fresh']['est']:.3f} & {r['oth_fresh']['est']:.3f} & {r['own_ctx']['est']:.3f} & {r['oth_ctx']['est']:.3f} & "
                    f"{sg(r['framing']['est'])} {ci(r['framing'])} & {sg(r['self_fresh']['est'])} {ci(r['self_fresh'])} & {sg(r['selfspec']['est'])} {ci(r['selfspec'])}{mark} & "
                    f"{n.get('own_fresh', 0)} & {n.get('oth_fresh', 0)} \\\\")
    return rows, d


rows, d_h = did_rows("human"); open(os.path.join(T, f"tab_did_human.tex"), "w").write("\n".join(rows) + "\n")
for L in ("strict", "vlm", "hybrid"):
    rows, _ = did_rows(L); open(os.path.join(T, f"tab_did_{L}.tex"), "w").write("\n".join(rows) + "\n")
com = {L: ld(f"did_common_{L}.json") for L in ("human", "strict", "vlm")}
order = sorted((j for j in com["human"] if j in NAME), key=lambda j: com["human"][j]["framing"]["est"])
open(os.path.join(T, f"tab_framing_common.tex"), "w").write("\n".join(
    f"{NAME[j]} & " + " & ".join(f"{sg(com[L][j]['framing']['est'])} {ci(com[L][j]['framing'])}" for L in ("human", "strict", "vlm"))
    + f" & {com['human'][j]['_n'].get('oth_fresh', 0)} \\\\" for j in order) + "\n")

DEC = ["janus7b", "showo2_7b", "bagel", "omnigen2", "janus1b", "showo2_1.5b", "qwen25vl3b", "qwen25vl3b_gsys", "qwen25vl7b", "internvl3_2b", "internvl3_8b", "omniverifier7b"]
DEC = [j for j in DEC if os.path.exists(os.path.join(CI, f"decomp_{j}_human.json"))]
CON = [("ctx-assistant_nocaption", "authorship, no request"), ("ctx", "closed loop"), ("ctx-user_prompt", "request, no authorship"), ("ctx-caption_inline", "inline request")]
rows = []
for j in DEC:
    d = ld(f"decomp_{j}_human.json"); cells = []
    for c, _ in CON:
        if f"d_oth_{c}" in d: cells.append(f"{sg(d[f'd_oth_{c}']['est'])} {ci(d[f'd_oth_{c}'])}")
        else: cells.append("= closed loop" if j == "bagel" and c == "ctx-user_prompt" else "--")
    own = [sg(d[f"d_own_{c}"]["est"]) if f"d_own_{c}" in d else "--" for c, _ in CON]
    rows.append(f"{NAME[j]} & {d['oth_fresh']['est']:.2f} & " + " & ".join(cells) + f" & {d['_n'].get('oth_fresh_viol', 0)} \\\\")
    if "own_fresh" in d: rows.append(f"\\quad own images & {d['own_fresh']['est']:.2f} & " + " & ".join(own) + f" & {d['_n'].get('own_fresh_viol', 0)} \\\\")
open(os.path.join(T, f"tab_decomp.tex"), "w").write("\n".join(rows) + "\n")

CT = ["bagel", "omnigen2", "janus7b", "showo2_7b", "qwen25vl3b", "qwen25vl3b_gsys", "qwen25vl7b", "internvl3_2b", "internvl3_8b", "omniverifier7b"]
CT = [j for j in CT if os.path.exists(os.path.join(CI, f"ctrl_{j}_human.json")) or os.path.exists(os.path.join(CI, f"decomp_{j}_human.json"))]
rows = []
for j in CT:
    d = ld(f"ctrl_{j}_human.json") if os.path.exists(os.path.join(CI, f"ctrl_{j}_human.json")) else ld(f"decomp_{j}_human.json")
    cell = lambda c: f"{sg(d[f'd_oth_{c}']['est'])} {ci(d[f'd_oth_{c}'])}"
    fa = lambda c: f"{d[f'fa_oth_{c}']['est']:.3f}"
    rows.append(f"{NAME[j]} & {d['oth_fresh']['est']:.2f} & {cell('ctx')} & {cell('ctx-assistant_mismatch')} & {cell('ctx-assistant_cf')} & "
                f"{fa('fresh')} & {fa('ctx')} & {fa('ctx-assistant_mismatch')} & {fa('ctx-assistant_cf')} & {d['_n'].get('oth_fresh_viol', 0)} \\\\")
open(os.path.join(T, f"tab_controls.tex"), "w").write("\n".join(rows) + "\n")
print("tables written:", sorted(f for f in os.listdir(T) if f.startswith("tab_")))

bt = ld("bytype_human.json"); TY = {"presence": "presence", "count": "count", "color": "colour", "position": "position"}
rows = []
for j in ["bagel", "omnigen2", "janus1b", "showo2_1.5b", "janus7b", "lumina", "blip3o", "showo2_7b"]:
    first = True
    for t in TY:
        k = f"{j}:{t}"
        if k not in bt: continue
        r = bt[k]; sig = lambda d: "$^{*}$" if (d["ci95"][0] >= 0.005 or d["ci95"][1] <= -0.005) else ""
        rows.append(f"{NAME[j] if first else ''} & {TY[t]} & {r['_n']['own_fresh']} & {r['own_fresh']['est']:.2f} & {r['oth_fresh']['est']:.2f} & "
                    f"{sg(r['self_fresh']['est'])} {ci(r['self_fresh'])}{sig(r['self_fresh'])} & {sg(r['selfspec']['est'])} {ci(r['selfspec'])}{sig(r['selfspec'])} \\\\")
        first = False
open(os.path.join(T, "tab_bytype.tex"), "w").write("\n".join(rows) + "\n")

au = ld("auroc_human.json"); mf = ld("matched_fa.json"); rows = []
for j in ["bagel", "omnigen2", "janus1b", "showo2_1.5b", "janus7b", "lumina", "blip3o", "showo2_7b"]:
    a = au[f"{j}:oth"]; m = mf[j]
    rec = (m["det_closed_matched"] - m["det_closed"]) / (m["det_fresh"] - m["det_closed"]) if m["det_fresh"] - m["det_closed"] > 0.05 and j != "lumina" else float("nan")
    rows.append(f"{NAME[j]} & {a['fresh']['auc']:.3f} & {a['ctx']['auc']:.3f} & {sg3(a['ctx']['d_vs_fresh'])} & "
                f"{a['fresh']['pauc']:.3f} & {a['ctx']['pauc']:.3f} & {sg3(a['ctx']['dp_vs_fresh'])} [{sg3(a['ctx']['dp_ci95'][0])}, {sg3(a['ctx']['dp_ci95'][1])}] & "
                f"{m['det_fresh']:.2f} & {m['det_closed']:.2f} & " + ("--" if j == "lumina" else f"{m['det_closed_matched']:.2f}") + " & " + ("--" if rec != rec else f"{100*rec:.0f}\\%") + " \\\\")
open(os.path.join(T, "tab_auroc.tex"), "w").write("\n".join(rows) + "\n")

rc = ld("reward_continuous.json"); rows = []
GN = {"bagel": "BAGEL", "janus7b": "Janus-Pro-7B", "showo2_7b": "Show-o2-7B"}
for j in ["bagel", "janus7b", "showo2_7b", "blip3o"]:
    cells = []
    for g in ["bagel", "janus7b", "showo2_7b"]:
        r = rc.get(f"{j}:{g}")
        cells.append("--" if r is None else f"{r['fresh']:.2f} / {r['closed']:.2f}")
    rows.append(f"{NAME[j]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(T, "tab_rewardcont.tex"), "w").write("\n".join(rows) + "\n")
print("extra tables written")

sd = {L: ld(f"stdid_{L}.json") for L in ("human", "hybrid", "strict", "vlm")}
def cell(d):
    lo90, hi90 = d["ci90"]; m = "\\textsuperscript{eq}" if (lo90 > -0.10 and hi90 < 0.10) else ("\\textsuperscript{ni}" if lo90 > -0.10 else "")
    return f"{sg(d['est'])} {ci(d)}{m}"
order = sorted((j for j in sd["human"] if j in NAME), key=lambda j: sd["human"][j]["closedloop_std"]["est"])
open(os.path.join(T, "tab_stdid.tex"), "w").write("\n".join(f"{NAME[j]} & " + " & ".join(cell(sd[L][j]["selfspec_std"]) if j in sd[L] else "--" for L in ("human", "hybrid", "strict", "vlm")) + " \\\\" for j in order) + "\n")
print("stdid table written")

ls = ld("label_summary.json"); rows = []
for g in ["janus7b", "showo2_7b", "bagel", "lumina", "blip3o", "omnigen2"]:
    r = ls[g]; ps = [r["pass1"], r["blind"], r["blind2"]]
    v = sum(p.get("violated", 0) for p in ps); s_ = sum(p.get("satisfied", 0) for p in ps); a = sum(p.get("ambiguous", 0) for p in ps)
    cov = sum(p["labelled_in_flags"] for p in ps) / r["flag_strict"]
    prec = lambda p: f"{p.get('violated', 0) / p['labelled']:.2f}" if p["labelled"] else "--"
    rows.append(f"{NAME[g]} & {r['flag_strict']} & " + " & ".join(str(p["labelled"]) for p in ps) + f" & {100*cov:.0f}\\% & {v} & {s_} & {a} & " + " & ".join(prec(p) for p in ps) + " \\\\")
open(os.path.join(T, "tab_human.tex"), "w").write("\n".join(rows) + "\n")
print("human table written")

mh, ms = ld("matched_human.json"), ld("matched_strict.json"); dh = ld("did_human.json")
rows = []
for j in sorted((j for j in mh if j in NAME), key=lambda j: dh[j]["framing"]["est"]):
    def part(m):
        r = m[j]; ok = r["relni"]["ci90"][0] > 0
        c90 = r["self_fresh"]["ci90"]; tm = "eq" if c90[0] > -0.10 and c90[1] < 0.10 else ("ni" if c90[0] > -0.10 else "")
        return (f"{r['_facts']} & {sg(r['framing']['est'])} & {sg(r['self_fresh']['est'])}" + (f"\\textsuperscript{{{tm}}}" if tm else "")
                + f" & {sg(r['selfspec']['est'])} {ci(r['selfspec'])} & " + ("\\checkmark" if ok else "--"))
    rows.append(f"{NAME[j]} & {part(mh)} & {part(ms)} \\\\")
open(os.path.join(T, "tab_matched.tex"), "w").write("\n".join(rows) + "\n")
print("matched table written")

au = ld("audit.json"); pc = lambda k, n: f"{k}/{n}"; w = lambda lo, hi: f"[{100*lo:.0f}, {100*hi:.0f}]"
rows = []
for t, nm in [("presence", "presence"), ("count", "count"), ("color", "colour"), ("position", "position"), ("all", "all")]:
    k, n, lo, hi, p = au[f"S_miss_{t}"]
    rows.append(f"Satisfied & {nm} & {pc(k, n)} & {100*p:.1f} {w(lo, hi)} \\\\")
k, n, (p, lo, hi) = au["C_confirm"]; rows.append(f"Count violation & count & {pc(k, n)} & {100*p:.1f} {w(lo, hi)} \\\\")
a, n = au["R_fact_agreement"]; z = 1.96; ph = a / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
rows.append(f"Repair endpoint & all & {pc(a, n)} & {100*ph:.1f} {w(c - h, c + h)} \\\\")
open(os.path.join(T, "tab_audit.tex"), "w").write("\n".join(rows) + "\n")
print("audit table written")

mk = ld("mask_probe.json"); cols = [f"{j}:{w}" for j in ("bagel", "janus1b", "janus7b", "omnigen2") for w in ("own", "others")]
conds = [("Fresh (no request)", "fresh|none|all"), ("Closed loop, no mask", "closed|none|all"),
         ("\\quad all later tokens", "closed|all|all"), ("\\quad tokens after the image", "closed|post_image|all"),
         ("\\quad image tokens only", "closed|image_only|all"), ("\\quad answer position only", "closed|answer|all"),
         ("\\quad all later tokens, Q1 only", "closed|all|Q1"), ("\\quad all later tokens, Q2 only", "closed|all|Q2"),
         ("\\quad all later tokens, Q3 only", "closed|all|Q3"), ("\\quad all later tokens, Q4 only", "closed|all|Q4"),
         ("Inline request, no mask", "inline|none|all"), ("\\quad all later tokens", "inline|all|all"),
         ("\\quad answer position only", "inline|answer|all"),
         ("\\quad all later tokens, Q2 only", "inline|all|Q2"), ("\\quad all later tokens, Q3 only", "inline|all|Q3")]
def mcell(c, k):
    if k not in mk[c]["cond"]: return "--"
    t = mk[c]["cond"][k]
    if "|none|" in k: return f"{t['det']:.2f}"
    r = round(100 * t['recovered']); return f"{t['det']:.2f} ({'$-$' if r < 0 else ''}{abs(r)})"
rows = []
for lab, k in conds:
    if k == "inline|none|all": rows.append("\\midrule")
    rows.append(f"{lab} & " + " & ".join(mcell(c, k) for c in cols) + " \\\\")
rows.append("\\midrule")
rows.append("Violated / satisfied facts & " + " & ".join(f"{mk[c]['n_viol']} / {mk[c]['n_sat']}" for c in cols) + " \\\\")
rows.append("Max.\\ FA, masked & " + " & ".join(f"{max(mk[c]['cond'][kk]['fa'] for _, kk in conds if kk in mk[c]['cond'] and '|none|' not in kk and not kk.startswith('fresh')):.3f}" for c in cols) + " \\\\")
open(os.path.join(T, "tab_mask.tex"), "w").write("\n".join(rows) + "\n")
print("mask table written")

rows = []
for j in DEC:
    d = ld(f"decomp_{j}_human.json")
    cells = [f"{d['fa_oth_fresh']['est']:.3f}"] + [f"{d[f'fa_oth_{c}']['est']:.3f}" if f"fa_oth_{c}" in d else ("= closed" if j == "bagel" else "--") for c, _ in CON]
    rows.append(f"{NAME[j]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(T, "tab_fadecomp.tex"), "w").write("\n".join(rows) + "\n")
print("fa decomposition table written")

pb = ld("pub_protocols.json")
PROTO = [("atomic_fresh", "GenEval facts, fresh", "no"), ("atomic_cl", "GenEval facts, closed loop", "yes"),
         ("ug_out", "UniGen outcome verification", "yes"), ("uc", "UniCorn 0--10 rubric", "yes"),
         ("ov", "OmniVerifier question", "yes"), ("cotv", "UniGen CoT-V", "yes"),
         ("dq_cl", "Own decomposition, closed loop", "yes"), ("dq_req", "Own decomposition, request prefix", "yes"),
         ("dq_fresh", "Own decomposition, fresh", "no"), ("dq_ext", "Oracle decomposition, fresh", "no"),
         ("dq_cl_ent", "Own, entailed only, closed loop", "yes"), ("dq_req_ent", "Own, entailed only, request prefix", "yes"),
         ("dq_fresh_ent", "Own, entailed only, fresh", "no"), ("dq_ext_ent", "Oracle, entailed only, fresh", "no"),
         ("dq_cl_sent", "Own, self-checked, closed loop", "yes"), ("dq_fresh_sent", "Own, self-checked, fresh", "no")]
rows = []
for k, lab, rq in PROTO:
    if not any(k in pb[j]["protocols"] for j in ("bagel", "janus7b", "omnigen2")): continue
    if k in ("ug_out", "dq_cl", "dq_cl_ent", "dq_cl_sent"): rows.append("\\midrule")
    cells = []
    for j in ("bagel", "janus7b", "omnigen2"):
        q = pb[j]["protocols"].get(k)
        if q is None: cells.append("-- & -- & -- & --"); continue
        au = "--" if k == "ov" else f"{q['auroc']:.2f}"
        d5 = "--" if q["det_fa5"] != q["det_fa5"] else f"{q['det_fa5']:.2f}"
        cells.append(f"{q['det']:.2f} & {q['fa']:.2f} & {d5} & {au}")
    rows.append(f"{lab} & {rq} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
rows.append("Images (flawed) & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{pb[j]['n_images']} ({pb[j]['n_flawed']})}}" for j in ("bagel", "janus7b", "omnigen2")) + " \\\\")
open(os.path.join(T, "tab_pub.tex"), "w").write("\n".join(rows) + "\n")
print("published-protocol table written")

ov = ld("ov_trained.json"); rows = []
for k, lab in [("native", "Native protocol"), ("closed", "GenEval facts, closed loop"), ("fresh", "GenEval facts, fresh")]:
    cells = [f"{ov[g][k]['det']:.2f} & {ov[g][k]['fa']:.2f}" for g in ("bagel", "janus7b", "omnigen2")]
    q = ov["all"][k]; extra = "-- & --" if q["auroc"] is None else f"{q['det_fa5']:.2f} & {q['auroc']:.3f}"
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {q['det']:.2f} [{q['det_ci'][0]:.2f}, {q['det_ci'][1]:.2f}] & {q['fa']:.3f} & {extra} \\\\")
rows.append("\\midrule")
rows.append("Images (flawed) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ov[g]['n_images']} ({ov[g]['n_flawed']})}}" for g in ("bagel", "janus7b", "omnigen2")) + f" & \\multicolumn{{4}}{{c}}{{{ov['all']['n_images']} ({ov['all']['n_flawed']})}} \\\\")
open(os.path.join(T, "tab_ovtrained.tex"), "w").write("\n".join(rows) + "\n")
print("trained-verifier table written")

rp = ld("reason_probe.json"); rows = []
for j in ("bagel", "janus7b", "omnigen2"):
    for w in ("own", "others"):
        r = rp.get(f"{j}:{w}")
        if r is None: continue
        cells = []
        for m in ("first_token", "reason"):
            q = r[m]; cells.append(f"{q['det_fresh']:.2f} & {q['det_closed']:.2f} & ${q['change']:+.2f}$ $[{q['change_ci'][0]:+.2f}, {q['change_ci'][1]:+.2f}]$")
        q = r["reason"]
        rows.append(f"{NAME[j] if w == 'own' else ''} & {w} & {r['n_viol']} & " + " & ".join(cells) + f" & {q['fa_fresh']:.2f} & {q['fa_closed']:.2f} \\\\")
open(os.path.join(T, "tab_reason.tex"), "w").write("\n".join(rows).replace("+0.", "0.").replace("$0.", "$+0.") + "\n")
print("reasoning table written")

rows = []
for j, oth in (("bagel", "Janus-Pro-7B, Show-o2-7B"), ("janus7b", "Show-o2-7B")):
    d = ld(f"instr_{j}.json")
    for side, lab in (("own", "own"), ("oth", oth)):
        if f"{side}_fresh" not in d: continue
        c = lambda k: f"{sg(d[f'd_{side}_{k}']['est'])} {ci(d[f'd_{side}_{k}'])}"
        rows.append(f"{NAME[j] if side == 'own' else ''} & {lab} & {d[f'{side}_fresh']['est']:.2f} & {c('ctx')} & {c('ctx-assistant_instr')} & "
                    f"{d[f'fa_{side}_ctx-assistant_instr']['est']:.3f} & {d['_n'].get(f'{side}_fresh_viol', 0)} \\\\")
open(os.path.join(T, "tab_instr.tex"), "w").write("\n".join(rows) + "\n")
print("instruction table written")

dp = ld("dpo_eval.json"); rows = []
for k, lab, tr in [("base", "Janus-Pro-1B (no training)", "--"), ("fresh", "fresh-verifier pairs", "0"), ("fresh_s1", "fresh-verifier pairs", "1"),
                   ("closed", "closed-loop pairs", "0"), ("closed_s1", "closed-loop pairs", "1"), ("fresh_sub", "fresh pairs, subsampled", "0")]:
    q = dp[k]; d = dp.get(f"{k}-base")
    s3 = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"
    dd = "--" if d is None else f"{s3(d['est'])} [{s3(d['ci95'][0])}, {s3(d['ci95'][1])}]"
    rows.append(f"{lab} & {tr} & {q['all_ok']:.3f} [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}] & {q['fact_acc']:.3f} & {q['by_tag']['counting']:.2f} & {q['by_tag']['color_attr']:.2f} & {dd} \\\\")
open(os.path.join(T, "tab_dpo.tex"), "w").write("\n".join(rows) + "\n")
print("dpo table written")

ex = ld("exhaustive.json")["did"]; rows = []
js = [j for j in NAME if f"exhaustive:{j}" in ex]
js = sorted((j for j in js if j != "qwen25vl3b_gsys"), key=lambda j: ex[f"exhaustive:{j}"]["closed_oth"]["est"])
if "qwen25vl3b_gsys" in NAME and "qwen25vl3b" in js: js.insert(js.index("qwen25vl3b") + 1, "qwen25vl3b_gsys")
for j in js:
    e, m, u = ex[f"exhaustive:{j}"], ex[f"main_same_images:{j}"], ex[f"oracle_unselected:{j}"]
    ss = f"{sg(e['selfspec']['est'])} {ci(e['selfspec'])}" if "selfspec" in e else "--"
    rows.append(f"{NAME[j]} & {e['_n']['oth_fresh']} & {e['oth_fresh']['est']:.2f} & {e['oth_ctx']['est']:.2f} & {sg(e['closed_oth']['est'])} {ci(e['closed_oth'])} & "
                f"{sg(m['closed_oth']['est'])} {ci(m['closed_oth'])} & {u['_n']['oth_fresh']} & {u['oth_fresh']['est']:.2f} & {sg(u['closed_oth']['est'])} & {ss} \\\\")
open(os.path.join(T, "tab_exh.tex"), "w").write("\n".join(rows) + "\n")
print("exhaustive table written")

tp = ld("theory_probe.json"); dh = ld("did_human.json"); rows = []
f2 = lambda v: f"{v:.2f}"
neg = lambda v: ("$-$" if v < 0 else "") + f"{abs(v):.2f}"
pct = lambda d: f"{d['est']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
for j in sorted((j for j in dh if j in NAME and f"{j}:ctx" in tp), key=lambda j: dh[j]["framing"]["est"]):
    q = tp[f"{j}:ctx"]; lam, b = -q["slope"]["est"], q["icpt"]["est"]
    rows.append(f"{NAME[j]} & {f2(lam)} & {sg(b)} & {f2(b / (1 - lam))} & {sg(q['d_viol']['est'])} & {sg(q['d_sat']['est'])} & "
                f"{f2(q['det_fresh']['est'] - q['det_c']['est'])} & {neg(q['share']['est'])} & {pct(q['share_pool'])} & {q['det_err_heldout']['est']:.2f} \\\\")
open(os.path.join(T, "tab_pool.tex"), "w").write("\n".join(rows) + "\n")
CL = [("ctx", "closed loop"), ("ctx-assistant_nocaption", "authorship, no request"), ("ctx-user_prompt", "request, no authorship"),
      ("ctx-assistant_mismatch", "mismatched request"), ("ctx-assistant_cf", "contradicting request")]
order = [j for j in ("bagel", "janus1b", "omnigen2", "showo2_1.5b", "janus7b", "blip3o", "lumina", "showo2_7b", "qwen25vl3b", "qwen25vl3b_gsys",
                     "internvl3_2b", "internvl3_8b", "qwen25vl7b", "omniverifier7b") if j in NAME]
rows = []
for j in order:
    first = True
    for c, lab in CL:
        if f"{j}:{c}" not in tp: continue
        q = tp[f"{j}:{c}"]; lam, b = -q["slope"]["est"], q["icpt"]["est"]
        rows.append(f"{NAME[j] if first else ''} & {lab} & {q['_n']['viol']} & {neg(lam) + ('$^{\\ast}$' if lam < 0 else '')} & {sg(b)} & {sg(b / (1 - lam))} & "
                    f"{q['det_fresh']['est']:.2f} & {q['det_c']['est']:.2f} & {q['det_pool']['est']:.2f} & {q['fa_fresh']['est']:.3f} & {q['fa_c']['est']:.3f} & {q['fa_pool']['est']:.3f} \\\\")
        first = False
    rows.append("\\addlinespace[2pt]")
open(os.path.join(T, "tab_poolfull.tex"), "w").write("\n".join(rows[:-1]) + "\n")
print("pool tables written")

def _rows(f): return [l for l in open(os.path.join(T, f)).read().split("\n") if l.strip()]
mark = lambda l: "dagger" in l
open(os.path.join(T, "tab_decomp_main.tex"), "w").write("\n".join(l for l in _rows("tab_decomp.tex") if not mark(l)) + "\n")
open(os.path.join(T, "tab_controls_main.tex"), "w").write("\n".join(l for l in _rows("tab_controls.tex") if not mark(l)) + "\n")
cell = lambda l: [c.strip().rstrip("\\").strip() for c in l.split("&")]
dec = {cell(l)[0]: cell(l) for l in _rows("tab_decomp.tex") if mark(l)}
con = {cell(l)[0]: cell(l) for l in _rows("tab_controls.tex") if mark(l)}
rows = []
for k in dec:
    d, c = dec[k], con[k]
    rows.append(f"{k} & {d[1]} & {d[3]} & {d[5]} & {c[3]} & {c[4]} & {c[6]} & {d[6]} \\\\")
open(os.path.join(T, "tab_nonunified.tex"), "w").write("\n".join(rows) + "\n")
keep = ("GenEval facts, fresh", "GenEval facts, closed loop", "UniGen", "UniCorn", "OmniVerifier question", "Own, entailed only, closed loop", "Own, entailed only, fresh", "Images")
pr = []
for l in _rows("tab_pub.tex"):
    if l.startswith("\\midrule"):
        if pr and not pr[-1].startswith("\\midrule"): pr.append(l)
    elif l.startswith(keep): pr.append(l)
open(os.path.join(T, "tab_pub_main.tex"), "w").write("\n".join(pr) + "\n")
mk = []
for l in _rows("tab_mask.tex"):
    if l.startswith("\\midrule") and "Inline" not in "".join(mk[-1:]): continue
    if "Q1 only" in l or "Q2 only" in l or "Q3 only" in l or "Q4 only" in l: continue
    if l.startswith("Inline") : break
    mk.append(l)
mk += [l for l in _rows("tab_mask.tex") if l.startswith(("Violated", "Max."))]
open(os.path.join(T, "tab_mask_main.tex"), "w").write("\\midrule\n".join(["\n".join(mk[:-2]) + "\n", "\n".join(mk[-2:]) + "\n"]))
print("main-text table versions written")
rt = json.load(open(os.path.join(CI, "repair_table.json"))); al = json.load(open(os.path.join(CI, "always_baseline.json")))
def rc(g, n, acc=True):
    r = rt[g]["runs"][n]; return f"{r['all_ok']:.3f} ({r['fact_acc']:.3f})" if acc else f"{r['all_ok']:.3f}"
def st(g, acc=True): return f"{rt[g]['start_all_ok']:.3f} ({rt[g]['start_fact_acc']:.3f})" if acc else f"{rt[g]['start_all_ok']:.3f}"
rows = [f"GenEval facts & Janus-Pro-7B & {st('janus7b')} & {rc('janus7b', 'janus7b__janus7b__ctx')} & {rc('janus7b', 'janus7b__janus7b')} & {rc('janus7b', 'janus7b__showo2_7b__staged')} \\\\",
        f" & Show-o2-7B & {st('showo2_7b')} & {rc('showo2_7b', 'showo2_7b__showo2_7b__ctx')} & {rc('showo2_7b', 'showo2_7b__showo2_7b')} & {rc('showo2_7b', 'showo2_7b__janus7b__staged')} \\\\",
        f" & BAGEL & {st('bagel')} & {rc('bagel', 'bagel__bagel__ctx')} & {rc('bagel', 'bagel__bagel')} & -- \\\\",
        f" & BAGEL, original request & {st('bagel', False)} & {rc('bagel', 'bagel__bagel__ctx_orig', False)} & -- & -- \\\\",
        f" & BAGEL, matched threshold & {st('bagel', False)} & {rc('bagel', 'bagel__bagel__ctx_thr', False)} & -- & -- \\\\",
        f" & BAGEL, no verifier & {st('bagel')} & \\multicolumn{{3}}{{l}}{{{al['always_by_round']['3']:.3f} ({al['always_fact_acc_final']:.3f}) without any verifier}} \\\\",
        "\\midrule",
        f"Own, oracle filter & BAGEL & {st('bagel')} & {rc('bagel', 'bagel__bagel__ctx_dq')} & {rc('bagel', 'bagel__bagel_dq')} & -- \\\\",
        "\\midrule",
        f"Own, open filter (ERASE) & BAGEL & {st('bagel')} & {rc('bagel', 'bagel__bagel__ctx_dqt')} & {rc('bagel', 'bagel__bagel_dqt')} & -- \\\\",
        f" & OmniGen2 & {st('omnigen2')} & {rc('omnigen2', 'omnigen2__omnigen2__ctx_dqt')} & {rc('omnigen2', 'omnigen2__omnigen2_dqt')} & -- \\\\",
        f" & Janus-Pro-7B & {st('janus7b')} & {rc('janus7b', 'janus7b__janus7b__ctx_dqt')} & {rc('janus7b', 'janus7b__janus7b_dqt')} & -- \\\\"]
ok = lambda g, n: rt[g]["runs"][n]["all_ok"]
gaps = [ok("janus7b", "janus7b__janus7b") - ok("janus7b", "janus7b__janus7b__ctx"), ok("showo2_7b", "showo2_7b__showo2_7b") - ok("showo2_7b", "showo2_7b__showo2_7b__ctx"),
        ok("bagel", "bagel__bagel") - ok("bagel", "bagel__bagel__ctx"), ok("bagel", "bagel__bagel") - ok("bagel", "bagel__bagel__ctx_orig"),
        ok("bagel", "bagel__bagel") - ok("bagel", "bagel__bagel__ctx_thr"), ok("bagel", "bagel__bagel") - al["always_by_round"]["3"],
        ok("bagel", "bagel__bagel_dq") - ok("bagel", "bagel__bagel__ctx_dq"), ok("bagel", "bagel__bagel_dqt") - ok("bagel", "bagel__bagel__ctx_dqt"),
        ok("omnigen2", "omnigen2__omnigen2_dqt") - ok("omnigen2", "omnigen2__omnigen2__ctx_dqt"), ok("janus7b", "janus7b__janus7b_dqt") - ok("janus7b", "janus7b__janus7b__ctx_dqt")]
out, i = [], 0
for r in rows:
    if r.startswith("\\midrule"): out.append(r); continue
    body = r[:-2].rstrip()
    if "without any verifier" in body: body = body.replace("\\multicolumn{3}{l}{", "\\multicolumn{3}{c}{")
    out.append(f"({i + 1}) & {body} & $+${gaps[i]:.3f} \\\\"); i += 1
rows = out
open(os.path.join(T, "tab_repair.tex"), "w").write("\n".join(rows) + "\n")
sd = {r: json.load(open(os.path.join(CI, f"repair_seeds_{r}.json"))) for r in ("oracle", "agree", "either")}
rows = []
for k, lab in (("bagel_geneval", "BAGEL, GenEval facts"), ("bagel_erase", "BAGEL, ERASE loop"), ("omnigen2_erase", "OmniGen2, ERASE loop")):
    o = sd["oracle"][k]; P = o["pooled"]
    cells = [f"{o['per_seed'][s]['fresh']:.3f} / {o['per_seed'][s]['closed']:.3f}" for s in ("0", "1", "2")]
    g = lambda r: f"$+${sd[r][k]['pooled']['gap']:.3f}"
    rows.append(f"{lab} & " + " & ".join(cells) + f" & $+${P['gap']:.3f} [{P['ci95'][0]:.3f}, {P['ci95'][1]:.3f}] & {P['disc'][0]} / {P['disc'][1]} & {g('agree')} / {g('either')} \\\\")
open(os.path.join(T, "tab_seeds.tex"), "w").write("\n".join(rows) + "\n")
print("repair tables written")
vr = json.load(open(os.path.join(CI, "vae_readout.json")))
def vrow(lab, k):
    v = vr[k]; f = lambda x: f"{x[0]:.2f} / {x[1]:.2f}"
    return f"{lab} & {f(v['det_own'])} & {f(v['det_oth'])} & {v['fa_oth'][0]:.3f} / {v['fa_oth'][1]:.3f} \\\\"
vf = json.load(open(os.path.join(CI, "vae_format.json")))["rules"]["decoded"]
def frow(lab, f):
    g = lambda s: f"{vf[s][f + '_fresh']:.2f} / {vf[s][f + '_cl']:.2f}"
    return f"{lab} & {g('own_det')} & {g('oth_det')} & {vf['oth_fa'][f + '_fresh']:.3f} / {vf['oth_fa'][f + '_cl']:.3f} \\\\"
fmt_rows = ["\\addlinespace[2pt]", frow("ViT, ``Answer:'' prefilled", "vit"), frow("VAE + ViT, ``Answer:'' prefilled", "vae")]
open(os.path.join(T, "tab_bagelvae.tex"), "w").write("\n".join([vrow("ViT", "bagel:decoded"), vrow("VAE + ViT, decoded answer", "bagel_vae:decoded"),
                                                               vrow("VAE + ViT, $p(\\text{yes})<0.5$", "bagel_vae:pyes")] + fmt_rows) + "\n")
sp = json.load(open(os.path.join(CI, "state_probe.json")))
rows = []
for lab, c in (("Saved image, ViT tokens", "vit"), ("Saved image, VAE latents + ViT tokens", "vae"), ("Run's own latent + ViT tokens", "state"), ("Run's own latent only", "statevae")):
    d = lambda x: f"{sp['det'][x]['est']:.2f} [{sp['det'][x]['ci95'][0]:.2f}, {sp['det'][x]['ci95'][1]:.2f}]"
    rows.append(f"{lab} & {d(c + '_fresh')} & {d(c + '_cl')} & {sp['fa'][c + '_fresh']['est']:.3f} / {sp['fa'][c + '_cl']['est']:.3f} & {sp['mass'][c + '_fresh']:.2f} / {sp['mass'][c + '_cl']:.2f} \\\\")
open(os.path.join(T, "tab_state.tex"), "w").write("\n".join(rows) + "\n")
print("vae and state tables written")
e2 = json.load(open(os.path.join(CI, "e2e.json")))
def ecell(d):
    v = f"{d['fully_correct']:.3f}"; c = d.get("vs_ERASE")
    if c is None: return v
    g = "0.00" if abs(c["est"]) < 0.005 else f"{c['est']:+.2f}".replace("+", "$+$").replace("-", "$-$")
    return f"{v} ({g})"
rows = []
for k, lab in (("ERASE", "ERASE (own entailed questions, fresh)"), ("ERASE, no feedback", "ERASE, regenerating from the request"), ("ERASE, closed loop", "Same questions, closed loop"), ("UniGen", "UniGen outcome verification"), ("UniGen, matched threshold", "UniGen, rejections matched to ERASE"), ("UniCorn", "UniCorn rubric")):
    cells = []
    for g in ("bagel", "omnigen2"):
        r, b = e2[g]["repair"][k], e2[g]["bon3"][k]
        cells += [ecell(r), str(r["regenerations"]), "--" if k in ("ERASE, no feedback", "UniGen, matched threshold") else ecell(b)]
    rows.append(f"({len(rows) + 1}) & {lab} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
for k, lab in (("seed 0 (no selection)", "No selection (seed 0)"), ("oracle (any seed correct)", "Any of the three correct")):
    rows.append(f" & {lab} & -- & -- & {e2['bagel']['bon3'][k]['fully_correct']:.3f} & -- & -- & {e2['omnigen2']['bon3'][k]['fully_correct']:.3f} \\\\")
open(os.path.join(T, "tab_e2e.tex"), "w").write("\n".join(rows) + "\n")
print("e2e table written")
ri = json.load(open(os.path.join(CI, "reqinline.json")))
rows = []
for key, lab in (("showo2_7b:own", "Show-o2-7B, own"), ("showo2_7b:oth", "Show-o2-7B, others"), ("janus7b:own", "Janus-Pro-7B, own"),
                 ("janus7b:oth", "Janus-Pro-7B, others"), ("bagel:oth", "BAGEL-7B-MoT, others")):
    d = ri[key]["det"]; fa = ri[key].get("fa", {})
    rows.append(f"{lab} & {d['fresh']:.2f} & {d['ctx']:.2f} & {d['caption']:.2f} & {d['request']:.2f} & $+${d['request_minus_caption']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] & {fa.get('caption', 0):.3f} / {fa.get('request', 0):.3f} & {d['n']} \\\\")
open(os.path.join(T, "tab_reqinline.tex"), "w").write("\n".join(rows) + "\n")
print("reqinline table written")
pe = json.load(open(os.path.join(CI, "pool_extras.json")))["selfspec"]
sg = lambda x: f"{x:+.2f}".replace("+", "$+$").replace("-", "$-$")
rows = [f"{NAME[j]} & {v['n_own_viol']} & {sg(v['observed'])} [{sg(v['observed_ci95'][0])}, {sg(v['observed_ci95'][1])}] & {sg(v['predicted'])} & {sg(v['residual'])} [{sg(v['residual_ci95'][0])}, {sg(v['residual_ci95'][1])}] \\\\"
        for j, v in pe.items()]
open(os.path.join(T, "tab_poolres.tex"), "w").write("\n".join(rows) + "\n")
print("pool residual table written")
cb = {r: json.load(open(os.path.join(CI, f"calib_{r}.json"))) for r in ("oracle", "agree", "either")}
def dcell(d): return f"{sg(d['est'])} [{sg(d['ci95'][0])}, {sg(d['ci95'][1])}]"
rows = []
for k in cb["oracle"]:
    o = cb["oracle"][k]
    rows.append(f"{k} & {o['fresh']['fully_correct']:.3f} & {o['closed']['fully_correct']:.3f} & {o['closed, transferred threshold']['fully_correct']:.3f} & "
                f"{o['closed, transferred threshold']['first_rejected']} / {o['fresh']['first_rejected']} & {dcell(o['thr_vs_fresh'])} & "
                f"{sg(cb['agree'][k]['thr_vs_fresh']['est'])} / {sg(cb['either'][k]['thr_vs_fresh']['est'])} \\\\")
open(os.path.join(T, "tab_calib.tex"), "w").write("\n".join(rows) + "\n")
print("calib table written")

GS = [("janus7b", "Janus-Pro-7B", [("vit", "SigLIP (main)"), ("state", "run's tokens"), ("reenc", "re-encoded"),
                                    ("statevit", "run's tokens + SigLIP"), ("reencvit", "re-encoded + SigLIP")]),
      ("janus1b", "Janus-Pro-1B", [("vit", "SigLIP (main)"), ("state", "run's tokens"), ("reenc", "re-encoded"),
                                    ("statevit", "run's tokens + SigLIP"), ("reencvit", "re-encoded + SigLIP")]),
      ("showo2_1.5b", "Show-o2-1.5B", [("main", "re-encoded (main)"), ("state", "run's latents"), ("main", "re-encoded, decoded"), ("state", "run's latents, decoded")]),
      ("showo2_7b", "Show-o2-7B", [("main", "re-encoded (main)"), ("state", "run's latents"), ("main", "re-encoded, decoded"), ("state", "run's latents, decoded")])]
rows = []
for m, name, forms in GS:
    fn = os.path.join(CI, f"state_{m}.json")
    if not os.path.exists(fn): continue
    g = json.load(open(fn))
    ci = lambda e: f"{e['est']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]"
    sg = lambda e: f"{e['est']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]".replace("+", "$+$").replace("-", "$-$")
    for k, (f, lab) in enumerate(forms):
        o = g["rules"]["decoded" if lab.endswith("decoded") else "pyes"]
        ref = {"state": "reenc", "statevit": "reencvit"}.get(f) if m.startswith("janus") else ("main" if f == "state" else None)
        diff = "-- & --"
        if ref: diff = " & ".join(sg(o["diff"][f"{f} vs {ref}, {c}"]["det"]) for c in ("fresh", "cl"))
        head = name if k == 0 else (f"($n={g['n_viol']}$)" if k == 1 else "")
        rows.append(f"{head} & {lab} & {g['mass'][f + '_fresh']:.2f} / {g['mass'][f + '_cl']:.2f} & {g['auroc'][f + '_fresh']:.2f} & "
                    f"{o['det'][f + '_fresh']['est']:.2f} / {o['det'][f + '_cl']['est']:.2f} & "
                    f"{o['fa'][f + '_fresh']['est']:.3f} / {o['fa'][f + '_cl']['est']:.3f} & {diff} \\\\")
    rows.append("\\addlinespace[2pt]")
open(os.path.join(T, "tab_genstate.tex"), "w").write("\n".join(rows[:-1]) + "\n")
print("generation-state table written")

vs = json.load(open(os.path.join(CI, "variant_selfspec.json")))
VN = {"bagel": "BAGEL", "janus1b": "Janus-Pro-1B", "janus7b": "Janus-Pro-7B", "omnigen2": "OmniGen2", "showo2_1.5b": "Show-o2-1.5B", "showo2_7b": "Show-o2-7B"}
CN = {"ctx-assistant_nocaption": "authorship only", "ctx": "closed loop", "ctx-user_prompt": "request, no authorship",
      "ctx-caption_inline": "inline ``Caption:''", "ctx-request_inline": "inline ``Request:''", "ctx-assistant_instr": "with instruction"}
sgn = lambda e: f"{e['est']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]".replace("+", "$+$").replace("-", "$-$")
rows = []
for j in VN:
    first = True
    for c in CN:
        d = vs.get(f"{j}:{c}")
        if d is None: continue
        chg = lambda e: sgn(e).split(" [")[0]
        rows.append(f"{VN[j] if first else ''} & {CN[c]} & {d['n_own']} / {d['n_oth']} & {chg(d['own_change'])} / {chg(d['oth_change'])} & "
                    f"{sgn(d['raw'])} & {sgn(d['std'])} & {sgn(d['residual'])} \\\\")
        first = False
    rows.append("\\addlinespace[2pt]")
open(os.path.join(T, "tab_varself.tex"), "w").write("\n".join(rows[:-1]) + "\n")
print("variant self-specific table written")

def mark(ci90, lo=-0.10, hi=0.10, est=None, ci95=None):
    if est is not None and ci90[1] < 0:
        return f"{est:+.2f}".replace("+", "$+$").replace("-", "$-$") + ("$^{**}$" if ci95 is not None and ci95[1] < 0 else "$^{*}$")
    return "eq" if ci90[0] > lo and ci90[1] < hi else ("ni" if ci90[0] > lo else "unres.")
did_marks = {}
for l in open(os.path.join(T, "tab_did_human.tex")):
    name = l.split("&")[0].strip(); sup = re.search(r"\\textsuperscript\{(\w+)\}", l.split("&")[7])
    did_marks[name] = sup.group(1) if sup else "unres."
mh = json.load(open(os.path.join(CI, "matched_human.json"))); sdh = json.load(open(os.path.join(CI, "stdid_human.json")))
pex = json.load(open(os.path.join(CI, "pool_extras.json")))["selfspec"]; vss = json.load(open(os.path.join(CI, "variant_selfspec.json")))
st_b = json.load(open(os.path.join(CI, "state_probe.json")))["diff"]
st_s = {m: json.load(open(os.path.join(CI, f"state_{m}.json"))) for m in ("showo2_1.5b", "showo2_7b")}
VSHORT = {"ctx-caption_inline": "inline", "ctx-request_inline": "inline", "ctx-assistant_instr": "instruction", "ctx-user_prompt": "no authorship", "ctx": "closed loop"}
sgv = lambda x: f"{x:+.2f}".replace("+", "$+$").replace("-", "$-$")
STATUS = {"bagel": "none found", "janus1b": "none in closed loop; fresh unresolved", "omnigen2": "none in closed loop; fresh unresolved",
          "showo2_1.5b": "none found", "janus7b": "deficits in variants; one-sided deficit on matched errors",
          "lumina": "fresh deficit by type ($-$0.13)", "blip3o": "deficit under $p(\\text{yes})$; unresolved under decoded answers",
          "showo2_7b": "small deficits: own latents, pool residual ($-$0.09), exhaustive fresh term ($-$0.18)"}
rows = []
for j in ("bagel", "janus1b", "omnigen2", "showo2_1.5b", "janus7b", "lumina", "blip3o", "showo2_7b"):
    cl = did_marks[NAME[j]]
    std = mark(sdh[j]["selfspec_std"]["ci90"], est=sdh[j]["selfspec_std"]["est"], ci95=sdh[j]["selfspec_std"]["ci95"])
    mcl = mark(mh[j]["selfspec"]["ci90"], est=mh[j]["selfspec"]["est"], ci95=mh[j]["selfspec"]["ci95"]); mfr = mark(mh[j]["self_fresh"]["ci90"], est=mh[j]["self_fresh"]["est"], ci95=mh[j]["self_fresh"]["ci95"])
    o = pex[j]; py = sgv(o["observed"]) + "$^{**}$" if o["observed_ci95"][1] < 0 else "n.s."
    neg = list(dict.fromkeys(VSHORT[c.split(":")[1]] for c in vss if c.startswith(j + ":") and vss[c]["std"]["ci95"][1] < 0))
    var = ("none" if not neg else ", ".join(neg)) if any(c.startswith(j + ":") for c in vss) else "n/t"
    if j == "bagel":
        lat = "n.s." if all(st_b[k]["det"]["ci95"][1] >= 0 for k in ("state vs re-encoded, fresh", "state vs re-encoded, closed loop")) else "deficit"
    elif j in st_s:
        sig = [(r, c) for r in ("pyes", "decoded") for c in ("fresh", "cl") if st_s[j]["rules"][r]["diff"][f"state vs main, {c}"]["det"]["ci95"][1] < 0]
        lat = "n.s." if not sig else sgv(st_s[j]["rules"]["decoded"]["diff"]["state vs main, cl"]["det"]["est"]) + "$^{**}$ (decoded, closed loop)"
    else:
        lat = "no answer" if j.startswith("janus") else "n/t"
    rows.append(f"{NAME[j]} & {cl} & {std} & {mcl} & {mfr} & {py} & {var} & {lat} & {STATUS[j]} \\\\")
open(os.path.join(T, "tab_status.tex"), "w").write("\n".join(rows) + "\n")
print("status table written")
