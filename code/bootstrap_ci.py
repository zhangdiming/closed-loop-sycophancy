import argparse, collections, glob, json, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
B = 2000
GEN_OF = {"bagel_vae": "bagel"}


def load_rows(judge_dir):
    rows = []
    for f in glob.glob(str(pathlib.Path(judge_dir) / "*.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    return rows


def labels(rows, kind, oracle="qwen3vl8b"):
    typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == oracle}
    vlm = {(r["gen"], r["image"], r["fid"]): r["answer"] for r in rows if r["judge"] == oracle and not r.get("context")}
    det = {(r["gen"], r["image"], r["fid"]): r["answer"] for r in rows if r["judge"] == "owlv2"}
    o = {}
    for k, v in vlm.items():
        if kind == "vlm" or typ[k] == "color": o[k] = v
        elif k in det and det[k] == v: o[k] = v
    human_gens = set()
    if kind in ("human", "hybrid", "aikit"):
        for g in sorted({k[0] for k in vlm}):
            lp, fp = ROOT / "data/human" / f"{g}_labels.json", ROOT / "runs/human" / f"{g}_flagged.json"
            if not (lp.exists() and fp.exists()): continue
            human_gens.add(g)
            pairs = [(json.load(open(fp)), json.load(open(lp))["labels"])]
            kit = "data/ai_review" if kind == "aikit" else "data/human"
            tf, tl = ROOT / "runs/human" / f"{g}_todo_flagged.json", ROOT / kit / f"{g}_todo_labels.json"
            if tf.exists() and tl.exists():
                tlab = json.load(open(tl))["labels"]
                k3 = ROOT / "data/human" / f"{g}_kappa3_labels.json"
                if os.environ.get("LABEL_SUB") and kind != "aikit" and k3.exists(): tlab = {**tlab, **{i: l for i, l in json.load(open(k3))["labels"].items() if i in tlab}}
                pairs.append((json.load(open(tf)), tlab))
            t2 = ROOT / "data/human" / f"{g}_todo2_labels.json"
            if kind != "aikit" and tf.exists() and t2.exists(): pairs.append((json.load(open(tf)), json.load(open(t2))["labels"]))
            if kind in ("human", "aikit"):
                for k in [k for k in list(o) if k[0] == g and typ[k] != "count" and o[k] == "no"]: o.pop(k)
            for flagged, lab in pairs:
                for i, f in enumerate(flagged):
                    l = lab.get(str(i)); k = (g, f["image"], f["fid"])
                    if l == "violated": o[k] = "no"
                    elif l == "satisfied": o[k] = "yes"
                    elif l == "ambiguous": o.pop(k, None)
    return o, human_gens


def cond_of(r):
    if r.get("perturb"): return None
    if not r.get("context"): return "fresh"
    m = r.get("context_mode")
    return "ctx" if m in (None, "assistant") else f"ctx-{m}"


def answers(rows):
    ans = collections.defaultdict(dict)
    for r in rows:
        if r["judge"] in ("qwen3vl8b", "owlv2"): continue
        c = cond_of(r)
        if c is None: continue
        ans[(r["judge"], c, r["gen"])][(r["image"], r["fid"])] = (r["pid"], r["answer"])
    for (j, c, g) in [k for k in ans if k[1] == "fresh"]:
        if (j, "ctx", g) in ans:
            both = set(ans[(j, "fresh", g)]) & set(ans[(j, "ctx", g)])
            for cc in ("fresh", "ctx"): ans[(j, cc, g)] = {k: v for k, v in ans[(j, cc, g)].items() if k in both}
    return ans


class Cells:
    def __init__(self):
        self.h = collections.defaultdict(lambda: collections.defaultdict(int)); self.n = collections.defaultdict(lambda: collections.defaultdict(int))

    def add(self, cell, pid, hit):
        self.h[cell][pid] += int(hit); self.n[cell][pid] += 1

    def boot(self, exprs, seed=0):
        pids = sorted({p for c in self.n for p in self.n[c]}); idx = {p: i for i, p in enumerate(pids)}
        cells = sorted(self.n); H = np.zeros((len(pids), len(cells))); N = np.zeros_like(H)
        for j, c in enumerate(cells):
            for p, v in self.n[c].items(): N[idx[p], j] = v; H[idx[p], j] = self.h[c][p]
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(pids), np.ones(len(pids)) / len(pids), size=B).astype(float)
        point = {c: H[:, j].sum() / max(N[:, j].sum(), 1) for j, c in enumerate(cells)}
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = {c: (W @ H[:, j]) / (W @ N[:, j]) for j, c in enumerate(cells)}
        out = {}
        for name, f in exprs.items():
            try:
                pt = f(point); dist = f(bs)
            except KeyError:
                continue
            dist = dist[~np.isnan(dist)]
            out[name] = {"est": float(pt), "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                         "ci90": [float(np.percentile(dist, 5)), float(np.percentile(dist, 95))]}
        out["_n"] = {c: int(N[:, j].sum()) for j, c in enumerate(cells)}; out["_pids"] = len(pids)
        return out


def fmt(d, k, w=6):
    if k not in d: return f"{'-':>{w + 16}s}"
    e, (lo, hi) = d[k]["est"], d[k]["ci95"]
    return f"{e:+.2f} [{lo:+.2f},{hi:+.2f}]" if k.startswith(("framing", "self", "d_")) else f"{e:.3f} [{lo:.2f},{hi:.2f}]"


def did(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows)
    judges = sorted({j for (j, _, _) in ans}); res = {}
    if a.common: print(f"others restricted to the common set {a.common}")
    print(f"labels={a.labels} (human labels for: {sorted(hg) if a.labels == 'human' else '-'}); B={B}, prompt-clustered")
    print(f"{'judge':12s} {'own fresh':>22s} {'oth fresh':>22s} {'own closed':>22s} {'oth closed':>22s} {'framing':>22s} {'self(fresh)':>22s} {'self-specific':>22s}  90% CI selfspec   TOST(0.10)")
    for J in judges:
        c = Cells()
        for (j, cond, g), d in ans.items():
            if j != J or cond not in ("fresh", "ctx"): continue
            side = "own" if g == GEN_OF.get(J, J) else "oth"
            if side == "oth" and a.common and g not in a.common: continue
            for (im, fid), (pid, an) in d.items():
                if o.get((g, im, fid)) == "no": c.add(f"{side}_{cond}", pid, an == "no")
        ex = {"own_fresh": lambda p: p["own_fresh"], "oth_fresh": lambda p: p["oth_fresh"], "own_ctx": lambda p: p["own_ctx"],
              "oth_ctx": lambda p: p["oth_ctx"], "framing": lambda p: p["oth_ctx"] - p["oth_fresh"],
              "self_fresh": lambda p: p["own_fresh"] - p["oth_fresh"],
              "selfspec": lambda p: (p["own_ctx"] - p["own_fresh"]) - (p["oth_ctx"] - p["oth_fresh"]),
              "relni": lambda p: (p["own_ctx"] - p["own_fresh"]) - 1.5 * (p["oth_ctx"] - p["oth_fresh"])}
        r = c.boot(ex)
        if "selfspec" not in r:
            if a.partial and "own_fresh" in r and "own_ctx" in r:
                res[J] = r; print(f"{J:12s} own images only: " + " ".join(fmt(r, k) for k in ["own_fresh", "own_ctx"]))
            continue
        res[J] = r
        lo90, hi90 = r["selfspec"]["ci90"]; tost = "equivalent" if (lo90 > -0.10 and hi90 < 0.10) else "not shown"
        print(f"{J:12s} " + " ".join(fmt(r, k) for k in ["own_fresh", "oth_fresh", "own_ctx", "oth_ctx", "framing", "self_fresh", "selfspec"])
              + f"  [{lo90:+.2f},{hi90:+.2f}]  {tost}")
    return res


def matched(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows)
    judges = sorted({j for (j, _, _) in ans}); res = {}
    print(f"labels={a.labels}; matched (prompt, fact) violations; B={B}, prompt-clustered")
    for J in judges:
        own_g = GEN_OF.get(J, J); items = collections.defaultdict(list)
        gens = {g for (j, c, g) in ans if j == J}
        for g in gens:
            if g != own_g and a.common and g not in a.common: continue
            fr, cl = ans.get((J, "fresh", g), {}), ans.get((J, "ctx", g), {})
            side = "own" if g == own_g else "oth"
            for k in set(fr) & set(cl):
                if o.get((g,) + k) == "no":
                    pid = fr[k][0]; items[(side, pid, k[1])].append((fr[k][1] == "no", cl[k][1] == "no"))
        keys = {(p, f) for (s_, p, f) in items if s_ == "own"} & {(p, f) for (s_, p, f) in items if s_ == "oth"}
        c = Cells()
        for (p, f) in keys:
            for side in ("own", "oth"):
                for frn, cln in items[(side, p, f)]:
                    c.add(f"{side}_fresh", p, frn); c.add(f"{side}_ctx", p, cln)
        ex = {"own_fresh": lambda q: q["own_fresh"], "oth_fresh": lambda q: q["oth_fresh"], "own_ctx": lambda q: q["own_ctx"],
              "oth_ctx": lambda q: q["oth_ctx"], "framing": lambda q: q["oth_ctx"] - q["oth_fresh"],
              "self_fresh": lambda q: q["own_fresh"] - q["oth_fresh"],
              "selfspec": lambda q: (q["own_ctx"] - q["own_fresh"]) - (q["oth_ctx"] - q["oth_fresh"]),
              "relni": lambda q: (q["own_ctx"] - q["own_fresh"]) - 1.5 * (q["oth_ctx"] - q["oth_fresh"])}
        if not keys: continue
        r = c.boot(ex)
        if "selfspec" not in r: continue
        r["_facts"] = len(keys); res[J] = r
        print(f"{J:12s} facts={len(keys):4d} n_own={r['_n']['own_fresh']:4d} n_oth={r['_n']['oth_fresh']:5d} "
              + " ".join(fmt(r, k) for k in ["framing", "self_fresh", "selfspec"]) + f"  90% [{r['selfspec']['ci90'][0]:+.3f},{r['selfspec']['ci90'][1]:+.3f}]")
    return res


def decomp(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows); J = a.judge
    gens = [J] + [g for g in a.others if g != J]
    conds = [c for c in a.conds if any((J, c, g) in ans for g in gens)]
    print(f"labels={a.labels}; judge={J}; own={J}; others={gens[1:]}; conditions={conds}; matched across conditions")
    c = Cells()
    for g in gens:
        side = "own" if g == GEN_OF.get(J, J) else "oth"
        avail = [cc for cc in conds if (J, cc, g) in ans]
        if len(avail) < len(conds):
            print(f"  note: {g} skipped, lacks {sorted(set(conds) - set(avail))}"); continue
        keys = set.intersection(*[set(ans[(J, cc, g)]) for cc in avail])
        for k in keys:
            lab = o.get((g,) + k)
            if lab not in ("yes", "no"): continue
            pid = ans[(J, avail[0], g)][k][0]
            for cc in avail:
                c.add(f"{side}_{cc}_{'viol' if lab == 'no' else 'sat'}", pid, ans[(J, cc, g)][k][1] == "no")
    ex = {}
    for side in ("own", "oth"):
        for cc in conds:
            ex[f"{side}_{cc}"] = (lambda s, q: (lambda p: p[f"{s}_{q}_viol"]))(side, cc)
            ex[f"fa_{side}_{cc}"] = (lambda s, q: (lambda p: p[f"{s}_{q}_sat"]))(side, cc)
            if cc != "fresh":
                ex[f"d_{side}_{cc}"] = (lambda s, q: (lambda p: p[f"{s}_{q}_viol"] - p[f"{s}_fresh_viol"]))(side, cc)
    r = c.boot(ex)
    print(f"{'condition':28s} {'own detection':>22s} {'Δ own vs fresh':>22s} {'others detection':>22s} {'Δ others vs fresh':>22s} {'FA own':>8s} {'FA oth':>8s}")
    for cc in conds:
        fa = lambda k: f"{r[k]['est']:8.3f}" if k in r else f"{'-':>8s}"
        print(f"{cc:28s} {fmt(r, f'own_{cc}')} {fmt(r, f'd_own_{cc}')} {fmt(r, f'oth_{cc}')} {fmt(r, f'd_oth_{cc}')} {fa(f'fa_own_{cc}')} {fa(f'fa_oth_{cc}')}")
    print("n:", {k: v for k, v in r["_n"].items() if k.endswith("viol")})
    return r


def _repair_status(name, oracle="qwen3vl8b"):
    facts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    d = ROOT / "runs/repair" / name
    rounds = sorted(p.name for p in d.glob("round*") if (p / "manifest.jsonl").exists())
    if (d / "trace.jsonl").exists():
        trace = [json.loads(l) for l in open(d / "trace.jsonl", encoding="utf-8")]
    else:
        per = collections.defaultdict(list)
        for rd in rounds:
            verdict = collections.defaultdict(dict)
            for f in sorted((d / "judge").glob(f"*__{rd}*.jsonl")):
                for r in (json.loads(l) for l in open(f, encoding="utf-8")): verdict[r["pid"]][r["fid"]] = r["answer"]
            for m in (json.loads(l) for l in open(d / rd / "manifest.jsonl", encoding="utf-8")):
                v = verdict.get(m["pid"], {})
                per[m["pid"]].append({"k": int(rd[5:]), "image": f"{rd}/{m['image']}", "n_failed": sum(x == "no" for x in v.values()) if v else 1})
        trace = [{"pid": pid, "rounds": sorted(rs, key=lambda x: x["k"])} for pid, rs in per.items()]
    ora = {}
    for rd in rounds:
        f = d / "oracle" / f"{oracle}__{rd}.jsonl"
        if f.exists():
            for r in (json.loads(l) for l in open(f, encoding="utf-8")): ora[(rd, r["image"], r["fid"])] = r["answer"]
    out = {}
    for t in trace:
        seq = []
        for k in range(len(rounds)):
            kk = min(k, len(t["rounds"]) - 1); rdk = f"round{kk}"; im = t["rounds"][kk]["image"].split("/")[1]
            labs = [ora.get((rdk, im, fct["fid"])) for fct in facts[t["pid"]]["facts"]]
            if any(l is None for l in labs): seq = None; break
            vok = (t["rounds"][k]["n_failed"] == 0) if k < len(t["rounds"]) else None
            seq.append((all(l == "yes" for l in labs), sum(l == "yes" for l in labs) / len(labs), vok))
        if seq: out[t["pid"]] = seq
    return out


def repair(a):
    names = [p.name for p in sorted((ROOT / "runs/repair").glob("*")) if (p / "round0").exists()]
    st = {n: _repair_status(n) for n in names}
    res = {}
    for n, s in st.items():
        nr = len(next(iter(s.values())))
        act = [sum(1 for seq in s.values() if len(seq) > k and seq[k][2] is not None) for k in range(nr)]
        vok = [sum(1 for seq in s.values() if seq[k][2]) for k in range(nr)]
        print(f"{n:30s} prompts={len(s)}  verifier passes per round: " + "  ".join(f"r{k}:{vok[k]}/{act[k]}" for k in range(nr)))
    groups = collections.defaultdict(list)
    for n in names: groups[n.split("__")[0]].append(n)
    rng = np.random.default_rng(0)
    for g, ns in groups.items():
        common = sorted(set.intersection(*[set(st[n]) for n in ns]))
        start = np.array([st[ns[0]][p][0][0] for p in common], float)
        fin = {n: np.array([st[n][p][-1][0] for p in common], float) for n in ns}
        acc = {n: np.array([st[n][p][-1][1] for p in common]) for n in ns}
        idx = rng.integers(0, len(common), size=(B, len(common)))
        print(f"\n== generator {g}: {len(common)} common prompts; start all-ok {start.mean():.3f}")
        for n in ns:
            print(f"  {n:30s} all-ok {fin[n].mean():.3f} [{np.percentile(fin[n][idx].mean(1), 2.5):.3f},{np.percentile(fin[n][idx].mean(1), 97.5):.3f}]  fact-acc {acc[n].mean():.3f}")
            res[f"run:{n}"] = {"generator": g, "start": float(start.mean()), "all_ok": float(fin[n].mean()), "fact_acc": float(acc[n].mean()),
                               "ci95": [float(np.percentile(fin[n][idx].mean(1), 2.5)), float(np.percentile(fin[n][idx].mean(1), 97.5))], "n": len(common)}
        for i, n1 in enumerate(ns):
            for n2 in ns[i + 1:]:
                d = fin[n2] - fin[n1]; bd = d[idx].mean(1)
                b10 = int(((fin[n1] == 1) & (fin[n2] == 0)).sum()); b01 = int(((fin[n1] == 0) & (fin[n2] == 1)).sum())
                from math import comb
                nn = b10 + b01; k = min(b10, b01)
                p_mc = min(1.0, 2 * sum(comb(nn, j) for j in range(k + 1)) / 2 ** nn) if nn else 1.0
                print(f"  {n2} - {n1}: {d.mean():+.3f} [{np.percentile(bd, 2.5):+.3f},{np.percentile(bd, 97.5):+.3f}]  discordant {b01} vs {b10}, exact McNemar p={p_mc:.3f}")
                res[f"{g}:{n2}-{n1}"] = {"est": float(d.mean()), "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], "mcnemar_p": p_mc, "discordant": [b01, b10]}
    return res


def bytype(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows)
    typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
    judges = sorted({j for (j, _, _) in ans}); res = {}
    print(f"labels={a.labels}; per fact type; 95% CI; B={B}")
    print(f"{'judge':12s} {'type':9s} {'own fresh':>18s} {'oth fresh':>18s} {'self(fresh)':>22s} {'self-specific':>22s}  n_own n_oth")
    for J in judges:
        for t in ("presence", "count", "color", "position"):
            c = Cells()
            for (j, cond, g), d in ans.items():
                if j != J or cond not in ("fresh", "ctx"): continue
                side = "own" if g == GEN_OF.get(J, J) else "oth"
                for (im, fid), (pid, an) in d.items():
                    k = (g, im, fid)
                    if typ.get(k) == t and o.get(k) == "no": c.add(f"{side}_{cond}", pid, an == "no")
            if not all(x in c.n for x in ("own_fresh", "oth_fresh", "own_ctx", "oth_ctx")): continue
            if sum(c.n["own_fresh"].values()) < 5: continue
            ex = {"own_fresh": lambda q: q["own_fresh"], "oth_fresh": lambda q: q["oth_fresh"],
                  "own_ctx": lambda q: q["own_ctx"], "oth_ctx": lambda q: q["oth_ctx"],
                  "self_fresh": lambda q: q["own_fresh"] - q["oth_fresh"],
                  "selfspec": lambda q: (q["own_ctx"] - q["own_fresh"]) - (q["oth_ctx"] - q["oth_fresh"])}
            r = c.boot(ex); res[f"{J}:{t}"] = r
            print(f"{J:12s} {t:9s} {fmt(r, 'own_fresh')} {fmt(r, 'oth_fresh')} {fmt(r, 'self_fresh')} {fmt(r, 'selfspec')}  {r['_n']['own_fresh']:5d} {r['_n']['oth_fresh']:5d}")
    return res


LEN_BIN = {"single_object": "short (5 words)", "colors": "medium (6-7 words)", "counting": "medium (6-7 words)",
           "two_object": "long (8-10 words)", "position": "long (8-10 words)", "color_attr": "long (8-10 words)"}


def length(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows)
    typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
    tag = {json.loads(l)["pid"]: json.loads(l)["tag"] for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8")}
    nwords = {json.loads(l)["pid"]: len(json.loads(l)["prompt"].split()) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8")}
    judges = sorted({j for (j, _, _) in ans}); res = {}; bins = ["short (5 words)", "medium (6-7 words)", "long (8-11 words)"]
    lenbin = lambda n: bins[0] if n <= 5 else (bins[1] if n <= 7 else bins[2])
    print(f"labels={a.labels}; presence facts only; framing = others closed - others fresh; 95% CI")
    for J in judges:
        c = Cells()
        for (j, cond, g), d in ans.items():
            if j != J or g == J or cond not in ("fresh", "ctx"): continue
            for (im, fid), (pid, an) in d.items():
                k = (g, im, fid)
                if typ.get(k) == "presence" and o.get(k) == "no": c.add(f"{lenbin(nwords[pid])}|{cond}", pid, an == "no")
        ex = {b: (lambda b: (lambda q: q[f"{b}|ctx"] - q[f"{b}|fresh"]))(b) for b in bins}
        ex.update({f"long-short": lambda q: (q[f"{bins[2]}|ctx"] - q[f"{bins[2]}|fresh"]) - (q[f"{bins[0]}|ctx"] - q[f"{bins[0]}|fresh"])})
        r = c.boot(ex); res[J] = r
        cell = lambda k: f"{r[k]['est']:+.2f} [{r[k]['ci95'][0]:+.2f},{r[k]['ci95'][1]:+.2f}]" if k in r else "-"
        print(f"{J:12s} " + "  ".join(f"{bb.split()[0]}: {cell(bb)} (n={r['_n'].get(bb + '|fresh', 0)})" for bb in bins) + f"  long-short {cell('long-short')}")
    return res


def stdid(a):
    rows = load_rows(a.judge_dir); o, hg = labels(rows, a.labels); ans = answers(rows)
    typ = {(r["gen"], r["image"], r["fid"]): r["type"] for r in rows if r["judge"] == "qwen3vl8b"}
    judges = sorted({j for (j, _, _) in ans}); res = {}; T = ("presence", "count", "color", "position")
    print(f"labels={a.labels}; type-standardised self-specific (weights = own type mix); 95% / 90% CI")
    for J in judges:
        c = Cells()
        for (j, cond, g), d in ans.items():
            if j != J or cond not in ("fresh", "ctx"): continue
            side = "own" if g == GEN_OF.get(J, J) else "oth"
            if side == "oth" and a.common and g not in a.common: continue
            for (im, fid), (pid, an) in d.items():
                k = (g, im, fid)
                if o.get(k) == "no" and typ.get(k) in T: c.add(f"{side}_{cond}_{typ[k]}", pid, an == "no")
        tt = [t for t in T if all(f"{sd}_{cd}_{t}" in c.n for sd in ("own", "oth") for cd in ("fresh", "ctx"))]
        nown = {t: sum(c.n[f"own_fresh_{t}"].values()) for t in tt}; tot = sum(nown.values())
        if not tt or tot == 0: continue
        w = {t: nown[t] / tot for t in tt}
        def f(q, kind):
            out = 0
            for t in tt:
                own_d = q[f"own_ctx_{t}"] - q[f"own_fresh_{t}"]; oth_d = q[f"oth_ctx_{t}"] - q[f"oth_fresh_{t}"]
                out = out + w[t] * ((own_d - oth_d) if kind == "ss" else oth_d if kind == "cl" else (q[f"own_fresh_{t}"] - q[f"oth_fresh_{t}"]))
            return out
        r = c.boot({"selfspec_std": lambda q: f(q, "ss"), "closedloop_std": lambda q: f(q, "cl"), "self_fresh_std": lambda q: f(q, "sf")})
        r["_weights"] = w; res[J] = r
        lo90 = r["selfspec_std"]["ci90"][0]
        print(f"{J:12s} self-specific_std {fmt(r, 'selfspec_std')}  90% lo {lo90:+.3f}  closed-loop_std(others) {fmt(r, 'closedloop_std')}  self(fresh)_std {fmt(r, 'self_fresh_std')}  w={ {k: round(v, 2) for k, v in w.items()} }")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["did", "decomp", "repair", "bytype", "length", "stdid", "matched"])
    ap.add_argument("--labels", default="human", choices=["human", "strict", "vlm", "hybrid", "aikit"]); ap.add_argument("--judge_dir", default=str(ROOT / "runs/judge"))
    ap.add_argument("--judge"); ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--conds", nargs="*", default=["fresh", "ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline",
                                                    "ctx-assistant_mismatch", "ctx-assistant_cf"])
    ap.add_argument("--common", nargs="*", default=None, help="did: restrict others' images to these generators")
    ap.add_argument("--partial", action="store_true", help="did: also keep judges with own-image cells only")
    ap.add_argument("--json", default=None); a = ap.parse_args()
    res = {"did": did, "decomp": decomp, "repair": repair, "bytype": bytype, "length": length, "stdid": stdid, "matched": matched}[a.what](a)
    if a.json: json.dump(res, open(a.json, "w"), indent=1, default=str)
