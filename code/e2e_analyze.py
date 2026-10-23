import argparse, collections, json, math
import numpy as np
import pathlib
from bootstrap_ci import _repair_status
FACTS = {r["pid"]: r["facts"] for r in (json.loads(l) for l in open("data/prompts.jsonl", encoding="utf-8"))}


def _load(d, j):
    o = {}
    for f in (d / "oracle").glob(f"{j}__round*.jsonl"):
        rd = f.stem.split("__")[1]
        for r in map(json.loads, open(f)): o[(rd, r["image"], r["fid"])] = r["answer"]
    return o


def status(name, rule):
    if rule == "oracle": return _repair_status(name)
    d = pathlib.Path("runs/repair") / name; q, w = _load(d, "qwen3vl8b"), _load(d, "owlv2"); out = {}
    for t in map(json.loads, open(d / "trace.jsonl", encoding="utf-8")):
        seq = []
        for rnd in (t["rounds"][0], t["rounds"][-1]):
            rd, im = rnd["image"].split("/"); ok = True
            for fc in FACTS[t["pid"]]:
                x = q.get((rd, im, fc["fid"])); y = w.get((rd, im, fc["fid"]), x) if fc["type"] != "color" else x
                if x is None: ok = None; break
                ok = ok and not ((x == "no" and y == "no") if rule == "agree" else (x == "no" or y == "no"))
            if ok is None: break
            seq.append((ok, None, None))
        if len(seq) == 2: out[t["pid"]] = seq
    return out

RUNS = {"bagel": {"ERASE": ["bagel__bagel_dqt", "bagel__bagel_dqt_s1", "bagel__bagel_dqt_s2"],
                  "ERASE, closed loop": ["bagel__bagel__ctx_dqt", "bagel__bagel__ctx_dqt_s1", "bagel__bagel__ctx_dqt_s2"],
                  "ERASE, no feedback": ["bagel__bagel_dqtnf", "bagel__bagel_dqtnf_s1", "bagel__bagel_dqtnf_s2"],
                  "UniGen": ["bagel__bagel_ug", "bagel__bagel_ug_s1", "bagel__bagel_ug_s2"],
                  "UniCorn": ["bagel__bagel_uc", "bagel__bagel_uc_s1", "bagel__bagel_uc_s2"],
                  "UniGen, matched threshold": ["bagel__bagel_ugthr", "bagel__bagel_ugthr_s1", "bagel__bagel_ugthr_s2"]},
        "omnigen2": {"ERASE": ["omnigen2__omnigen2_dqt", "omnigen2__omnigen2_dqt_s1", "omnigen2__omnigen2_dqt_s2"],
                     "ERASE, closed loop": ["omnigen2__omnigen2__ctx_dqt", "omnigen2__omnigen2__ctx_dqt_s1", "omnigen2__omnigen2__ctx_dqt_s2"],
                     "ERASE, no feedback": ["omnigen2__omnigen2_dqtnf", "omnigen2__omnigen2_dqtnf_s1", "omnigen2__omnigen2_dqtnf_s2"],
                     "UniGen": ["omnigen2__omnigen2_ug", "omnigen2__omnigen2_ug_s1", "omnigen2__omnigen2_ug_s2"],
                     "UniCorn": ["omnigen2__omnigen2_uc", "omnigen2__omnigen2_uc_s1", "omnigen2__omnigen2_uc_s2"],
                     "UniGen, matched threshold": ["omnigen2__omnigen2_ugthr", "omnigen2__omnigen2_ugthr_s1", "omnigen2__omnigen2_ugthr_s2"]}}


def mcnemar(b01, b10):
    n = b01 + b10; k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0


def trace(name):
    return {(r["pid"], r["seed"]): r for r in (json.loads(l) for l in open(f"runs/repair/{name}/trace.jsonl"))}


def score0(rec, kind):
    v = rec["rounds"][0]["verdicts"]
    if kind.startswith("UniGen"): return v[0]["p_yes"] if v[0].get("p_yes") is not None else -1
    if kind == "UniCorn": return v[0]["score"] if v[0].get("score") is not None else -1
    ps = [1 - x["p_no"] for x in v if x.get("p_no") is not None]
    return float(np.mean(ps)) if ps else 1.0


def compare(fin, ref, pids_of, B=2000, seed=0):
    keys = sorted(set(fin) & set(ref)); x = np.array([fin[k] for k in keys], float); y = np.array([ref[k] for k in keys], float)
    pids = sorted({pids_of(k) for k in keys}); P = {p: i for i, p in enumerate(pids)}; idx = np.array([P[pids_of(k)] for k in keys])
    rng = np.random.default_rng(seed); d = x - y; bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids))[idx]; bs.append((w * d).sum() / w.sum())
    b01 = int(((x == 1) & (y == 0)).sum()); b10 = int(((x == 0) & (y == 1)).sum())
    return {"n": len(keys), "est": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "disc": [b01, b10], "p": mcnemar(b01, b10)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--rule", default="oracle", choices=["oracle", "agree", "either"]); a = ap.parse_args(); res = {}
    for g, V in RUNS.items():
        st = {k: [status(n, a.rule) for n in ns] for k, ns in V.items()}
        tr = {k: [trace(n) for n in ns] for k, ns in V.items()}
        fin = {k: {(p, s): float(st[k][s][p][-1][0]) for s in range(3) for p in st[k][s]} for k in V}
        out = {"repair": {}, "bon3": {}}
        for k in V:
            regen = sum(len(r["rounds"]) - 1 for t in tr[k] for r in t.values()); rej0 = sum(r["rounds"][0]["n_failed"] > 0 for t in tr[k] for r in t.values())
            unp = sum(1 for t in tr[k] for r in t.values() for rd in r["rounds"] if rd["verdicts"] and rd["verdicts"][0].get("fid") == "unicorn" and rd["verdicts"][0].get("score") is None)
            out["repair"][k] = {"fully_correct": float(np.mean(list(fin[k].values()))), "n": len(fin[k]), "regenerations": regen,
                                "first_rejected": rej0, "unparsable": unp,
                                "per_seed": [float(np.mean([v for (p, s), v in fin[k].items() if s == sd])) for sd in range(3)]}
            if k != "ERASE": out["repair"][k]["vs_ERASE"] = compare(fin[k], fin["ERASE"], lambda key: key[0])
            if k in ("UniGen", "UniCorn", "ERASE, closed loop", "UniGen, matched threshold"): out["repair"][k]["vs_ERASE_nofb"] = compare(fin[k], fin["ERASE, no feedback"], lambda key: key[0])
        start = {(p, s): float(st["ERASE"][s][p][0][0]) for s in range(3) for p in st["ERASE"][s]}
        out["start"] = float(np.mean(list(start.values())))
        pids = sorted({p for (p, s) in start if all((p, sd) in start for sd in range(3))})
        picks = {}
        for k in V:
            pk = {}
            for p in pids:
                sc = [score0(tr[k][sd][(p, sd)], "ERASE" if k.startswith("ERASE") else ("UniGen" if k.startswith("UniGen") else k)) if (p, sd) in tr[k][sd] else -1 for sd in range(3)]
                pk[(p, 0)] = start[(p, int(np.argmax(sc)))]
            picks[k] = pk
        picks["seed 0 (no selection)"] = {(p, 0): start[(p, 0)] for p in pids}
        picks["oracle (any seed correct)"] = {(p, 0): float(max(start[(p, sd)] for sd in range(3))) for p in pids}
        for k, pk in picks.items():
            out["bon3"][k] = {"fully_correct": float(np.mean(list(pk.values())))}
            if k != "ERASE": out["bon3"][k]["vs_ERASE"] = compare(pk, picks["ERASE"], lambda key: key[0])
        res[g] = out
        print(f"== {g}: start {out['start']:.3f}")
        for k, v in out["repair"].items():
            c = v.get("vs_ERASE"); cs = f"  vs ERASE {c['est']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] disc {c['disc']} p={c['p']:.4f}" if c else ""
            c2 = v.get("vs_ERASE_nofb"); cs += f"  vs ERASE-nofb {c2['est']:+.3f} [{c2['ci95'][0]:+.3f},{c2['ci95'][1]:+.3f}] disc {c2['disc']} p={c2['p']:.4f}" if c2 else ""
            print(f"  repair {k:20s} {v['fully_correct']:.3f} (seeds {', '.join(f'{x:.3f}' for x in v['per_seed'])}) regen {v['regenerations']} first-rejected {v['first_rejected']} unparsable {v['unparsable']}{cs}")
        for k, v in out["bon3"].items():
            c = v.get("vs_ERASE"); cs = f"  vs ERASE {c['est']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] disc {c['disc']} p={c['p']:.4f}" if c else ""
            print(f"  best-of-3 {k:26s} {v['fully_correct']:.3f}{cs}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
