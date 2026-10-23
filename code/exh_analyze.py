import collections, json, sys
from bootstrap_ci import load_rows, labels, answers, Cells, GEN_OF, ROOT
GENS = ["bagel", "omnigen2", "janus7b", "showo2_7b"]

def main():
    lab = json.load(open(sys.argv[1]))["exh"]; key = json.load(open(ROOT / "runs/exhaustive_kit/key.json"))
    M = {"violated": "no", "satisfied": "yes"}
    E = {(it["gen"], it["image"], it["fid"]): M[lab["labels"][str(it["idx"])]] for it in key if lab["labels"].get(str(it["idx"])) in M}
    typ = {(it["gen"], it["image"], it["fid"]): it["type"] for it in key}
    rows = load_rows(ROOT / "runs/judge")
    vlm, _ = labels(rows, "vlm"); strict, _ = labels(rows, "strict"); hum, _ = labels(rows, "human")
    res = {"n_items": len(key), "n_labelled_binary": len(E), "ambiguous": sum(v == "ambiguous" for v in lab["labels"].values())}
    print(f"exhaustive: {len(key)} items, {len(E)} binary, {sum(v == 'no' for v in E.values())} violated")
    agree = {}
    for name, O in (("vlm", vlm), ("strict", strict), ("main_human", hum)):
        for t in ("all", "presence", "color", "count", "position"):
            ks = [k for k in E if k in O and (t == "all" or typ[k] == t)]
            hv = [k for k in ks if E[k] == "no"]; ov = [k for k in ks if O[k] == "no"]
            a = {"n": len(ks), "agree": sum(E[k] == O[k] for k in ks) / max(len(ks), 1), "human_viol": len(hv), "oracle_viol": len(ov),
                 "oracle_misses": sum(O[k] == "yes" for k in hv), "oracle_false_flags": sum(E[k] == "yes" for k in ov),
                 "uncovered": sum(1 for k in E if (t == "all" or typ[k] == t) and k not in O)}
            agree[f"{name}:{t}"] = a
            print(f"  {name:10s} {t:9s} n={a['n']:4d} agree={a['agree']:.3f} human_viol={a['human_viol']:3d} oracle_viol={a['oracle_viol']:3d}"
                  f" missed_by_oracle={a['oracle_misses']:3d} oracle_false={a['oracle_false_flags']:3d} not_in_oracle={a['uncovered']}")
    res["agreement"] = agree
    ans = answers(rows); judges = sorted({j for (j, c, g) in ans if g in GENS and c == "ctx"})
    unsel = {k: "no" for k, v in E.items() if v == "no" and vlm.get(k) == "yes"}
    sets = {"exhaustive": E, "main_same_images": {k: v for k, v in hum.items() if k in typ}, "oracle_unselected": unsel}
    ex = {"own_fresh": lambda p: p["own_fresh"], "oth_fresh": lambda p: p["oth_fresh"], "own_ctx": lambda p: p["own_ctx"], "oth_ctx": lambda p: p["oth_ctx"],
          "closed_oth": lambda p: p["oth_ctx"] - p["oth_fresh"], "closed_own": lambda p: p["own_ctx"] - p["own_fresh"],
          "self_fresh": lambda p: p["own_fresh"] - p["oth_fresh"], "selfspec": lambda p: (p["own_ctx"] - p["own_fresh"]) - (p["oth_ctx"] - p["oth_fresh"])}
    res["did"] = {}
    for sname, O in sets.items():
        print(f"\n== {sname}: closed-loop change on others' / own violations, self (fresh), self-specific; n = violations (others, own)")
        for J in judges:
            c = Cells(); own_g = GEN_OF.get(J, J)
            for g in GENS:
                if (J, "fresh", g) not in ans or (J, "ctx", g) not in ans: continue
                side = "own" if g == own_g else "oth"
                for cond in ("fresh", "ctx"):
                    for (im, fid), (pid, an) in ans[(J, cond, g)].items():
                        if O.get((g, im, fid)) == "no": c.add(f"{side}_{cond}", pid, an == "no")
            r = c.boot(ex); res["did"][f"{sname}:{J}"] = r
            f = lambda k: f"{r[k]['est']:+.2f} [{r[k]['ci95'][0]:+.2f},{r[k]['ci95'][1]:+.2f}]" if k in r else " " * 20
            n = r["_n"]
            print(f"  {J:14s} oth {r.get('oth_fresh', {}).get('est', float('nan')):.2f}->{r.get('oth_ctx', {}).get('est', float('nan')):.2f} {f('closed_oth')}"
                  f"  own {f('closed_own')}  self {f('self_fresh')}  selfspec {f('selfspec')}  n=({n.get('oth_fresh', 0)},{n.get('own_fresh', 0)})")
    if "--json" in sys.argv: json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)

if __name__ == "__main__":
    main()
