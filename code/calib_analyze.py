import argparse, json
import numpy as np
import e2e_analyze as E

SETS = {"BAGEL, GenEval facts": ("bagel__bagel_s{s}", "bagel__bagel__ctx_s{s}", "bagel__bagel__ctx_thr_s{s}"),
        "BAGEL, ERASE questions": ("bagel__bagel_dqt_s{s}", "bagel__bagel__ctx_dqt_s{s}", "bagel__bagel__ctx_dqtthr_s{s}"),
        "OmniGen2, ERASE questions": ("omnigen2__omnigen2_dqt_s{s}", "omnigen2__omnigen2__ctx_dqt_s{s}", "omnigen2__omnigen2__ctx_dqtthr_s{s}")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rule", default="oracle", choices=["oracle", "agree", "either"]); ap.add_argument("--json")
    a = ap.parse_args(); res = {}
    for k, names in SETS.items():
        fin, rej = {}, {}
        for lab, n in zip(("fresh", "closed", "closed, transferred threshold"), names):
            fin[lab] = {}; rej[lab] = 0
            for s in (1, 2):
                st = E.status(n.format(s=s), a.rule)
                fin[lab].update({(p, s): float(v[-1][0]) for p, v in st.items()})
                rej[lab] += sum(r["rounds"][0]["n_failed"] > 0 for r in E.trace(n.format(s=s)).values())
        out = {lab: {"fully_correct": float(np.mean(list(v.values()))), "n": len(v), "first_rejected": rej[lab]} for lab, v in fin.items()}
        out["thr_vs_fresh"] = E.compare(fin["closed, transferred threshold"], fin["fresh"], lambda key: key[0])
        out["thr_vs_closed"] = E.compare(fin["closed, transferred threshold"], fin["closed"], lambda key: key[0])
        res[k] = out
        c1, c2 = out["thr_vs_fresh"], out["thr_vs_closed"]
        print(f"{k:26s} fresh {out['fresh']['fully_correct']:.3f} (rej {rej['fresh']})  closed {out['closed']['fully_correct']:.3f} (rej {rej['closed']})  "
              f"thr {out['closed, transferred threshold']['fully_correct']:.3f} (rej {rej['closed, transferred threshold']})  "
              f"thr-fresh {c1['est']:+.3f} [{c1['ci95'][0]:+.3f},{c1['ci95'][1]:+.3f}] p={c1['p']:.4f}  thr-closed {c2['est']:+.3f} p={c2['p']:.4f}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
