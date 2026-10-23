import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_ci import ROOT, _repair_status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    names = [p.name for p in sorted((ROOT / "runs/repair").glob("*")) if (p / "round0").exists()
             and not p.name.endswith(("_s1", "_s2")) and not p.name.endswith("__none")]
    st = {n: _repair_status(n) for n in names}
    groups = collections.defaultdict(list)
    for n in names: groups[n.split("__")[0]].append(n)
    res = {}
    for g, ns in groups.items():
        common = sorted(set.intersection(*[set(st[n]) for n in ns]))
        res[g] = {"n": len(common), "start_all_ok": float(np.mean([st[ns[0]][p][0][0] for p in common])),
                  "start_fact_acc": float(np.mean([st[ns[0]][p][0][1] for p in common])), "runs": {}}
        for n in ns:
            res[g]["runs"][n] = {"all_ok": float(np.mean([st[n][p][-1][0] for p in common])),
                                 "fact_acc": float(np.mean([st[n][p][-1][1] for p in common]))}
        print(g, json.dumps(res[g], indent=None))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
