import json, random, pathlib, sys
from bootstrap_ci import load_rows, labels, ROOT
judge = sys.argv[1]
gens = {"janus7b": ["janus7b", "showo2_7b"], "janus1b": ["janus1b", "janus7b", "showo2_7b"], "omnigen2": ["omnigen2", "janus7b", "showo2_7b"], "bagel": ["bagel", "janus7b", "showo2_7b"]}[judge]
rows = [r for r in load_rows(str(ROOT / "runs/judge")) if r["judge"] in ("qwen3vl8b", "owlv2") and r["gen"] in gens]
lab, _ = labels(rows, "human")
rng = random.Random(2027); sub = {}
for g in gens:
    v = sorted([k[1], k[2], "violated"] for k, a in lab.items() if k[0] == g and a == "no")
    s = sorted([k[1], k[2], "satisfied"] for k, a in lab.items() if k[0] == g and a == "yes")
    sub[g] = v + rng.sample(s, min(300, len(s)))
    print(g, len(v), "violated +", min(300, len(s)), "satisfied")
out = ROOT / "runs/mask"; out.mkdir(parents=True, exist_ok=True)
json.dump(sub, open(out / f"subset_{judge}.json", "w"))
