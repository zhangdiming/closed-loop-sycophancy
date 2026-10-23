import collections, hashlib, json, pathlib, random, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parent; OUT = pathlib.Path(sys.argv[1]); rng = random.Random(2028)
GENS = ["bagel", "omnigen2", "janus7b", "showo2_7b"]
prompts = [json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8")]
by_tag = collections.defaultdict(list)
for p in prompts: by_tag[p["tag"]].append(p)
sample = [p for t in sorted(by_tag) for p in rng.sample(sorted(by_tag[t], key=lambda r: r["pid"]), 25)]
images = [(g, p) for p in sample for g in GENS if (ROOT / f"runs/gen/{g}/{p['pid']}_s0.png").exists()]
rng.shuffle(images)
(OUT / "images").mkdir(parents=True, exist_ok=True); items = []
for n, (g, p) in enumerate(images):
    src = ROOT / f"runs/gen/{g}/{p['pid']}_s0.png"; h = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    shutil.copy(src, OUT / f"images/{h}.png")
    for k, f in enumerate(p["facts"]):
        items.append({"idx": len(items), "img": f"images/{h}.png", "q": f["question"].replace(" Answer yes or no.", "").rstrip("?"),
                      "im": n + 1, "qk": k + 1, "qn": len(p["facts"]), "gen": g, "pid": p["pid"], "image": f"{p['pid']}_s0.png",
                      "fid": f["fid"], "type": f["type"], "tag": p["tag"]})
shipped = [{"gen": "exh", "idx": it["idx"], "img": it["img"], "q": it["q"], "im": it["im"], "qk": it["qk"], "qn": it["qn"], "sha": "exhaustive-v1"} for it in items]
open(OUT / "items.js", "w").write("const ITEMS=" + json.dumps(shipped, ensure_ascii=False) + ";\nconst NIMG=" + str(len(images)) + ";\n")
json.dump(items, open(OUT / "key.json", "w"), indent=1)
print(len(sample), "prompts,", len(images), "images,", len(items), "facts", dict(collections.Counter(it["type"] for it in items)))
