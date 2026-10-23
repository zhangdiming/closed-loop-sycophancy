import json, random, shutil, pathlib, hashlib, collections, sys
import bootstrap_ci as b
ROOT = pathlib.Path(__file__).resolve().parent; OUT = pathlib.Path(sys.argv[1]); rng = random.Random(2027)
prompts = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
fact = {f["fid"]: f for p in prompts.values() for f in p["facts"]}
rows = b.load_rows(ROOT / "runs/judge"); strict, _ = b.labels(rows, "strict")
vlm = {(r["gen"], r["image"], r["fid"]): r["answer"] for r in rows if r["judge"] == "qwen3vl8b" and not r.get("context")}
GENS = ["janus7b", "showo2_7b", "bagel", "lumina", "blip3o", "omnigen2"]
items = []
def q(fid): return fact[fid]["question"].replace(" Answer yes or no.", "").rstrip("?")
for g in GENS:
    for t in ("presence", "count", "color", "position"):
        pool = sorted(k for k, v in vlm.items() if k[0] == g and v == "yes" and fact[k[2]]["type"] == t and k[2].startswith("ge_"))
        for k in rng.sample(pool, min(6, len(pool))):
            items.append({"set": "S", "gen": g, "src": f"runs/gen/{g}/{k[1]}", "image": k[1], "fid": k[2], "type": t, "oracle": "yes"})
for g in GENS:
    pool = sorted(k for k, v in strict.items() if k[0] == g and v == "no" and fact[k[2]]["type"] == "count")
    for k in rng.sample(pool, min(15, len(pool))):
        items.append({"set": "C", "gen": g, "src": f"runs/gen/{g}/{k[1]}", "image": k[1], "fid": k[2], "type": "count", "oracle": "no"})
st = {n: b._repair_status(n) for n in ("bagel__bagel", "bagel__bagel__ctx")}
def final_image(name, pid):
    t = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "runs/repair" / name / "trace.jsonl", encoding="utf-8")}[pid]
    return t["rounds"][-1]["image"]
disc = sorted(p for p in set(st["bagel__bagel"]) & set(st["bagel__bagel__ctx"]) if st["bagel__bagel"][p][-1][0] != st["bagel__bagel__ctx"][p][-1][0])
ora = {}
for name in st:
    for f in (ROOT / "runs/repair" / name / "oracle").glob("qwen3vl8b__round*.jsonl"):
        for r in map(json.loads, open(f)): ora[(name, f.stem.split("__")[1], r["image"], r["fid"])] = r["answer"]
for name in st:
    for pid in disc:
        im = final_image(name, pid); rd, fn = im.split("/")
        for fc in prompts[pid]["facts"]:
            items.append({"set": "R", "gen": "bagel", "run": name, "pid": pid, "src": f"runs/repair/{name}/{im}", "image": fn, "round": rd,
                          "fid": fc["fid"], "type": fc["type"], "oracle": ora.get((name, rd, fn, fc["fid"]))})
rng.shuffle(items)
(OUT / "images").mkdir(parents=True, exist_ok=True); seen = {}
for i, it in enumerate(items):
    src = ROOT / it["src"]; h = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    if h not in seen: seen[h] = f"images/{h}.png"; shutil.copy(src, OUT / seen[h])
    it["idx"] = i; it["img"] = seen[h]; it["q"] = q(it["fid"])
shipped = [{"gen": "audit", "idx": it["idx"], "img": it["img"], "q": it["q"], "sha": "audit-v1"} for it in items]
open(OUT / "items.js", "w").write("const ITEMS=" + json.dumps(shipped, ensure_ascii=False) + ";\n")
json.dump(items, open(OUT / "key.json", "w"), indent=1)
print(len(items), collections.Counter(it["set"] for it in items), "discordant prompts", len(disc), "images", len(seen))
