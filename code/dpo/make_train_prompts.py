import argparse, json, pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from build_prompts import facts_for, article, plural, NUM

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per_tag", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    meta = [json.loads(l) for l in open(ROOT / "data/geneval_metadata.jsonl", encoding="utf-8")]
    test = {m["prompt"] for m in meta}
    classes = sorted({o["class"] for m in meta for o in m["include"]})
    colors = sorted({o["color"] for m in meta for o in m["include"] if "color" in o})
    rels = ["left of", "right of", "above", "below"]
    def make(tag):
        A, B = rng.sample(classes, 2)
        if tag == "two_object":
            return {"tag": tag, "include": [{"class": A, "count": 1}, {"class": B, "count": 1}], "prompt": f"a photo of {article(A)} {A} and {article(B)} {B}"}
        if tag == "counting":
            n = rng.choice([2, 3, 4])
            return {"tag": tag, "include": [{"class": A, "count": n}], "prompt": f"a photo of {NUM[n]} {plural(A)}"}
        if tag == "colors":
            c = rng.choice(colors)
            return {"tag": tag, "include": [{"class": A, "count": 1, "color": c}], "prompt": f"a photo of {article(c)} {c} {A}"}
        if tag == "position":
            r = rng.choice(rels)
            return {"tag": tag, "include": [{"class": B, "count": 1}, {"class": A, "count": 1, "position": [r, 0]}], "prompt": f"a photo of {article(A)} {A} {r} {article(B)} {B}"}
        c1, c2 = rng.sample(colors, 2)
        return {"tag": tag, "include": [{"class": A, "count": 1, "color": c1}, {"class": B, "count": 1, "color": c2}],
                "prompt": f"a photo of {article(c1)} {c1} {A} and {article(c2)} {c2} {B}"}
    rows, seen = [], set(test)
    for tag in ("two_object", "counting", "colors", "position", "color_attr"):
        k = 0; quota = min(a.per_tag, 150) if tag == "counting" else a.per_tag
        while k < quota:
            m = make(tag)
            if m["prompt"] in seen: continue
            seen.add(m["prompt"]); pid = f"tr_{len(rows):04d}"
            rows.append({"pid": pid, "source": "train", "tag": tag, "prompt": m["prompt"], "facts": facts_for(m, pid)}); k += 1
    with open(ROOT / "data/train_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts, {sum(len(r['facts']) for r in rows)} facts")

if __name__ == "__main__":
    main()
