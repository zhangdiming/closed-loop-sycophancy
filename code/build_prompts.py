import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
NUM = {1: "one", 2: "two", 3: "three", 4: "four"}
VOWELS = "aeiou"

def article(w):
    return "an" if w[0] in VOWELS else "a"

def plural(w):
    if w.endswith("s") or w.endswith("ch") or w.endswith("sh") or w.endswith("x"):
        return w + "es"
    if w.endswith("y") and w[-2] not in VOWELS:
        return w[:-1] + "ies"
    return w + "s"

def q(text):
    return text + " Answer yes or no."

def facts_for(meta, pid):
    facts = []
    inc = meta["include"]
    def add(t, question, args):
        facts.append({"fid": f"{pid}_f{len(facts)}", "type": t, "question": q(question),
                      "expected": "yes", "args": args})
    for i, o in enumerate(inc):
        c, n = o["class"], o.get("count", 1)
        if n == 1:
            add("presence", f"Is there {article(c)} {c} in the image?", {"class": c})
        else:
            add("presence", f"Are there any {plural(c)} in the image?", {"class": c})
            add("count", f"Are there exactly {NUM[n]} {plural(c)} in the image?", {"class": c, "count": n})
        if "color" in o:
            add("color", f"Is the {c} {o['color']}?", {"class": c, "color": o["color"]})
    for i, o in enumerate(inc):
        if "position" in o:
            rel, j = o["position"]
            other = inc[j]["class"]
            add("position", f"Is the {o['class']} {rel} the {other}?",
                {"class": o["class"], "relation": rel, "other": other})
    return facts

def main():
    src = ROOT / "data" / "geneval_metadata.jsonl"
    out = ROOT / "data" / "prompts.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    n_facts = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, m in enumerate(rows):
            pid = f"ge_{i:04d}"
            facts = facts_for(m, pid)
            n_facts += len(facts)
            f.write(json.dumps({"pid": pid, "source": "geneval", "tag": m["tag"], "prompt": m["prompt"],
                                "facts": facts}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts, {n_facts} facts -> {out}")

if __name__ == "__main__":
    main()
