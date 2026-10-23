import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
_P = None

NUM = {2: "two", 3: "three", 4: "four"}
CF_COUNT = {2: 3, 3: 4, 4: 2}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]
FLIP = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}
SUBST = ["umbrella", "bicycle", "clock", "chair", "bottle", "laptop", "suitcase", "banana", "cake", "truck", "kite", "vase"]


def _prompts():
    global _P
    if _P is None:
        _P = {r["pid"]: r for r in (json.loads(l) for l in open(ROOT / "data/prompts.jsonl", encoding="utf-8"))}
    return _P


def _classes(r):
    return {f["args"]["class"] for f in r["facts"]} | {f["args"]["other"] for f in r["facts"] if "other" in f["args"]}


def mismatch(pid, offset=7):
    P = _prompts(); r = P[pid]
    same = [q for q in P.values() if q["tag"] == r["tag"]]
    i = next(k for k, q in enumerate(same) if q["pid"] == pid)
    for step in range(offset, offset + len(same)):
        q = same[(i + step) % len(same)]
        if q["pid"] != pid and not (_classes(q) & _classes(r)):
            return q["prompt"]
    return None


def _plural(w):
    return w + ("es" if re.search(r"(s|x|ch|sh)$", w) else "s")


def _fix_articles(s):
    return re.sub(r"\b(a|an) (\w)", lambda m: ("an " if m.group(2).lower() in "aeiou" else "a ") + m.group(2), s)


def counterfactual(prompt, fact):
    t, a = fact["type"], fact["args"]
    out = None
    if t == "position":
        rel = a["relation"]
        if rel in prompt: out = prompt.replace(rel, FLIP[rel], 1)
    elif t == "count":
        n = a["count"]
        if NUM.get(n) and re.search(rf"\b{NUM[n]}\b", prompt):
            out = re.sub(rf"\b{NUM[n]}\b", NUM[CF_COUNT[n]], prompt, count=1)
    elif t == "color":
        c, cls = a["color"], a["class"]
        used = set(re.findall(r"\b(" + "|".join(COLORS) + r")\b", prompt))
        alt = next(x for x in COLORS[COLORS.index(c) + 3:] + COLORS[:COLORS.index(c) + 3] if x not in used and x != c)
        if f"{c} {cls}" in prompt: out = prompt.replace(f"{c} {cls}", f"{alt} {cls}", 1)
    elif t == "presence":
        cls = a["class"]
        present = set(re.findall(r"[a-z]+", prompt))
        alt = next((x for x in SUBST if x not in present and x != cls), None)
        pl = re.search(rf"\b{re.escape(cls)}(es|s)\b", prompt)
        if alt and pl: out = prompt[:pl.start()] + _plural(alt) + prompt[pl.end():]
        elif alt and re.search(rf"\b{re.escape(cls)}\b", prompt): out = re.sub(rf"\b{re.escape(cls)}\b", alt, prompt, count=1)
    if out is None or out == prompt: return None
    return _fix_articles(out)


if __name__ == "__main__":
    P = _prompts(); ok = 0; tot = 0; ex = {}
    for r in P.values():
        assert mismatch(r["pid"]) is not None, r["pid"]
        for f in r["facts"]:
            tot += 1; c = counterfactual(r["prompt"], f)
            ok += c is not None; ex.setdefault(f["type"], (r["prompt"], f["question"], c))
    print(f"counterfactual coverage {ok}/{tot}")
    for k, v in ex.items(): print(k, "|", v)
    print("mismatch example:", P["ge_0300"]["prompt"], "->", mismatch("ge_0300"))
