import json, pathlib, collections
from PIL import Image, ImageDraw
ROOT = pathlib.Path(__file__).resolve().parent
P = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
F = {f["fid"]: f for r in P.values() for f in r["facts"]}
for g in ["janus7b", "showo2_7b", "bagel", "omnigen2", "lumina", "blip3o"]:
    o = [json.loads(l) for l in open(ROOT / f"runs/judge/qwen3vl8b__{g}.jsonl")]
    det = {(r["image"], r["fid"]): r["answer"] for r in map(json.loads, open(ROOT / f"runs/judge/owlv2__{g}.jsonl"))}
    done = {(f["image"], f["fid"]) for f in json.load(open(ROOT / f"runs/human/{g}_flagged.json"))}
    todo = [r for r in o if r["answer"] == "no" and F[r["fid"]]["type"] != "count" and (r["image"], r["fid"]) not in done
            and (F[r["fid"]]["type"] == "color" or det.get((r["image"], r["fid"])) == "no")]
    todo = sorted(todo, key=lambda r: (F[r["fid"]]["type"], r["image"]))
    items = [{"image": r["image"], "fid": r["fid"], "type": F[r["fid"]]["type"], "question": F[r["fid"]]["question"]} for r in todo]
    json.dump(items, open(ROOT / f"runs/human/{g}_todo_flagged.json", "w"), indent=1)
    W, cols, per = 240, 4, 24
    for s in range(0, len(items), per):
        chunk = items[s:s + per]; rows_n = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows_n * (W + 44)), "white"); d = ImageDraw.Draw(sheet)
        for i, it in enumerate(chunk):
            im = Image.open(ROOT / f"runs/gen/{g}/{it['image']}").convert("RGB").resize((W - 8, W - 8))
            x, y = (i % cols) * W, (i // cols) * (W + 44); sheet.paste(im, (x + 4, y + 4))
            q = it["question"].replace(" Answer yes or no.", "")
            d.text((x + 4, y + W - 2), f"[{s + i}] {q[:44]}", fill="black"); d.text((x + 4, y + W + 12), q[44:88], fill="black")
        sheet.save(ROOT / f"runs/human/{g}_todo_sheet{s // per}.png")
    print(g, "to inspect:", len(items), dict(collections.Counter(it["type"] for it in items)))
