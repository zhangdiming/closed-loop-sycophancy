import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/probe/attention_janus7b.jsonl")]
agg = collections.defaultdict(list)
for r in rows: agg[r["mode"]].append(r)
print(f"{'mode':20s} {'n':>4s} {'att_image':>9s} {'att_caption':>11s} {'att_question':>12s} {'img/(img+cap)':>13s} {'p_yes':>6s}")
for m in ["fresh", "assistant_nocaption", "assistant", "user_prompt", "caption_inline"]:
    rs = agg.get(m, [])
    if not rs: continue
    f = lambda k: sum(r[k] for r in rs) / len(rs)
    ratio = f("att_image") / (f("att_image") + f("att_caption") + 1e-9)
    print(f"{m:20s} {len(rs):4d} {f('att_image'):9.3f} {f('att_caption'):11.3f} {f('att_question'):12.3f} {ratio:13.3f} {f('p_yes'):6.3f}")

import numpy as np
for m in ["fresh", "assistant", "caption_inline"]:
    rs = [r for r in agg.get(m, []) if r.get("layer_img")]
    if not rs: continue
    li = np.mean([r["layer_img"] for r in rs], 0); lc = np.mean([r["layer_cap"] for r in rs], 0) if rs[0].get("layer_cap") else np.zeros_like(li)
    L = len(li); q = L // 4
    print(f"{m:20s} image mass by layer quartile: " + " ".join(f"{li[i*q:(i+1)*q].mean():.3f}" for i in range(4)) +
          "   caption: " + " ".join(f"{lc[i*q:(i+1)*q].mean():.3f}" for i in range(4)))
