import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444", "text.color": "#1F2933",
    "axes.labelcolor": "#1F2933", "legend.frameon": False})

COL, FULL = 3.31, 6.85
INK, GRAY, GRAY_L, PAPER = "#1F2933", "#8B96A3", "#D9DEE4", "#F4F6F8"
JAN, JAN_L = "#1F4E79", "#7DA3C8"
SHO, SHO_L = "#C8552B", "#EBA17F"
BAG, OMN, BLP, LUM = "#6B3F8C", "#2A9D8F", "#C9A227", "#7A9E6E"
CAP = "#E9C46A"
MODEL_COL = {"BAGEL-7B-MoT": BAG, "OmniGen2": OMN, "Janus-Pro-1B": JAN_L, "Show-o2-1.5B": SHO_L,
             "Janus-Pro-7B": JAN, "Lumina-DiMOO": LUM, "BLIP3-o-8B": BLP, "Show-o2-7B": SHO}
TEXT_COL = dict(MODEL_COL, **{"Janus-Pro-1B": "#4A78A8", "Show-o2-1.5B": "#B8643C", "BLIP3-o-8B": "#94740C", "Lumina-DiMOO": "#4F7543"})


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right" if x < 0 else "left")


def rbox(ax, x, y, w, h, text, fc, ec=GRAY, fs=6.8, lw=0.6, bold=False, tc=INK, pad=0.012, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.018", fc=fc, ec=ec, lw=lw,
                                mutation_aspect=1, transform=ax.transAxes, clip_on=False))
    ax.text(x + (w / 2 if align == "center" else 0.02), y + h / 2, text, ha=align, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, transform=ax.transAxes, linespacing=1.15)


def arrow(ax, p, q, color=GRAY, lw=0.9, style="-|>", ms=6, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, transform=ax.transAxes,
                                 connectionstyle=cs, clip_on=False, shrinkA=1, shrinkB=1))


from matplotlib.patches import Polygon, Arc, Circle
FW, FH = FULL, 3.25
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
BOOK, BOOK_D, CUP, TABLE, FRAME = "#3E6E96", "#284B6B", "#C8552B", "#C9B79C", "#FBF8F1"
UFC, UEC, AFC, AEC = "#EAF0F6", "#A9BCD0", "#FBEDE6", "#E8B49C"

def box(x, y, w, h, fc, ec, lw=0.6, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

def arr(p, q, c=GRAY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=c, lw=0.8, shrinkA=0, shrinkB=0, zorder=2))

EXIMG = plt.imread(os.path.join(FIG, "assets", "bagel_ge_0416_s1.png"))
_EX = json.load(open(os.path.join(ROOT, "data", "ci", "fig1_example.json")))["p_yes"]
EX_PYES = [round(_EX[k], 2) for k in ("fresh", "generic request", "closed loop", "closed loop", "inline request")]

def thumb(x, y, s, z=3):
    im = ax.imshow(EXIMG, extent=(x, x + s, y, y + s), zorder=z, interpolation="lanczos")
    clip = FancyBboxPatch((x, y), s, s, boxstyle=f"round,pad=0,rounding_size={0.03 * s / 0.4}", fc="none", ec="#C9C2B6", lw=0.5, zorder=z + 1)
    ax.add_patch(clip); im.set_clip_path(clip)

def mark(x, y, ok, r=0.055):
    ax.add_patch(Circle((x, y), r, fc="#2E8B57" if ok else "#B5322B", ec="none", zorder=4))
    ax.text(x, y - 0.004, "✓" if ok else "✗", ha="center", va="center", fontsize=6.2, color="white", fontfamily="DejaVu Sans", zorder=5)

def avatar(x, y, who, r=0.06):
    ax.add_patch(Circle((x, y), r, fc=UEC if who == "U" else AEC, ec="none", zorder=4))
    ax.text(x, y - 0.003, who, ha="center", va="center", fontsize=5.2, color="white", fontweight="bold", zorder=5)

ax.text(0.02, FH - 0.04, "(a)", fontsize=9, fontweight="bold", va="top")
box(0.12, 2.62, 1.98, 0.36, "#FCF3DC", CAP, lw=0.7)
ax.text(0.20, 2.89, "request", fontsize=5.6, color="#8A6D1F", style="italic", va="center")
ax.text(1.11, 2.735, "“a photo of a sandwich below a knife”", fontsize=6.8, ha="center", va="center")
arr((1.11, 2.62), (1.11, 2.47))
box(0.56, 2.19, 1.10, 0.28, PAPER, GRAY)
ax.text(1.11, 2.33, "BAGEL as generator", fontsize=6.4, ha="center", va="center")
arr((0.62, 2.19), (0.62, 2.06))
thumb(0.22, 1.22, 0.82)
ax.text(0.63, 1.15, "generated image", fontsize=5.8, ha="center", va="top", color="#5E6B78", style="italic")
ax.text(1.18, 2.00, "atomic facts", fontsize=6.4, fontweight="bold", va="center")
for k, (q, ok) in enumerate([("Is there a knife?", True), ("Is there a sandwich?", True), ("Is the sandwich\nbelow the knife?", False)]):
    yy = 1.78 - k * 0.22 - (0.03 if k == 2 else 0)
    mark(1.24, yy, ok); ax.text(1.34, yy, q, fontsize=6.0, va="center", linespacing=1.0)
ax.text(1.18, 1.10, "labels: oracle, detector, human", fontsize=5.6, color="#5E6B78", style="italic", va="center")
ORD = ["janus7b", "janus1b", "showo2_7b", "showo2_1.5b", "bagel", "blip3o", "omnigen2", "lumina"]
COV = {"bagel": "bagel blip3o janus7b lumina omnigen2 showo2_7b", "blip3o": "bagel blip3o janus7b lumina omnigen2 showo2_7b",
       "janus1b": "janus1b janus7b showo2_1.5b showo2_7b", "janus7b": "bagel blip3o janus1b janus7b lumina omnigen2 showo2_7b",
       "lumina": "janus7b lumina showo2_7b", "omnigen2": "bagel blip3o janus7b lumina omnigen2 showo2_7b",
       "showo2_1.5b": "janus1b showo2_1.5b showo2_7b janus7b", "showo2_7b": "bagel blip3o janus7b lumina omnigen2 showo2_1.5b showo2_7b"}
mx, my, cell = 0.30, 0.10, 0.085
for i, jj in enumerate(ORD):
    for j, gg in enumerate(ORD):
        run = gg in COV[jj].split()
        fc = JAN if jj == gg else ("#B9CCE0" if gg in ("janus7b", "showo2_7b") else "#DCE6F0") if run else "white"
        ax.add_patch(Rectangle((mx + j * cell, my + (7 - i) * cell), cell, cell, fc=fc, ec="white" if run else GRAY_L, lw=0.5, zorder=2))
ax.text(mx + 4 * cell, my - 0.03, "generator", ha="center", va="top", fontsize=6.0)
ax.text(mx - 0.05, my + 4 * cell, "judge", ha="right", va="center", fontsize=6.0, rotation=90)
for k, (fc, lab) in enumerate([(JAN, "own images"), ("#B9CCE0", "common others' set"), ("#DCE6F0", "other generators"), ("white", "not run")]):
    yk = 0.66 - k * 0.13
    ax.add_patch(Rectangle((1.18, yk - 0.04), 0.08, 0.08, fc=fc, ec=GRAY_L, lw=0.5, zorder=2))
    ax.text(1.31, yk, lab, fontsize=5.9, va="center")
ax.text(1.18, 0.12, "43 judge–generator pairs", fontsize=5.9, va="center", color="#5E6B78", style="italic")
ax.plot([2.22, 2.22], [0.08, FH - 0.08], color=GRAY_L, lw=0.6)

ax.text(2.30, FH - 0.04, "(b)", fontsize=9, fontweight="bold", va="top")
names = ["fresh", "authorship,\nno request", "closed loop", "request,\nno authorship", "inline request"]
REQ, QQ = "a sandwich\nbelow a knife", "Is the sandwich\nbelow the knife?"
ctx = [[("User", [("img",), ("q", QQ)])],
       [("User", [("txt", "Generate an image.")]), ("Assistant", [("img",)]), ("User", [("q", QQ)])],
       [("User", [("req", REQ)]), ("Assistant", [("img",)]), ("User", [("q", QQ)])],
       [("User", [("req", REQ), ("img",), ("q", QQ)])],
       [("User", [("img",), ("req", "Caption:\na sandwich\nbelow a knife"), ("q", QQ)])]]
LAB, PADT, PADI, GAPT = 0.10, 0.035, 0.035, 0.05
def item_h(it):
    t = it[0]
    if t == "img": return 0.36
    n = it[1].count("\n") + 1
    return {"req": 0.06 + 0.085 * n, "q": 0.03 + 0.085 * n, "txt": 0.03 + 0.085 * n}[t]
X0, CW, GAP, TOP = 2.40, 0.82, 0.07, 2.84
for k in range(5):
    x = X0 + k * (CW + GAP)
    has_req = any(it[0] == "req" for _, its in ctx[k] for it in its)
    ax.text(x + CW / 2, FH - 0.10, names[k], ha="center", va="top", fontsize=6.8, fontweight="bold", linespacing=1.0)
    ax.add_patch(Rectangle((x, TOP + 0.04), CW, 0.025, fc=CAP if has_req else "#E6E9ED", ec="none"))
    y = TOP
    for role, items in ctx[k]:
        h = LAB + PADT * 2 + sum(item_h(it) for it in items) + PADI * (len(items) - 1)
        user = role == "User"
        box(x, y - h, CW, h, UFC if user else AFC, UEC if user else AEC, lw=0.55, r=0.04)
        ax.text(x + 0.05, y - 0.06, role, fontsize=5.0, color="#5E6B78" if user else "#A0583A", style="italic", va="center", zorder=4)
        yy = y - LAB - PADT
        for it in items:
            hh = item_h(it); t = it[0]
            if t == "img": thumb(x + (CW - hh) / 2, yy - hh, hh, z=3)
            elif t == "req":
                box(x + 0.06, yy - hh, CW - 0.12, hh, "#FCF3DC", CAP, lw=0.6, r=0.03, z=3)
                ax.text(x + CW / 2, yy - hh / 2, it[1], ha="center", va="center", fontsize=5.5, linespacing=1.05, style="italic", zorder=4)
            else:
                ax.text(x + CW / 2, yy - hh / 2, it[1], ha="center", va="center", fontsize=5.7, linespacing=1.05, zorder=4)
            yy -= hh + PADI
        y -= h + GAPT
YV = 1.02
ax.text(X0, YV + 0.25, "BAGEL on this image: p(yes) and verdict", fontsize=5.9, va="center", fontweight="bold")
ax.add_patch(Rectangle((X0 + 2.62, YV + 0.235), 0.16, 0.03, fc=CAP, ec="none"))
ax.text(X0 + 2.82, YV + 0.25, "request in the judge's context", fontsize=5.8, va="center")
for k, p in enumerate(EX_PYES):
    x = X0 + k * (CW + GAP); ok = p < 0.5
    box(x, YV, CW, 0.16, "#EEF6F0" if ok else "#FBEAE8", "#2E8B57" if ok else "#B5322B", lw=0.6, r=0.04)
    ax.text(x + 0.30, YV + 0.08, f"{p:.2f}  {'no' if ok else 'yes'}", ha="center", va="center", fontsize=6.0, zorder=4)
    mark(x + CW - 0.14, YV + 0.08, ok, r=0.045)
_C5 = ["fresh", "ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline"]
_D = {k: json.load(open(os.path.join(ROOT, "data", "ci", f"decomp_{k}_human.json"))) for k in ("janus7b", "showo2_7b", "bagel")}
_own = lambda k: [_D[k][f"own_{c}"]["est"] if f"own_{c}" in _D[k] else None for c in _C5]
det_j = _own("janus7b"); det_s = _own("showo2_7b")
det_b = _own("bagel"); det_b[3] = det_b[2]
YB = 0.80
ax.text(X0, YB, "detection of own errors (five-context facts):", fontsize=5.9, va="center", fontweight="bold")
for i, (nm, c) in enumerate([("Janus-Pro-7B", JAN), ("Show-o2-7B", SHO), ("BAGEL-7B-MoT", BAG)]):
    xx = X0 + 1.66 + i * 0.78
    ax.add_patch(Rectangle((xx, YB - 0.025), 0.08, 0.05, fc=c, ec="none")); ax.text(xx + 0.11, YB, nm, fontsize=5.9, va="center", color=c)
for k in range(5):
    x = X0 + k * (CW + GAP)
    for r, (v, c) in enumerate([(det_j[k], JAN), (det_s[k], SHO), (det_b[k], BAG)]):
        yy = 0.55 - r * 0.18
        ax.add_patch(Rectangle((x, yy), CW, 0.13, fc="#F1F3F5", ec="none"))
        ax.add_patch(Rectangle((x, yy), CW * v, 0.13, fc=c, ec="none"))
        inside = v > 0.6
        ax.text(x + CW * v + (-0.03 if inside else 0.03), yy + 0.065, f"{v:.2f}", ha="right" if inside else "left", va="center",
                fontsize=7.0, color="white" if inside else INK)
fig.savefig(os.path.join(FIG, "Fig1.pdf")); plt.close(fig)

CI = os.path.join(ROOT, "data", "ci")
ld = lambda f: json.load(open(os.path.join(CI, f)))
KEY = {"BAGEL-7B-MoT": "bagel", "OmniGen2": "omnigen2", "Janus-Pro-1B": "janus1b", "Show-o2-1.5B": "showo2_1.5b",
       "Janus-Pro-7B": "janus7b", "Lumina-DiMOO": "lumina", "BLIP3-o-8B": "blip3o", "Show-o2-7B": "showo2_7b"}
did = {L: ld(f"did_{L}.json") for L in ("human", "strict", "vlm")}
judges = sorted(KEY, key=lambda j: did["human"][KEY[j]]["framing"]["est"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 2.95), gridspec_kw={"width_ratios": [2.0, 1.25], "wspace": 0.08})
ypos = np.arange(len(judges))[::-1]
for yi, j in zip(ypos, judges):
    c = MODEL_COL[j]; h = did["human"][KEY[j]]
    fr, (lo, hi) = h["framing"]["est"], h["framing"]["ci95"]
    ax1.barh(yi, fr, height=0.56, color=c, alpha=0.88, lw=0, zorder=2)
    ax1.errorbar(fr, yi, xerr=[[fr - lo], [hi - fr]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=3)
    ax1.plot(did["strict"][KEY[j]]["framing"]["est"], yi + 0.36, marker="s", ms=2.8, mfc="white", mec=INK, mew=0.55, ls="none", zorder=4)
    ax1.plot(did["vlm"][KEY[j]]["framing"]["est"], yi - 0.36, marker="^", ms=3.0, mfc="white", mec=INK, mew=0.55, ls="none", zorder=4)
    ax1.text(-0.835, yi, f"{fr:+.2f}".replace("-", "$-$"), ha="left", va="center", fontsize=7.0, color=TEXT_COL[j], fontweight="bold")
    ss = h["selfspec"]; e, (l95, h95), (l90, h90) = ss["est"], ss["ci95"], ss["ci90"]
    ax2.plot([l95, h95], [yi, yi], color=c, lw=0.8, alpha=0.7, solid_capstyle="butt")
    ax2.plot([l90, h90], [yi, yi], color=c, lw=2.8, solid_capstyle="butt")
    ax2.plot(e, yi, marker="o", ms=3.3, color="white", mec=c, mew=1.0, zorder=3)
ax1.set_yticks(ypos); ax1.set_yticklabels(judges, fontsize=7); ax1.axvline(0, color=INK, lw=0.6)
ax1.set_xlim(-0.85, 0.04); ax1.set_xticks([-0.75, -0.5, -0.25, 0]); ax1.set_xticklabels(["$-$0.75", "$-$0.50", "$-$0.25", "0"])
ax1.set_xlabel("closed-loop term"); ax1.set_ylim(-0.7, len(judges) - 0.3); ax1.tick_params(axis="y", length=0)
ax1.grid(axis="x", color=GRAY_L, lw=0.4, zorder=0)
ax2.axvspan(-0.10, 0.10, color=GRAY_L, alpha=0.55, lw=0, zorder=0); ax2.axvline(0, color=INK, lw=0.6)
ax2.set_xlim(-0.32, 0.32); ax2.set_xticks([-0.2, 0, 0.2]); ax2.set_xticklabels(["$-$0.2", "0", "0.2"])
ax2.set_yticks([]); ax2.set_ylim(ax1.get_ylim()); ax2.spines["left"].set_visible(False); ax2.set_xlabel("self-specific term")
ax2.text(0.5, 1.005, "margin $\\pm$0.10", transform=ax2.transAxes, ha="center", va="bottom", fontsize=7.0, color="#5E6B78")
ha_ = [Rectangle((0, 0), 1, 1, fc=GRAY, label="human-verified, 95% CI"), Line2D([], [], marker="s", mfc="white", mec=INK, ls="none", ms=2.8, label="strict"),
       Line2D([], [], marker="^", mfc="white", mec=INK, ls="none", ms=3.0, label="oracle-only")]
hb_ = [Line2D([], [], color=GRAY, lw=2.8, label="90% CI"), Line2D([], [], color=GRAY, lw=0.8, label="95% CI")]
fig.legend(handles=ha_, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, handlelength=1.1, handletextpad=0.4, columnspacing=0.8, fontsize=7.0, borderaxespad=0.3)
fig.legend(handles=hb_, loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=1, handlelength=1.3, handletextpad=0.4, fontsize=7.0, borderaxespad=0.3, labelspacing=0.2)
fig.text(0.01, 0.955, "(a)", fontsize=9, fontweight="bold"); fig.text(0.655, 0.955, "(b)", fontsize=9, fontweight="bold")
fig.subplots_adjust(left=0.26, right=0.985, bottom=0.23, top=0.935)
fig.savefig(os.path.join(FIG, "Fig2.pdf")); plt.close(fig)

CONDS = ["fresh", "ctx-assistant_nocaption", "ctx", "ctx-user_prompt", "ctx-caption_inline"]
ctx_short = ["fresh", "authorship,\nno request", "closed\nloop", "request,\nno authorship", "inline\nrequest"]
D3 = [("Janus-Pro-7B", "janus7b"), ("BAGEL-7B-MoT", "bagel"), ("OmniGen2", "omnigen2"),
      ("Janus-Pro-1B", "janus1b"), ("Show-o2-1.5B", "showo2_1.5b"), ("Show-o2-7B", "showo2_7b")]
fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.75), sharey=True, gridspec_kw={"hspace": 0.52, "wspace": 0.07})
x = np.arange(5); bw = 0.36
for ax, (name, k) in zip(axes.ravel(), D3):
    d = ld(f"decomp_{k}_human.json"); c = MODEL_COL[name]
    ax.axvspan(1.5, 4.5, color=CAP, alpha=0.15, lw=0, zorder=0)
    for side, off, fc, hatch in (("own", -bw / 2, c, None), ("oth", bw / 2, "white", None)):
        for i, cc in enumerate(CONDS):
            key = f"{side}_{cc}"
            if key not in d: continue
            e = d[key]["est"]; lo, hi = d[key]["ci95"]
            ax.bar(i + off, e, width=bw, color=fc, edgecolor=c, lw=0.8, zorder=2)
            ax.errorbar(i + off, e, yerr=[[e - lo], [hi - e]], fmt="none", ecolor=INK, elinewidth=0.55, capsize=1.2, zorder=3)
    if k == "bagel": ax.text(3.0, 0.06, "= closed loop\n(no roles)", ha="center", va="bottom", fontsize=5.4, color="#5E6B78", style="italic", linespacing=1.0)
    n_own = d["_n"].get("own_fresh_viol", 0); n_oth = d["_n"].get("oth_fresh_viol", 0)
    ax.set_title(name, color=TEXT_COL[name], fontweight="bold", fontsize=8, pad=3, loc="left")
    ax.text(1.0, 1.02, f"$n$ = {n_own} / {n_oth}", transform=ax.transAxes, fontsize=5.8, color="#5E6B78", ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(ctx_short, fontsize=5.9, linespacing=1.0); ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRAY_L, lw=0.4, zorder=0)
for ax in axes[:, 0]: ax.set_ylabel("detection rate")
hl = [Rectangle((0, 0), 1, 1, fc=INK, ec=INK, label="own images"),
      Rectangle((0, 0), 1, 1, fc="white", ec=INK, label="others' images (Janus-Pro-7B, Show-o2-7B)"),
      Line2D([], [], color=INK, lw=0.6, marker="_", ms=4, label="95% CI"),
      Rectangle((0, 0), 1, 1, fc=CAP, alpha=0.35, label="request in the judge's context")]
fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.5, handlelength=1.4, columnspacing=1.6, borderaxespad=0.2)
fig.subplots_adjust(left=0.065, right=0.995, bottom=0.165, top=0.94)
fig.savefig(os.path.join(FIG, "Fig3.pdf")); plt.close(fig)

DIDH = ld("did_human.json")
probe = {
    "Show-o2-7B": ((0.37, 0.39, 0.617), DIDH["showo2_7b"]["framing"]["est"]),
    "Show-o2-1.5B": ((0.53, 0.71, 0.794), DIDH["showo2_1.5b"]["framing"]["est"]),
    "Janus-Pro-7B": ((0.33, 0.48, 0.902), DIDH["janus7b"]["framing"]["est"]),
    "Janus-Pro-1B": ((0.32, 0.61, 0.820), DIDH["janus1b"]["framing"]["est"])}
quart = {
    ("Janus-Pro-7B", "fresh"): ([0.090, 0.038, 0.089, 0.059], None),
    ("Janus-Pro-7B", "closed loop"): ([0.088, 0.037, 0.086, 0.070], [0.004, 0.004, 0.003, 0.003]),
    ("Janus-Pro-7B", "inline caption"): ([0.080, 0.030, 0.055, 0.044], [0.017, 0.019, 0.039, 0.012]),
    ("Show-o2-7B", "fresh"): ([0.508, 0.559, 0.466, 0.567], None),
    ("Show-o2-7B", "closed loop"): ([0.438, 0.536, 0.455, 0.551], [0.005, 0.001, 0.001, 0.002]),
    ("Show-o2-7B", "inline caption"): ([0.447, 0.494, 0.398, 0.548], [0.100, 0.095, 0.147, 0.066])}
PERTOK = {"Show-o2-7B": 7.19, "Show-o2-1.5B": 1.59, "Janus-Pro-7B": 1.19, "Janus-Pro-1B": 1.15}
from matplotlib.colors import LogNorm, LinearSegmentedColormap
fig = plt.figure(figsize=(FULL, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.45], wspace=0.55, left=0.07, right=0.885, bottom=0.2, top=0.87)
ORD = sorted(probe, key=lambda n: probe[n][1])
ax = fig.add_subplot(gs[0])
LAB4 = {"Show-o2-7B": (0, -0.04, "center"), "Show-o2-1.5B": (0.04, 0.0, "left"), "Janus-Pro-7B": (0.04, 0.0, "left"), "Janus-Pro-1B": (0.04, 0.0, "left")}
ax.axhline(0, color=GRAY_L, lw=0.6, zorder=0)
for name, (py, fr) in probe.items():
    c = MODEL_COL[name]; mk = "o" if "Janus" in name else "s"; xv = PERTOK[name] / 10.0
    ax.scatter(xv, fr, s=36, color=c, marker=mk, zorder=3, edgecolor="white", lw=0.6)
    dx, dy, ha = LAB4[name]; ax.text(xv + dx, fr + dy, name, ha=ha, va="center", fontsize=6.2, color=TEXT_COL[name])
ax.set_xlim(0, 0.8); ax.set_ylim(-0.47, 0.04); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([0, -0.1, -0.2, -0.3, -0.4])
ax.set_yticklabels(["0", "$-$0.1", "$-$0.2", "$-$0.3", "$-$0.4"])
ax.set_xlabel("image attention per token ($\\times10^{-3}$)"); ax.set_ylabel("closed-loop term")
ax = fig.add_subplot(gs[1])
CTXC = {"fresh": ("#9AA5B1", "o"), "closed loop": (CAP, "o"), "inline request": ("#B5541E", "o")}
for yi, name in enumerate(ORD[::-1]):
    py = probe[name][0]; c = MODEL_COL[name]
    ax.plot([min(py), max(py)], [yi, yi], color=GRAY_L, lw=2.6, solid_capstyle="round", zorder=1)
    for (cx, (cc, mk)), v in zip(CTXC.items(), py):
        ax.scatter(v, yi, s=30, color=cc, marker=mk, edgecolor="white", lw=0.6, zorder=3)
    ax.text(max(py) + 0.03, yi, f"{max(py):.2f}", ha="left", va="center", fontsize=6.0, color="#5E6B78")
ax.set_yticks(range(len(ORD))); ax.set_yticklabels(ORD[::-1], fontsize=6.6)
for t, name in zip(ax.get_yticklabels(), ORD[::-1]): t.set_color(TEXT_COL[name])
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0.2, 1.02); ax.set_ylim(-0.6, len(ORD) - 0.4); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(axis="x", color=GRAY_L, lw=0.4, zorder=0); ax.set_xlabel("$p(\\mathrm{yes})$ on violated facts")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4.2, mfc=cc, mec="white", label=cx) for cx, (cc, _) in CTXC.items()],
          loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=3, fontsize=5.9, handletextpad=0.2, columnspacing=0.7, borderaxespad=0.1)
ax = fig.add_subplot(gs[2]); cax = ax.inset_axes([1.20, 0.0, 0.04, 1.0])
rows, rlab = [], []
for name in ("Janus-Pro-7B", "Show-o2-7B"):
    for cx in ("fresh", "closed loop", "inline caption"): rows.append(quart[(name, cx)][0]); rlab.append(f"image, {cx.replace('caption', 'request')}")
    rows.append(quart[(name, "inline caption")][1]); rlab.append("request tokens, inline request")
M = np.array(rows)
cmap = LinearSegmentedColormap.from_list("ink", ["#F7F4EC", "#E9C46A", "#C8552B", "#5A1F3B"])
im = ax.imshow(M, aspect="auto", cmap=cmap, norm=LogNorm(vmin=0.01, vmax=0.6))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]; ax.text(j, i, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                             color="white" if v > 0.12 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"]); ax.xaxis.set_ticks_position("top"); ax.tick_params(length=0)
ax.set_yticks(range(len(rlab))); ax.set_yticklabels(rlab, fontsize=6.0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.axhline(3.5, color="white", lw=2.5)
for y0, name in ((1.5, "Janus-Pro-7B"), (5.5, "Show-o2-7B")):
    ax.text(3.60, y0, name, rotation=270, ha="left", va="center", fontsize=6.3, color=TEXT_COL[name], fontweight="bold")
ax.set_xlabel("layer quartile", labelpad=3)
cb = fig.colorbar(im, cax=cax); cb.set_ticks([0.01, 0.03, 0.1, 0.3]); cb.set_ticklabels(["0.01", "0.03", "0.1", "0.3"])
cb.ax.tick_params(labelsize=5.8, length=2); cb.outline.set_visible(False); cb.set_label("attention mass (log scale)", fontsize=6.0, labelpad=3)
for xx, lab in zip([0.005, 0.29, 0.585], ["(a)", "(b)", "(c)"]):
    fig.text(xx, 0.93, lab, fontsize=9, fontweight="bold")
fig.savefig(os.path.join(FIG, "FigS1.pdf")); plt.close(fig)

pairs = {"BAGEL-7B-MoT": ((0.75, 0.23), (0.26, 0.74)), "Janus-Pro-7B": ((0.58, 0.42), (0.35, 0.64)), "Show-o2-7B": ((0.66, 0.34), (0.57, 0.41))}
RC = ld("reward_continuous.json"); RCK = {"BAGEL-7B-MoT": "bagel:bagel", "Janus-Pro-7B": "janus7b:janus7b", "Show-o2-7B": "showo2_7b:showo2_7b"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.65), gridspec_kw={"width_ratios": [0.72, 1.6], "wspace": 0.36})
yy = 0; ylabels = []; ypos = []
for name, (fr, cl) in pairs.items():
    c = MODEL_COL[name]
    for k, (lab, (corr, tie)) in enumerate([("fresh", fr), ("closed loop", cl)]):
        inv = max(0.0, 1 - corr - tie)
        ax1.barh(yy, corr, color=c, height=0.68, alpha=1.0 if k == 0 else 0.5, lw=0)
        ax1.barh(yy, tie, left=corr, color=GRAY_L, height=0.68, lw=0)
        ax1.barh(yy, inv, left=corr + tie, color=INK, height=0.68, lw=0)
        ax1.text(0.02, yy, f"{corr:.2f}", ha="left", va="center", fontsize=6.2, color="white" if k == 0 else INK)
        v = RC[RCK[name]]["fresh" if k == 0 else "closed"]
        ax1.plot(v, yy, marker="D", ms=3.8, mfc="white", mec=INK, mew=0.8, zorder=5)
        ylabels.append(lab); ypos.append(yy); yy -= 0.85
    ax1.text(-0.30, yy + 1.275, name, transform=ax1.get_yaxis_transform(), ha="right", va="center", fontsize=6.8, color=c, fontweight="bold")
    yy -= 0.5
ax1.set_yticks(ypos); ax1.set_yticklabels(ylabels, fontsize=6.4); ax1.tick_params(axis="y", length=0)
ax1.set_xlim(0, 1); ax1.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax1.set_xlabel("fraction of oracle-ranked pairs (own images)")
ax1.spines["left"].set_visible(False)
ph = [Rectangle((0, 0), 1, 1, fc=GRAY, label="correct (counts)"), Rectangle((0, 0), 1, 1, fc=GRAY_L, label="tied"),
      Rectangle((0, 0), 1, 1, fc=INK, label="inverted"), Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, ms=3.8, label="correct (probabilities)")]
ax1.legend(handles=ph, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=5.9, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)
RP = ld("repair.json")
_start = {}
for l in open(os.path.join(CI, "repair.txt")):
    m = re.match(r"== generator (\S+): .*start all-ok ([0-9.]+)", l)
    if m: _start[m.group(1)] = float(m.group(2))
    m = re.match(r"\s+(\S+)\s+all-ok ([0-9.]+) \[([0-9.]+),([0-9.]+)\]", l)
    if m and f"run:{m.group(1)}" not in RP:
        RP[f"run:{m.group(1)}"] = {"generator": m.group(1).split("__")[0], "all_ok": float(m.group(2)), "ci95": [float(m.group(3)), float(m.group(4))]}
for k, r in RP.items():
    if k.startswith("run:") and "start" not in r: r["start"] = _start[r["generator"]]
def run(n): return RP.get(f"run:{n}")
LAD = [("Janus-Pro-7B", 0, [("janus7b__janus7b__ctx", 1), ("janus7b__janus7b", 2), ("janus7b__showo2_7b__staged", 3)]),
       ("Show-o2-7B", 0, [("showo2_7b__showo2_7b__ctx", 1), ("showo2_7b__showo2_7b", 2), ("showo2_7b__janus7b__staged", 3)]),
       ("BAGEL", 0, [("bagel__bagel__ctx", 1), ("bagel__bagel", 2), ("bagel__bagel__ctx_thr", 4)]),
       ("BAGEL", 1, [("bagel__bagel__ctx_dq", 1), ("bagel__bagel_dq", 2)]),
       ("BAGEL", 2, [("bagel__bagel__ctx_dqt", 1), ("bagel__bagel_dqt", 2)]),
       ("OmniGen2", 2, [("omnigen2__omnigen2__ctx_dqt", 1), ("omnigen2__omnigen2_dqt", 2)]),
       ("Janus-Pro-7B", 2, [("janus7b__janus7b__ctx_dqt", 1), ("janus7b__janus7b_dqt", 2)])]
BLOCKS = ["GenEval facts", "own questions,\noracle filter", "own questions,\nopen filter"]
VER = [("start", "#C5CCD4"), ("self, closed loop", "#E0A93B"), ("self, fresh", "#3E6E96"), ("other, fresh", "#1F2933"),
       ("closed, matched threshold", "#B9802A")]
bw = 0.2; gx = []; x0 = 0.0; prev = None
for gi, (name, blk, runs) in enumerate(LAD):
    if prev is not None: x0 += 1.0 + (0.45 if blk != prev else 0.0)
    prev = blk; gx.append(x0)
    rr = [run(n) for n, _ in runs]
    vals = [((rr[0]["start"], None), 0)] + [((r["all_ok"], r["ci95"]), ci) for r, (_, ci) in zip(rr, runs)]
    for vi, (v, ci) in enumerate(vals):
        xx = x0 + (vi - (len(vals) - 1) / 2) * bw; y = v[0]
        ax2.bar(xx, y, width=bw * 0.92, color=VER[ci][1], lw=0, zorder=2, hatch="////" if ci == 4 else None, edgecolor="white" if ci == 4 else None)
        if v[1] is not None: ax2.errorbar(xx, y, yerr=[[y - v[1][0]], [v[1][1] - y]], fmt="none", ecolor=INK, elinewidth=0.55, capsize=1.2, zorder=3)
        top = v[1][1] if v[1] is not None else y
        pass
for b, lab in enumerate(BLOCKS):
    xs = [gx[i] for i, g in enumerate(LAD) if g[1] == b]
    lo, hi = min(xs) - 0.42, max(xs) + 0.42
    ax2.annotate("", xy=(lo, -0.205), xytext=(hi, -0.205), xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6), annotation_clip=False)
    ax2.text((lo + hi) / 2, -0.225, lab, transform=ax2.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color=GRAY, linespacing=1.0)
    if b: ax2.axvline(min(xs) - 0.72, color=GRAY_L, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax2.set_xlim(gx[0] - 0.5, gx[-1] + 0.5); ax2.set_xticks(gx); ax2.set_xticklabels([n.replace("-7B", "-\n7B") for n, _, _ in LAD], fontsize=6.2, linespacing=1.0)
ax2.set_ylim(0.5, 1.06); ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax2.set_ylabel("prompts with all facts satisfied")
ax2.tick_params(axis="x", length=0); ax2.grid(axis="y", color=GRAY_L, lw=0.4, zorder=0)
ax2.legend(handles=[Rectangle((0, 0), 1, 1, fc=c, label=l, hatch="////" if i == 4 else None, ec="white" if i == 4 else c) for i, (l, c) in enumerate(VER)], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
           fontsize=5.9, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)
fig.text(0.005, 0.95, "(a)", fontsize=9, fontweight="bold"); fig.text(0.40, 0.95, "(b)", fontsize=9, fontweight="bold")
fig.subplots_adjust(left=0.17, right=0.995, bottom=0.27, top=0.80)
fig.savefig(os.path.join(FIG, "Fig5.pdf")); plt.close(fig)
print("Fig1-Fig5 written")

PS = ld("pool_scatter.json"); TPF = ld("theory_probe.json")
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (key, name, col) in zip(axes, (("bagel", "BAGEL-7B-MoT", BAG), ("showo2_7b", "Show-o2-7B", SHO))):
    d = PS[key]; lam, b = -TPF[f"{key}:ctx"]["slope"]["est"], TPF[f"{key}:ctx"]["icpt"]["est"]; lim = 10
    ax.add_patch(Rectangle((-lim, 0), lim, lim, fc="#FBEFE6", ec="none", zorder=0))
    for pts, c, lab, z in ((d["satisfied"], GRAY, "satisfied", 1), (d["violated"], col, "violated", 2)):
        p = np.array(pts); ax.scatter(p[:, 0], p[:, 1], s=1.6, c=c, alpha=0.35, lw=0, zorder=z, label=lab, rasterized=True)
    x = np.linspace(-lim, lim, 2)
    ax.plot(x, x, color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.plot(x, (1 - lam) * x + b, color=INK, lw=0.9, zorder=4)
    m = b / (1 - lam); ax.axvline(-m, color=INK, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    ax.axhline(0, color=GRAY, lw=0.5, zorder=3); ax.axvline(0, color=GRAY, lw=0.5, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_title(name, color=TEXT_COL.get(name, col), fontsize=7.5, pad=2)
    ax.text(0.03, 0.97, f"$\\lambda$={lam:.2f}\n$m$={m:.2f}", transform=ax.transAxes, fontsize=7.0, va="top", ha="left", color=INK)
    ax.set_xlabel("fresh log-odds of ``yes''".replace("``", "“").replace("''", "”"), fontsize=7)
axes[0].set_ylabel("closed-loop log-odds", fontsize=7)
h = [Line2D([], [], marker="o", ls="", ms=2.5, color=BAG, label="violated"), Line2D([], [], marker="o", ls="", ms=2.5, color=GRAY, label="satisfied"),
     Line2D([], [], color=INK, lw=0.9, label="pool fit"), Line2D([], [], color=GRAY, lw=0.6, ls=(0, (2, 2)), label="no change")]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.0), fontsize=7.0, handletextpad=0.3, columnspacing=0.8)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.8)
fig.savefig(os.path.join(FIG, "Fig4.pdf")); plt.close(fig)
