#!/usr/bin/env python3
"""Draw clean pipeline figure — horizontal flow with vote_fusion branch."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10,
})

# colours
C_VGGT  = "#BBDEFB"
C_INIT  = "#C5CAE9"
C_XMEM  = "#C8E6C9"
C_SAM3  = "#FFE0B2"
C_VOTE  = "#F8BBD0"
C_DIFF  = "#D1C4E9"
C_OUT   = "#E0E0E0"
C_EDGE  = "#616161"
C_ARROW = "#757575"

fig, ax = plt.subplots(1, 1, figsize=(16, 4.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(cx, cy, w, h, title, subtitle="", color="#EEEEEE", fs=9):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.35", edgecolor=C_EDGE,
                       facecolor=color, linewidth=1.3, zorder=2)
    ax.add_patch(r)
    ax.text(cx, cy + (h/6 if subtitle else 0), title,
            ha="center", va="center", fontsize=fs, fontweight="bold",
            color="#333333", zorder=3)
    if subtitle:
        ax.text(cx, cy - h/4, subtitle, ha="center", va="center",
                fontsize=fs-2, color="#666666", zorder=3)

def arrow(x1, y1, x2, y2, rad=0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="->",
        color=C_ARROW, lw=1.6, connectionstyle=f"arc3,rad={rad}", zorder=1))

# ── layout (horizontal) ────────────────────────────────────────────────────
# Row 0 (y=80): VGGT4D → Init → .............................. → DiffuEraser → Output
# Row 1 (y=55):                    XMem → ┐
#                                         ├→ Vote Fusion
# Row 2 (y=30):                    SAM3 → ┘

# x coordinates
xv = 8    # VGGT4D
xi = 20   # Init
xx = 34   # XMem
xs = 48   # SAM3
vf = 62   # Vote Fusion
df = 76   # DiffuEraser
ou = 90   # Output

W, H = 10, 11  # box width, height
H2 = H - 2

yv = 78   # VGGT4D / Init / DiffuEraser / Output y
yt = 50   # Tracker row y
yf = 23   # Vote Fusion y

# ── boxes ──────────────────────────────────────────────────────────────────
box(xv, yv, W, H,  "VGGT4D", "Dynamic Prior", C_VGGT)
box(xi, yv, W, H,  "Init Mask", "Construction", C_INIT)
box(xx, yt, W, H2, "XMem", "fine boundaries", C_XMEM)
box(xs, yt, W, H2, "SAM3", "stable tracking", C_SAM3)
box(vf, yf, W+1, H2+1, "Vote Fusion", "intersection", C_VOTE)
box(df, yv, W, H,  "DiffuEraser", "/ ProPainter", C_DIFF)
box(ou, yv, W-2, H-2, "Inpainted", "Video", C_OUT)

# ── arrows ─────────────────────────────────────────────────────────────────
# VGGT4D → Init
arrow(xv+W/2, yv, xi-W/2, yv)

# Init → SAM3 (primary path)
arrow(xi+W/2, yv, xs-W/2, yt+1, rad=0.15)

# Init → XMem
arrow(xi+W/2, yv, xx-W/2, yt, rad=-0.08)

# XMem → Vote
arrow(xx, yt-H2/2, xx, yf+H2/2+3)
arrow(xx-0.5, yf, vf-W/2-1, yf)

# SAM3 → Vote (down)
arrow(xs, yt-H2/2, xs, yf+H2/2+3)
arrow(xs+0.5, yf, vf+W/2+1, yf)

# SAM3 → DiffuEraser (primary, direct path)
arrow(xs+W/2, yt, df-W/2, yv-2, rad=0.15)

# Vote → DiffuEraser
arrow(vf, yf+H2/2+3, vf, df-H/2+1)
arrow(vf, df-H/2-1, df-W/2, df-H/2-1)

# DiffuEraser → Output
arrow(df+W/2, yv, ou-W/2+2, yv)

# ── annotations ────────────────────────────────────────────────────────────
ax.text(xx, yt+H2/2+3.5, "refined boundaries",
        ha="center", fontsize=7.5, fontstyle="italic", color="#2E7D32")
ax.text(xs, yt+H2/2+3.5, "temporal coherence",
        ha="center", fontsize=7.5, fontstyle="italic", color="#E65100")
ax.text(vf, yf-H2/2-2, r"$\cap$ consensus",
        ha="center", fontsize=7.8, fontstyle="italic", color="#AD1457")

# ── stage labels at top ────────────────────────────────────────────────────
stages = [(xv, "Stage 1\nMask"), (xx, "Stage 2\nTrack"),
          (vf, "Vote"), (df, "Stage 3\nInpaint")]
for x, lab in stages:
    ax.text(x, 95, lab, ha="center", fontsize=7.5,
            fontweight="bold", color="#888888")

# ── save ───────────────────────────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(repo_root, "report", "figs", "pipeline.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white",
            edgecolor="none", pad_inches=0.12)
print(f"Saved {out}")
plt.close()
