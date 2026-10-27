import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = r"C:\code\ris-drl\paper\figs"

# ---- Figure 1: unified framework schematic -----------------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.2))
ax.axis("off")
boxes = [
    (0.02, 0.30, 0.20, 0.44, "Channel\nrealizations\n($|h_r|,\\angle h_r, |G|, |h_d|$)"),
    (0.27, 0.62, 0.20, 0.30, "DRL policies\nSAC / TD3 / DDPG\n(continuous)"),
    (0.27, 0.12, 0.20, 0.30, "Codebook DQN\n$B$ = 2 / 3 / 4 bits"),
    (0.27, 0.37, 0.20, 0.30, "Deep unfolding\n$L=6$ unrolled\nPG steps"),
    (0.52, 0.37, 0.19, 0.30, "$\\theta \\in [0,2\\pi)^{64}$\nphase action"),
    (0.75, 0.37, 0.22, 0.30, "Shared scorer\nZF sum rate\n(SINR = SNR)\nfrozen 500 ch."),
]
casc = (0.02, 0.02, 0.20, 0.20, "Riemannian AO\nreference\n(offline, eval only)")
for x, y, w, h, label in boxes + [casc]:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="#eef3fb", ec="#3a5a8c", lw=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)
def arrow(a, b, rad=0.0):
    x1, y1, w1, h1, _ = a; x2, y2, w2, h2, _ = b
    ax.add_patch(FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=12, color="#3a5a8c", lw=1.1))
def arrow_lab(a, b, label, rad=0.0, dy=0.055):
    """Arrow with a small label at the midpoint (visually distinguishes what each method consumes)."""
    x1, y1, w1, h1, _ = a; x2, y2, w2, h2, _ = b
    mx, my = (x1 + w1 + x2) / 2, (y1 + h1 / 2 + y2 + h2 / 2) / 2 + dy
    ax.add_patch(FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=12, color="#3a5a8c", lw=1.1))
    ax.text(mx, my, label, ha="center", va="center", fontsize=7.0, color="#a03030")
OBS = "observes 1,056 reals\n(stress: 1,600)"
arrow_lab(boxes[0], boxes[1], OBS, 0.15, dy=0.045)
arrow_lab(boxes[0], boxes[2], OBS, -0.15, dy=-0.075)
arrow_lab(boxes[0], boxes[3], "uses full CSI")
for b in boxes[1:4]:
    arrow(b, boxes[4], 0.0)
arrow(boxes[4], boxes[5])
arrow_lab(casc, boxes[5], "uses full CSI", -0.25, dy=-0.085)
ax.text(0.5, 0.95, "All methods share one evaluation entry point, one frozen test set, one rate function - "
        "but not the same channel information",
        ha="center", fontsize=8.6, style="italic")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "unified_framework.png"), dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure 2: CSI robustness (from csi_sweep.json) ---------------------------
d = json.load(open(os.path.join(r"C:\code\ris-drl", "results/unfold/csi_sweep.json")))
eps = d["eps"]; m = d["mean_rate_by_method"]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for key, label, mk in (("ao", "AO reference", "o"), ("unfolding", "Deep unfolding", "s"),
                       ("random", "Random phases", "^")):
    if key in m:
        ax.plot(eps, m[key], marker=mk, ms=5, lw=1.6, label=label)
ax.set_xlabel(r"channel-state error fraction $\varepsilon$")
ax.set_ylabel("mean sum rate (bps/Hz)")
ax.grid(alpha=0.35); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "csi_robustness.png"), dpi=200)
plt.close(fig)
print("figures written")
