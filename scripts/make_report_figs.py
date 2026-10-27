import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = r"C:\code\ris-drl\paper\figs"

# ---- Figure 1: unified framework schematic -----------------------------------
casc = (0.02, 0.02, 0.20, 0.20, "Riemannian AO\nreference\n(offline, eval only)", "consumes full CSI")
boxes = [
    (0.02, 0.30, 0.20, 0.44, "Channel\nrealizations\n($|h_r|,\\angle h_r, |G|, |h_d|$)", None),
    (0.27, 0.70, 0.20, 0.24, "DRL policies\nSAC / TD3 / DDPG\n(continuous)",
     "observes partial stack:\n1,056 reals (stress: 1,600)"),
    (0.27, 0.08, 0.20, 0.24, "Codebook DQN\n$B$ = 2 / 3 / 4 bits",
     "observes partial stack:\n1,056 reals (stress: 1,600)"),
    (0.27, 0.41, 0.20, 0.22, "Deep unfolding\n$L=6$ unrolled PG steps", "consumes full CSI"),
    (0.52, 0.40, 0.19, 0.24, "$\\theta \\in [0,2\\pi)^{64}$\nphase action", None),
    (0.75, 0.36, 0.22, 0.32, "Shared scorer\nZF sum rate\n(SINR = SNR)\nfrozen 500 ch.", None),
]
fig, ax = plt.subplots(figsize=(9.6, 3.9))
ax.axis("off")
for x, y, w, h, label, ann in boxes + [casc]:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="#eef3fb", ec="#3a5a8c", lw=1.2))
    if ann:
        # main label sits slightly high; the red annotation lives inside the box, never overlapping an arrow
        ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center", fontsize=8.2)
        ax.text(x + w / 2, y + h * 0.26, ann, ha="center", va="center",
                fontsize=6.6, color="#a03030", style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)
def arrow(a, b, rad=0.0):
    x1, y1, w1, h1, _, _ = a; x2, y2, w2, h2, _, _ = b
    ax.add_patch(FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=12, color="#3a5a8c", lw=1.1))
arrow(boxes[0], boxes[1], 0.15)
arrow(boxes[0], boxes[2], -0.15)
arrow(boxes[0], boxes[3])
for b in boxes[1:4]:
    arrow(b, boxes[4], 0.0)
arrow(boxes[4], boxes[5])
# AO joins via the free right-hand gap and the corridor between DQN (top 0.32) and
# unfolding (bottom 0.41); drawn as plain segment + elbow to avoid crossing boxes.
ax.add_patch(FancyArrowPatch((casc[0] + casc[2], casc[1] + casc[3] * 0.6),
                             (0.245, casc[1] + casc[3] * 0.6),
                             arrowstyle="-", mutation_scale=1, color="#a03030", lw=1.1))
ax.add_patch(FancyArrowPatch((0.245, casc[1] + casc[3] * 0.6), (0.245, 0.365),
                             arrowstyle="-", mutation_scale=1, color="#a03030", lw=1.1))
ax.add_patch(FancyArrowPatch((0.245, 0.365), (boxes[5][0], 0.38),
                             arrowstyle="-|>", mutation_scale=12, color="#a03030", lw=1.1))
ax.text(0.5, 0.95, "All methods share one evaluation entry point, one frozen test set, one rate function - "
        "but not the same channel information (red notes mark what each method consumes)",
        ha="center", fontsize=8.4, style="italic")
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
