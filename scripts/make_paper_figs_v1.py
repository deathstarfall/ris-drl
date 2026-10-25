# scripts/make_paper_figs_v1.py -- render the 6 figure PDFs expected by
# paper/sections/sec5b (IfFileExists self-healing placeholders), v1.0.1.
# Every number comes from a committed artifact (DD-018 rule). Missing
# artifact -> raises and aborts; nothing invented.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})

def jload(rel):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"missing artifact: {rel}")
    return json.loads(p.read_text(encoding="utf-8-sig"))

def save(fig, name):
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")

# 1) ao_convergence.pdf -- AO progress over iterations (nan-mean over 500 ch)
z = np.load(ROOT / "results/ao/ao_test_channels_history.npz")
hist = z["history"]                     # (500, 202)
curve = np.nanmean(hist, axis=0)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot(np.arange(curve.size), curve, lw=1.4, color="tab:blue")
ax.set_xlabel("iteration")
ax.set_ylabel("mean sum-rate (bps/Hz)")
ax.set_title("Riemannian AO: mean progress over 500 channels")
ax.grid(alpha=0.3)
save(fig, "ao_convergence.pdf")

# 2) csi_sweep.pdf + 6) csi-robustness.pdf -- same artifact, two views
d = jload("results/unfold/csi_sweep.json")
eps = d["eps"]
rates = d["mean_rate_by_method"]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for m, r in rates.items():
    ax.plot(eps, r, marker="o", ms=3.5, lw=1.2, label=m)
ax.set_xlabel(r"CSI error $\varepsilon$ (channel sd)")
ax.set_ylabel("mean sum-rate (bps/Hz)")
ax.set_title("CSI-robustness sweep (500 channels, seed 0)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "csi_sweep.pdf")

fig, ax = plt.subplots(figsize=(3.5, 2.4))
for m, r in rates.items():
    loss = 10.0 * np.log10(np.asarray(r) / r[0])
    ax.plot(eps, loss, marker="s", ms=3.5, lw=1.2, label=m)
ax.set_xlabel(r"CSI error $\varepsilon$ (channel sd)")
ax.set_ylabel("rate loss vs $\\varepsilon=0$ (dB)")
ax.set_title("Degradation in dB (same sweep artifact)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "csi-robustness.pdf")

# 3) generalization.pdf -- test vs unseen (gen channels) per method
g = jload("results/ao/generalization.json")["methods"]
names = ["ao", "unfolding", "random"]
labels = ["AO", "Unfolding", "Random"]
test = [g[m]["test"]["mean"] for m in names]
unseen = [g[m]["unseen"]["mean"] for m in names]
x = np.arange(len(names)); w = 0.36
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.bar(x - w/2, test, w, label="test (in-distribution)")
ax.bar(x + w/2, unseen, w, label="unseen (generation shift)")
for xi, v in zip(x - w/2, test): ax.text(xi, v + 0.4, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x + w/2, unseen): ax.text(xi, v + 0.4, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(x, labels); ax.set_ylabel("mean sum-rate (bps/Hz)")
ax.set_title("Generalization: test vs unseen-generation channels")
ax.legend(); ax.grid(alpha=0.3, axis="y")
save(fig, "generalization.pdf")

# 4) ao_pg_gap.pdf -- PG vs AO per-channel distribution on test set
pg = jload("results/ao/pg_test_channels.json")
pg_pc = np.asarray(pg["pg_per_channel"], dtype=float)
ao_j = jload("results/ao/ao_test_channels.json")
if isinstance(ao_j, dict) and "rates" in ao_j:
    ao_pc = np.asarray(ao_j["rates"], dtype=float)
else:  # fallback: last AO iteration of the history artifact
    ao_pc = z["history"][:, -1]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.hist(pg_pc, bins=40, alpha=0.65, label=f"PG (mean {pg['pg_mean']:.3f})")
ax.hist(ao_pc, bins=40, alpha=0.65, label=f"AO (mean {pg['ao_mean']:.3f})")
ax.axvline(pg["pg_mean"], color="tab:blue", ls="--", lw=1)
ax.axvline(pg["ao_mean"], color="tab:orange", ls="--", lw=1)
ax.set_xlabel("per-channel sum-rate (bps/Hz)")
ax.set_ylabel("# channels")
ax.set_title("AO vs PG twin on the frozen test set")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "ao_pg_gap.pdf")

# 5) loss_ablation.pdf -- unfolding loss ablation (raw artifact values)
order = ["unfold_L6_rate", "unfold_L6_circular", "unfold_L6_mse"]
vals = []
for n in order:
    m = jload(f"results/unfold/{n}/metrics.json")
    vals.append(m["final_test_rate"])
fig, ax = plt.subplots(figsize=(3.5, 2.4))
bars = ax.bar(["sum-rate", "circular", "MSE"], vals, width=0.5, color=["tab:green", "tab:orange", "tab:red"])
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("test rate with zero phase (bps/Hz)")
ax.set_title("Unfolding loss ablation (L=6, lr=0.01, seed 0)")
ax.grid(alpha=0.3, axis="y")
save(fig, "loss_ablation.pdf")

print("ALL 6 compat figures written.")
