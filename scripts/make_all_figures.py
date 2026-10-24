"""ALL figures & the summary table. Every number comes from results/*.json — never hand-typed.
   Safe to run any day: missing inputs are skipped with a printed note."""
import sys, pathlib, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.stats import describe, paired_vs_random, seed_band

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"

def J(path):
    f = R / path
    return json.loads(f.read_text()) if f.exists() else (print(f"[skip] missing {path}") or None)

def fig_baseline_bars(baselines):
    ax = plt.figure(figsize=(6, 3.8)).gca()
    names = list(baselines); means = [baselines[k]["mean"] for k in names]
    sds   = [baselines[k]["sd"] for k in names]
    ax.bar(range(len(names)), means, yerr=sds, capsize=4, color="#4a7dbd")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("sum rate (bps/Hz)"); ax.set_title("Baselines on the frozen 500-channel test set")
    plt.tight_layout(); plt.savefig(FIGS / "fig_baselines.pdf"); plt.savefig(FIGS / "fig_baselines.png"); plt.close()

def fig_reward_band(agent):
    dfs = []
    for s in range(5):
        f = R / agent / f"seed{s}" / "metrics.json"
        if not f.exists(): print(f"[skip] {agent} seed{s} missing"); return
        dfs.append(json.loads(f.read_text()))
    band = seed_band(dfs); steps = sorted(band)
    mu  = [band[s]["mean"] for s in steps]; sdv = [band[s]["sd"] for s in steps]
    plt.figure(figsize=(6, 3.8))
    plt.plot(steps, mu, label=f"{agent} (mean of 5 seeds)")
    plt.fill_between(steps, np.array(mu) - np.array(sdv), np.array(mu) + np.array(sdv),
                     alpha=0.25, label="± SD over seeds")
    rb = J("baselines/random/eval_rates.json")
    if rb: plt.axhline(np.mean(rb["rates"]), ls="--", c="k", label="random-phase baseline")
    ab = J("ao/ao_rates.json")
    if ab: plt.axhline(np.mean(ab["rates"]), ls=":", c="r", label="AO upper bound")
    plt.xlabel("env steps"); plt.ylabel("eval sum rate (bps/Hz)")
    plt.title(f"{agent.upper()} training (identical 200k-step budget)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / f"fig_band_{agent}.pdf"); plt.savefig(FIGS / f"fig_band_{agent}.png"); plt.close()

def fig_ao_convergence():
    d = J("ao/ao_rates.json")
    if not d or "convergence" not in d: return
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(d["convergence"]); plt.xlabel("CG iteration"); plt.ylabel("mean sum rate")
    plt.title("Riemannian AO convergence (≥200 iters, tol 1e-4)")
    plt.tight_layout(); plt.savefig(FIGS / "fig_ao_convergence.pdf"); plt.close()

def fig_bits_and_csi():
    b = J("bits_sweep.json")
    if b:
        plt.figure(figsize=(5, 3.4))
        for m, ys in b["mean_rate_by_method"].items():
            plt.plot(b["bits"], ys, marker="o", label=m)
        plt.xticks(b["bits"]); plt.xlabel("phase bits B"); plt.ylabel("mean sum rate (bps/Hz)")
        plt.title("Rate under discrete phase levels (2/3/4-bit)"); plt.legend(); plt.tight_layout()
        plt.savefig(FIGS / "fig_bits.pdf"); plt.close()
    c = J("unfold/csi_sweep.json")
    if c:
        plt.figure(figsize=(5, 3.4))
        for m, ys in c["mean_rate_by_method"].items():
            plt.plot(c["eps"], ys, marker="s", label=m)
        plt.xlabel(r"CSI error $\varepsilon$"); plt.ylabel("mean sum rate (bps/Hz)")
        plt.title("Imperfect-CSI robustness (true-ch loss)"); plt.legend(); plt.tight_layout()
        plt.savefig(FIGS / "fig_csi.pdf"); plt.close()

def summary_table():
    cols = []
    rnd = J("baselines/random/eval_rates.json")
    for path, label in [("baselines/random/eval_rates.json", "Random"),
                        ("baselines/zero/eval_rates.json", "All-zero"),
                        ("baselines/no_ris/eval_rates.json", "No RIS"),
                        ("baselines/best_of_200/eval_rates.json", "Best-of-200"),
                        ("ao/ao_rates.json", "Riemannian AO"),
                        ("unfold/unfold_rates.json", "Deep unfolding"),
                        ("sac/best/eval_rates.json", "SAC (best seed)"),
                        ("td3/best/eval_rates.json", "TD3 (best seed)"),
                        ("ddpg/best/eval_rates.json", "DDPG (best seed)")]:
        d = J(path)
        if not d or "rates" not in d: continue
        row = describe(d["rates"], label)
        if rnd: row.update(paired_vs_random(d["rates"], rnd["rates"], label))
        cols.append(row)
    if not cols: print("[skip] no results yet"); return
    md = ["| Method | mean | SD | SEM | Δ vs random | p (Wilcoxon) | 95% CI of gain | %channels>random |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cols:
        md.append(f"| {c['method']} | {c['mean']:.2f} | {c.get('sd','-'):.2f} | {c.get('sem','-'):.3f} | "
                  f"{c.get('mean_gain',''):.2f} | {c.get('p_wilcoxon',''):.1e} | "
                  f"{c.get('ci95','')} | {100*c.get('frac_channels_positive',0):.0f}% |")
    (ROOT / "results" / "summary_table.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    bl = {n: J(f"baselines/{n}/eval_rates.json") for n in
          ["random","zero","no_ris","best_of_200"]}
    bl = {k: v for k, v in bl.items() if v}
    if bl: fig_baseline_bars(bl)
    for a in ["sac", "td3", "ddpg"]: fig_reward_band(a)
    fig_ao_convergence(); fig_bits_and_csi(); summary_table()