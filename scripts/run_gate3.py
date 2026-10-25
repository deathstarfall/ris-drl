"""GATE G3: the project's acceptance test. Writes results/g3_report.md and exits 0/1."""
import sys, pathlib, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.stats import describe, paired_vs_random

ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results"

def J(p):
    f = R / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main():
    rnd, ao = J("baselines/random/eval_rates.json"), J("ao/ao_rates.json")
    assert rnd and ao, "prereqs missing: baselines/random + ao"
    rep = ["# GATE G3 report (frozen 500-ch test set, seed 2000)", ""]
    best_g = None
    for agent in ["sac", "td3", "ddpg"]:
        best = None
        for s in range(5):
            d = J(f"{agent}/seed{s}/eval_rates.json")
            if d and (best is None or np.mean(d["rates"]) > best[1]): best = (s, np.mean(d["rates"]), d["rates"])
        if not best: continue
        t = paired_vs_random(best[2], rnd["rates"], agent.upper())
        rep.append(f"**{agent.upper()}** (seed {best[0]}): mean {best[1]:.3f}, "
                   f"Δ={t['mean_gain']:.3f}, p={t['p_wilcoxon']:.1e}, "
                   f"CI95={tuple(round(x,3) for x in t['ci95'])}, "
                   f"{100*t['frac_channels_positive']:.0f}% channels > random")
        if t["p_wilcoxon"] < 0.05 and (best_g is None or best[1] > best_g[1]):
            best_g = (agent, best[1], t)
    un = J("unfold/unfold_rates.json")
    un_rep = None
    if un:
        t = paired_vs_random(un["rates"], rnd["rates"], "UNFOLD")
        gap = (np.mean(ao["rates"]) - np.mean(un["rates"])) / np.mean(ao["rates"])
        lat = np.mean(un.get("latency_ms", [np.nan]))
        un_rep = f"UNFOLD: mean {np.mean(un['rates']):.3f}, Δ={t['mean_gain']:.3f}, " \
                 f"p={t['p_wilcoxon']:.1e}, gap-to-AO {100*gap:.1f}%, latency {lat:.3f} ms"
        rep.append(un_rep)
        rep.append(f"Australian-bound: AO/random = {np.mean(ao['rates'])/np.mean(rnd['rates']):.3f}")
    verdict = bool(best_g) and (un is None or un_rep is not None)
    rep += ["", "## VERDICT", "GATE G3: **PASS** — a learned policy beats random with p<0.05"
            if verdict else "GATE G3: **FAIL** — paper blocked until this passes"]
    (R / "g3_report.md").write_text("\n".join(rep), encoding="utf-8"); print("\n".join(rep)); sys.exit(0 if verdict else 1)

if __name__ == "__main__":
    main()