"""GATE G1: prove the design is noise-limited and phases matter, BEFORE anyone trains.
   Pass rules (DD-010, constants editable only via a logged design decision):
   (i)   mean(random) - mean(no_ris)   >= 1.5 bps/Hz
   (ii)  mean(bo200) - mean(random)    >= 0.20 * (mean(bo200) - mean(no_ris))
   (iii) median |Δrate| when ONE element flips by π/2 > 1% of mean(random)"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.channel import load_dataset
from src.baselines import run_baselines, _random_theta
from src.precoder import sum_rate

def main():
    chs = load_dataset("data/test_channels.npz")
    res = run_baselines(chs, draw_seed=123, callback=lambda i, m:
                        print(f"\r  {i+1}/{len(chs)}", end="") if i % 50 == 0 else None)
    rng = np.random.default_rng(123)
    deltas = []
    for i, ch in enumerate(chs):
        N = ch["h_r"].shape[1]
        th = _random_theta(rng, N)
        th2 = th.copy(); th2[0] += np.pi / 2          # flip element 0 by π/2
        deltas.append(abs(sum_rate(th2, ch) - res["random"][i]))
        if i % 50 == 0: print(f"\r  sens {i+1}/{len(chs)}", end="")
    print()
    d = {k: float(np.mean(v)) for k, v in res.items()}
    med = float(np.median(deltas))
    print("\n================= GATE G1 =================")
    for k, v in d.items(): print(f"  {k:12s} mean rate = {v:7.3f} bps/Hz  (n={len(res[k])}, "
                                 f"SD={np.std(res[k], ddof=1):.3f})")
    print(f"  per-element π/2 flip |Δrate| median = {med:.4f}")
    g1a = d["random"] - d["no_ris"] >= 1.5
    g1b = d["best_of_200"] - d["random"] >= 0.20 * (d["best_of_200"] - d["no_ris"])
    g1c = med > 0.01 * d["random"]
    print(f"  (i)  random - no_ris = {d['random']-d['no_ris']:.3f} >= 1.5 ... {'PASS' if g1a else 'FAIL'}")
    print(f"  (ii) bo200 margin   = {d['best_of_200']-d['random']:.3f} ... {'PASS' if g1b else 'FAIL'}")
    print(f"  (iii) sensitivity    = {med:.4f} ... {'PASS' if g1c else 'FAIL'}")
    verdict = g1a and g1b and g1c
    print("===========================================")
    print("GATE G1:", "PASS ✓ — Shreyansh may start campaigns" if verdict else
          "FAIL ✗ — D0 rules: tune ONLY user-zone/p_tx (docs/link_budget.md), rerun")
    sys.exit(0 if verdict else 1)

if __name__ == "__main__":
    main()