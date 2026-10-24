#!/usr/bin/env python3
"""
scripts/run_gate2.py — Balraj Singh (24BCE0531) — runbook §2 G2 / §7 D3–D5
==========================================================================

GATE G2 — the make-or-break AO-bound gate.  Runs Riemannian AO on the FROZEN
test channels and prints the verdict with the exact frozen criteria:

    AO/random >= 1.3 mean ratio;  AO converges by iter <= 200

Outputs (results/ao/, script-generated only — NEVER hand-edit):
    ao_<stem>.json            per-channel AO rates + random/best-of-200 references + summary
    ao_<stem>_thetas.npy      (n_channels, N) AO phases  -> quantization / CSI / unfolding ablations
    ao_<stem>_history.npz     per-channel convergence curves (G2 figure data)
    pg_<stem>.json            plain projected-gradient twin (AO >> PG reviewer comparison)
    gate2_verdict.txt         the printed verdict block

Usage:
    # D2 overnight smoke (50 channels, ~minutes):
    python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --max-channels 50
    # D3-D5 full gate (500 channels; use --workers for wall-clock):
    python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --workers 4

If G2 FAILS: run tests/test_ao_grad.py FIRST (§8 playbook).  Only if that suite
is green does geometry become the suspect (-> 30-min link-budget meeting, log it).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import (  # noqa: E402
    ao_riemannian, best_of_random, load_channel_npz, pg_sanity,
    random_rates, run_ao_dataset, sum_rate_numpy,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 — AO bound gate (Balraj)")
    ap.add_argument("--channels", default="data/test_channels.npz",
                    help="frozen test npz (Ayansh's D1 handoff; NEVER regenerated)")
    ap.add_argument("--out", default="results/ao")
    ap.add_argument("--iters", type=int, default=200, help="frozen cap (runbook §4.2)")
    ap.add_argument("--tol", type=float, default=1e-4, help="frozen ||rgrad||_F stop")
    ap.add_argument("--n-random", type=int, default=200,
                    help="random draws per channel for the reference mean")
    ap.add_argument("--max-channels", type=int, default=0, help="0 = all (smoke runs use 50)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random baseline")
    ap.add_argument("--skip-pg", action="store_true", help="skip the PG sanity twin")
    args = ap.parse_args()

    ch_path = Path(args.channels)
    if not ch_path.exists():
        print(f"ERROR: {ch_path} not found.  G2 needs Ayansh's frozen test set (handoff D1):\n"
              f"  Ayansh commits data/test_channels.npz (500 ch, seed 2000) — see runbook §4.4.")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = ch_path.stem

    chs = load_channel_npz(ch_path)
    if args.max_channels > 0:
        chs = chs[: args.max_channels]
    n = len(chs)
    N = np.asarray(chs[0]["h_r"]).shape[1]
    print(f"G2 run: {n} channels x N={N}, iters<={args.iters}, tol={args.tol}, "
          f"workers={args.workers}, seed={args.seed}")

    # --- AO on the saved channels -----------------------------------------------
    t0 = time.perf_counter()
    ao_res = run_ao_dataset(chs, iters=args.iters, tol=args.tol, workers=args.workers)
    wall = time.perf_counter() - t0

    ao_rates = np.array([r["rate"] for r in ao_res])
    ao_iters = np.array([r["iters"] for r in ao_res])
    ao_times = np.array([r["time_s"] for r in ao_res])
    tol_hits = np.array([r["converged"] for r in ao_res])
    ao_rgrad = np.array([r["rgrad_norm"] for r in ao_res])

    # --- references on the SAME saved channels (paired discipline, runbook §1.5) --
    rng = np.random.default_rng(args.seed)
    rand_means = np.array([random_rates(ch, rng, args.n_random).mean() for ch in chs])
    rand_all = np.concatenate([random_rates(ch, rng, args.n_random) for ch in chs])
    best200 = np.array([best_of_random(ch, rng, args.n_random) for ch in chs])

    # --- PG sanity twin -----------------------------------------------------------
    pg_rates = None
    if not args.skip_pg:
        print("PG sanity twin (AO >> PG is the reviewer comparison)...")
        pg_res = [pg_sanity(ch, iters=args.iters, tol=args.tol) for ch in chs]
        pg_rates = np.array([r["rate"] for r in pg_res])
        np.save(out / f"pg_{stem}_rates.npy", pg_rates)
        (out / f"pg_{stem}.json").write_text(json.dumps({
            "stem": stem, "n": n, "N": N,
            "pg_mean": float(pg_rates.mean()), "pg_sd": float(pg_rates.std(ddof=1)),
            "ao_mean": float(ao_rates.mean()), "ao_sd": float(ao_rates.std(ddof=1)),
            "pg_per_channel": pg_rates.tolist(),
        }, indent=2))

    # --- frozen criteria -----------------------------------------------------------
    ratio = float(ao_rates.mean() / rand_means.mean())
    iters_ok = bool(ao_iters.max() <= args.iters and np.median(ao_iters) <= args.iters)
    frac_ge_random = float((ao_rates > rand_means).mean())
    pass_gate = bool(ratio >= 1.3) and iters_ok

    lines = []
    lines.append("================ G2 — AO BOUND GATE (Balraj) ================")
    lines.append(f" channels evaluated      : {n}  (N={N}, from {ch_path})")
    lines.append(f" AO mean rate            : {ao_rates.mean():8.3f} ± {ao_rates.std(ddof=1):.3f} bps/Hz")
    lines.append(f" random mean rate        : {rand_means.mean():8.3f} ± {rand_means.std(ddof=1):.3f} bps/Hz"
                 f"   (mean of {args.n_random} draws/channel, seed {args.seed})")
    lines.append(f" best-of-200 mean rate   : {best200.mean():8.3f} ± {best200.std(ddof=1):.3f} bps/Hz")
    if pg_rates is not None:
        lines.append(f" PG-twin mean rate       : {pg_rates.mean():8.3f} ± {pg_rates.std(ddof=1):.3f} bps/Hz"
                     f"   (AO/PG = {ao_rates.mean() / pg_rates.mean():.3f})")
    lines.append("-" * 62)
    lines.append(f" AO / random mean ratio  : {ratio:8.3f}    [criterion >= 1.30]")
    lines.append(f" AO > random per-channel : {100 * frac_ge_random:7.1f} %    [informational]")
    lines.append(f" iters to terminate       : median {np.median(ao_iters):.0f}, "
                 f"max {ao_iters.max()} (cap {args.iters})    [criterion <= {args.iters}]")
    lines.append(f" termination quality      : tol-hit {100 * tol_hits.mean():5.1f} %  | "
                 f"numerical-stall {100 * (1 - tol_hits.mean()):5.1f} %  | "
                 f"||rgrad|| at stop: median {np.median(ao_rgrad):.2e}, max {ao_rgrad.max():.2e}")
    lines.append(f" wall time               : {wall:.1f} s total | median {np.median(ao_times):.3f} s/channel"
                 f"   (AO is the seconds-scale upper bound for the G3 latency table)")
    lines.append("-" * 62)
    lines.append(f" G2 VERDICT: {'PASS' if pass_gate else 'FAIL'}")
    if not pass_gate:
        if ratio < 1.3:
            lines.append("   -> ratio criterion failed. §8 playbook: run tests/test_ao_grad.py")
            lines.append("      FIRST; only if green, convene the geometry meeting")
            lines.append("      (users deeper in RIS zone / weaker direct link) and log it.")
        if not iters_ok:
            lines.append("   -> convergence criterion failed: inspect Armijo/retraction logic.")
    lines.append("=" * 62)
    report = "\n".join(lines)
    print(report)
    (out / "gate2_verdict.txt").write_text(report + "\n")

    # --- artifacts -----------------------------------------------------------------
    np.save(out / f"ao_{stem}_thetas.npy", np.stack([r["theta"] for r in ao_res]))
    hlen = max(len(r["history"]) for r in ao_res) + 1
    hist = np.full((n, hlen), np.nan)
    for i, r in enumerate(ao_res):
        h = np.asarray(r["history"], dtype=float)
        hist[i, :len(h)] = h
        hist[i, len(h):] = h[-1]                     # pad with the final value
    np.savez_compressed(out / f"ao_{stem}_history.npz", history=hist,
                        iters=ao_iters, tol_hit=tol_hits)

    (out / f"ao_{stem}.json").write_text(json.dumps({
        "stem": stem, "n": n, "N": N, "seed": args.seed,
        "config": {"iters": args.iters, "tol": args.tol, "n_random": args.n_random},
        "ao": {"mean": float(ao_rates.mean()), "sd": float(ao_rates.std(ddof=1)),
               "median_iters": float(np.median(ao_iters)), "max_iters": int(ao_iters.max()),
               "tol_hit_frac": float(tol_hits.mean()),
               "rgrad_norm_median": float(np.median(ao_rgrad)),
               "rgrad_norm_max": float(ao_rgrad.max()),
               "median_time_s": float(np.median(ao_times)),
               "per_channel": ao_rates.tolist()},
        "random": {"mean": float(rand_means.mean()), "sd": float(rand_means.std(ddof=1)),
                   "pooled_sd": float(rand_all.std(ddof=1)), "n_draws": args.n_random},
        "best_of_random": {"mean": float(best200.mean()), "sd": float(best200.std(ddof=1))},
        "criteria": {"ao_random_ratio": ratio, "ratio_min": 1.3,
                     "iters_ok": iters_ok, "verdict": "PASS" if pass_gate else "FAIL"},
    }, indent=2))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
