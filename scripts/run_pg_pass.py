#!/usr/bin/env python3
"""DD-043: regenerate the PG-twin reference on a channel file under the CURRENT
(= corrected) rate function.  Fills results/ao/pg_<stem>.json with the schema
make_paper_figs_v1.py expects (stem, n, N, pg_mean, pg_sd, ao_mean, ao_sd,
pg_per_channel).  The AO side is read from the corrected AO reference JSON in
the same output dir; it must exist (run scripts/run_ao_reference.py FIRST for
the same --channels on the same --out dir).

Usage:
    python scripts/run_pg_pass.py --channels data/test_channels.npz --out results/ao
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ao_riemannian import load_channel_npz, pg_sanity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    stem = Path(a.channels).stem                    # e.g. test_channels
    ao_path = Path(a.out) / f"ao_{stem}.json"
    assert ao_path.exists(), f"corrected AO reference missing: {ao_path}\n" \
        f"run: python scripts/run_ao_reference.py --channels {a.channels} --out {a.out} FIRST"
    ao = json.load(open(ao_path, encoding="utf-8"))

    chs = load_channel_npz(a.channels)
    print(f"PG twin pass: {len(chs)} channels from {a.channels} under corrected rate", flush=True)
    rates, t0 = [], time.perf_counter()
    for i, ch in enumerate(chs):
        r = pg_sanity(ch)["rate"]
        rates.append(float(r))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(chs)}  mean-so-far {sum(rates)/len(rates):.3f}  ({el:.0f}s)", flush=True)

    import numpy as np
    out = {"stem": stem, "n": len(chs), "N": int(chs[0]["h_r"].shape[1]),
           "pg_mean": float(np.mean(rates)), "pg_sd": float(np.std(rates, ddof=1)),
           "ao_mean": float(ao.get("ao_mean", ao.get("mean", float("nan")))),
           "ao_sd": float(ao.get("ao_sd", ao.get("sd", float("nan")))),
           "pg_per_channel": rates}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"pg_{stem}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(f"PG mean {out['pg_mean']:.3f} ± {out['pg_sd']:.3f}  vs AO {out['ao_mean']:.3f}  -> {p}", flush=True)

if __name__ == "__main__":
    main()
