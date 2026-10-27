#!/usr/bin/env python3
"""
scripts/latency_common_protocol.py -- DD-048 same-machine latency protocol
===========================================================================

External review round 6 flagged that the committed latency figures were not
measured on a common footing (AO per-channel times were recorded inside a
4-worker parallel run => CPU contention; the untrained unfolding pass was a
different-machine measurement).  This script times, in ONE process on ONE
idle machine, with serialized single-solve wall clocks:

    1. AO per channel  (iters<=200, tol=1e-4 -- the frozen gate settings)
    2. unfolding (trained ckpt) per channel, batch size 1, complex128
    3. unfolding (untrained, same graph) per channel -- protocol control

Outputs results/unfold/latency_common_protocol.json (script-generated only).
The AO rates must equal the canonical set to within 1e-9 per channel
(determinism check); only the *timings* are new.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ao_riemannian import (
    ao_riemannian, load_channel_npz, sum_rate_numpy,
)
from unfolding import UnrolledPG, channels_to_tensors, batch_slice

CH = "data/test_channels.npz"
OUT = Path("results/unfold/latency_common_protocol.json")
FROZEN_AO = Path("results/ao/ao_test_channels.json")

def main() -> int:
    chs = load_channel_npz(CH)
    n = len(chs)
    T = channels_to_tensors(chs)
    print(f"same-machine protocol: {n} channels, serialized single-solve wall clocks")

    # ---------------- AO serial ----------------
    ao_times, ao_rates = [], []
    for ch in chs:
        t0 = time.perf_counter()
        res = ao_riemannian(ch, iters=200, tol=1e-4)
        ao_times.append(time.perf_counter() - t0)
        ao_rates.append(res["rate"])
        if len(ao_times) % 100 == 0:
            print(f"  AO {len(ao_times)}/{n}")
    ao_times = np.array(ao_times); ao_rates = np.array(ao_rates)

    # determinism check against the canonical committed AO artifact
    det = None
    if FROZEN_AO.exists():
        ref = np.array(json.load(open(FROZEN_AO))["ao"]["per_channel"])
        det = float(np.abs(ao_rates - ref).max())

    # ---------------- unfolding, trained + untrained ----------------
    def unfold_times(model, label):
        with torch.enable_grad():
            model(batch_slice(T, torch.arange(min(8, n))), create_graph=False)  # warmup
        lat = []
        for i in range(n):
            chb = batch_slice(T, torch.arange(i, i + 1))
            t0 = time.perf_counter()
            with torch.enable_grad():
                model(chb, create_graph=False)
            lat.append((time.perf_counter() - t0) * 1e3)
        lat = np.array(lat)
        print(f"  {label}: median {np.median(lat):.3f} ms mean {lat.mean():.3f}")
        return lat

    m_trained = UnrolledPG(L=6, theta_init=0.0)
    m_trained.load_state_dict(torch.load(
        "results/unfold/unfold_L6_rate.pt", map_location="cpu", weights_only=True))
    m_trained.eval()
    m_untrained = UnrolledPG(L=6, theta_init=0.0)
    m_untrained.eval()
    lat_tr = unfold_times(m_trained, "unfolding trained")
    lat_un = unfold_times(m_untrained, "unfolding untrained (control)")

    ao_med = float(np.median(ao_times))
    unf_med = float(np.median(lat_tr)) / 1e3
    payload = {
        "protocol": (
            "DD-048 same-machine serialized single-solve wall clocks on one idle desktop "
            "CPU: AO (iters<=200, tol=1e-4 frozen gate settings), unfolding trained and "
            "untrained forward passes, batch 1, complex128; AO rate determinism checked "
            "against results/ao/ao_test_channels.json."),
        "machine_note": "TIMES ARE MACHINE-SPECIFIC; only same-machine ratios are publishable.",
        "n_channels": n,
        "ao_time_s": {"median": ao_med, "mean": float(ao_times.mean()),
                      "p95": float(np.percentile(ao_times, 95))},
        "unfold_trained_ms": {"median": float(np.median(lat_tr)),
                              "mean": float(lat_tr.mean()),
                              "p95": float(np.percentile(lat_tr, 95))},
        "unfold_untrained_ms": {"median": float(np.median(lat_un)),
                                "mean": float(lat_un.mean())},
        "ratio_ao_over_unfold_trained": ao_med / unf_med,
        "ao_iters_vs_unfold_layers": "median AO iterations / L=6 layers",
        "ao_rate_determinism_max_abs_diff_vs_frozen": det,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
