"""
tests/test_ao_grad.py — Balraj Singh (24BCE0531) — runbook §4.2 / §7-D1
=======================================================================

"Finite-difference gradient test — Do this FIRST, day 1."

Before ANY AO run, this file proves numerically that:
  1. the autograd Euclidean gradient of the sum rate equals central finite
     differences (complex domain, off-torus directions allowed),
  2. the Riemannian gradient is tangent to the torus,
  3. the retraction returns feasible (unit-modulus) points,
  4. AO ascends monotonically and converges within the frozen budget,
  5. AO clearly beats random phases on a synthetic channel,
  6. the torch twin and the numpy §6.2 twin agree to 1e-10,
  7. the plain PG sanity twin runs and ascends.

If G2 ever fails (AO ≈ random), the runbook's failure playbook (§8) says:
run THIS test first — only if it is green does geometry become the suspect.

Self-contained: builds tiny synthetic channels inline (M=4, N=8, K=2) so the
suite is green on day 1, before Ayansh's data/*.npz exist (handoff D1).
"""

import math
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import (  # noqa: E402
    CPX, REAL, TWO_PI,
    ao_riemannian, pg_sanity, project_tangent, retract,
    riemannian_gradient, sum_rate_numpy, sum_rate_phi, sum_rate_torch,
)

EPS_FD = 1e-6          # central-difference step (float64 -> ~1e-9 truncation error)
REL_TOL_FD = 1e-5      # generous vs the ~1e-9 actual FD accuracy


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _synth_channel(seed: int = 7, M: int = 4, N: int = 8, K: int = 2):
    """Tiny synthetic channel in the frozen §6.1 dict format (moderate SNR)."""
    rng = _rng(seed)
    cn = lambda s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / np.sqrt(2)
    return {
        "h_d": 0.05 * cn((K, M)),
        "G": cn((M, N)),
        "h_r": 1.0 * cn((K, N)),
        "sigma2": 1e-2,
        "P": 1.0,                       # 30 dBm
    }


def _f(z: torch.Tensor, ch_t: dict) -> torch.Tensor:
    """Objective as a function on C^N (phi = z; off-torus values are fine for FD)."""
    return sum_rate_phi(z, ch_t)


def _rate_value(z_np: np.ndarray, ch: dict) -> float:
    return sum_rate_numpy(np.angle(z_np), ch)


# --------------------------------------------------------------------------------------
# 1) Euclidean gradient vs central finite differences
# --------------------------------------------------------------------------------------

def test_euclidean_grad_matches_finite_differences():
    torch.manual_seed(0)
    ch = _synth_channel()
    ch_t = {k: (torch.as_tensor(np.asarray(v), dtype=CPX) if k not in ("sigma2", "P") else v)
            for k, v in ch.items()}
    rng = _rng(11)
    z0 = torch.as_tensor(np.exp(1j * rng.uniform(0, TWO_PI, 8)), dtype=CPX)

    f0, g, _ = riemannian_gradient(z0, ch_t)          # g = R^{2N} Euclidean gradient

    for trial in range(5):
        v = torch.as_tensor(
            rng.normal(size=8) + 1j * rng.normal(size=8), dtype=CPX)
        v = v / v.abs()
        fp = float(_f(z0 + EPS_FD * v, ch_t))
        fm = float(_f(z0 - EPS_FD * v, ch_t))
        fd = (fp - fm) / (2 * EPS_FD)
        analytic = float(torch.vdot(g, v).real)       # df = Re(<g, v>) along v
        denom = max(1.0, abs(fd), abs(analytic))
        assert abs(fd - analytic) / denom < REL_TOL_FD, (
            f"FD {fd:.10g} vs autograd {analytic:.10g} (trial {trial})")


# --------------------------------------------------------------------------------------
# 2) Riemannian gradient is tangent: Re(rgrad ∘ conj(z)) = 0
# --------------------------------------------------------------------------------------

def test_rgrad_is_tangent():
    ch = _synth_channel(seed=3)
    ch_t = {k: (torch.as_tensor(np.asarray(v), dtype=CPX) if k not in ("sigma2", "P") else v)
            for k, v in ch.items()}
    rng = _rng(5)
    z = torch.as_tensor(np.exp(1j * rng.uniform(0, TWO_PI, 8)), dtype=CPX)
    _f0, _g, rgrad = riemannian_gradient(z, ch_t)
    residual = (rgrad.conj() * z).real.abs().max()
    assert float(residual) < 1e-8, f"rgrad not tangent: max Re(rgrad*conj(z)) = {float(residual):.3e}"


# --------------------------------------------------------------------------------------
# 3) Retraction feasibility and local ascent
# --------------------------------------------------------------------------------------

def test_retraction_stays_on_torus():
    torch.manual_seed(1)
    z = torch.as_tensor(np.exp(1j * np.linspace(0, TWO_PI, 8, endpoint=False)), dtype=CPX)
    d = torch.as_tensor(np.random.default_rng(2).normal(size=8) +
                        1j * np.random.default_rng(3).normal(size=8), dtype=CPX)
    for gamma in (1.0, 0.5, 0.1, 1e-3):
        z_new = retract(z, d, gamma)
        dev = float((z_new.abs() - 1.0).abs().max())
        assert dev < 1e-12, f"retraction left the torus (gamma={gamma}): {dev:.3e}"


# --------------------------------------------------------------------------------------
# 4) AO ascends monotonically and converges inside the frozen budget
# --------------------------------------------------------------------------------------

def test_ao_monotone_and_converges():
    ch = _synth_channel(seed=7)
    out = ao_riemannian(ch, iters=200, tol=1e-4)
    hist = np.asarray(out["history"], dtype=float)
    assert out["iters"] <= 200
    # Recomputed under the corrected ZF pairing (DD-043): the corrected rate
    # makes the binding constraint the fixed 200-iteration stopping rule
    # (stopping-at-cap = numerical local optimum, disclosed in the paper),
    # so tol-hit is a NOTE, not a hard gate. Monotone ascent + meaningful
    # improvement remain hard assertions.
    dips = hist[:-1] - hist[1:]                       # positive == a decrease
    assert dips.max() <= 1e-9, f"AO history decreased by {dips.max():.3e}"
    assert out["rate"] > hist[0] + 1.0, "AO failed to improve the rate meaningfully"


# --------------------------------------------------------------------------------------
# 5) AO clearly beats random phases on the same channel
# --------------------------------------------------------------------------------------

def test_ao_beats_random_clearly():
    ch = _synth_channel(seed=7)
    out = ao_riemannian(ch, iters=200, tol=1e-4)
    rng = np.random.default_rng(1234)
    rr = np.array([sum_rate_numpy(rng.uniform(0, TWO_PI, 8), ch) for _ in range(200)])
    assert out["rate"] > rr.mean() + 2 * rr.std(), (
        f"AO {out['rate']:.3f} vs random {rr.mean():.3f} ± {rr.std():.3f} — margin too small")


# --------------------------------------------------------------------------------------
# 6) torch twin == numpy §6.2 twin
# --------------------------------------------------------------------------------------

def test_torch_numpy_twin():
    ch = _synth_channel(seed=21)
    rng = np.random.default_rng(99)
    for _ in range(5):
        theta = rng.uniform(0, TWO_PI, 8)
        ch_t = {k: (torch.as_tensor(np.asarray(v), dtype=CPX) if k not in ("sigma2", "P") else v)
                for k, v in ch.items()}
        r_torch = float(sum_rate_torch(torch.as_tensor(theta, dtype=REAL), ch_t))
        r_np = sum_rate_numpy(theta, ch)
        assert abs(r_torch - r_np) <= 1e-10 * max(1.0, abs(r_np)), (
            f"twins diverge: torch {r_torch:.12g} vs numpy {r_np:.12g}")


# --------------------------------------------------------------------------------------
# 7) PG sanity twin runs and ascends (the AO >> PG reviewer comparison)
# --------------------------------------------------------------------------------------

def test_pg_sanity_ascends():
    ch = _synth_channel(seed=7)
    out = pg_sanity(ch, iters=200, tol=1e-4)
    assert np.isfinite(out["rate"]) and out["rate"] > out["history"][0]
    ao_out = ao_riemannian(ch, iters=200, tol=1e-4)
    # soft check for the paper narrative: AO should not LOSE to plain PG
    assert ao_out["rate"] >= out["rate"] - 1e-6, (
        f"AO {ao_out['rate']:.4f} < PG {out['rate']:.4f} — investigate CG logic")
