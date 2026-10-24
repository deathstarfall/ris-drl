"""
src/ao_riemannian.py — Balraj Singh (24BCE0531) — runbook §4.2 / §6.6 / G2 owner
================================================================================

RIS phase-shift optimization on the torus manifold  M = {z in C^N : |z_i| = 1}.

This module is self-contained on purpose: it does NOT import Ayansh's
src/channel.py or src/precoder.py, so PR2 (Balraj's branch feat/opt-bound) can be
reviewed and pytest-greened in parallel with PR1.  The frozen interface contract
(runbook §1.1 channel dict, §6.2 sum-rate formula) is reproduced EXACTLY here and
cross-checked by tests/test_ao_grad.py::test_torch_numpy_twin.  At integration,
either this torch rate replaces/moves into precoder.py or both are kept — that is
a one-reviewer decision, logged in docs/design_decisions.md (entry B1).

Contents
--------
  sum_rate_phi(phi, ch_t)       differentiable torch sum rate, complex-unit parametrization
  sum_rate_torch(theta, ch_t)   same objective in the angle parametrization (frozen API)
  sum_rate_numpy(theta, ch)     numpy reference twin of runbook §6.2 (TEST-ONLY use lives in tests)
  riemannian_gradient(...)      autograd Euclidean gradient + tangent projection (§4.2 formula)
  ao_riemannian(ch, ...)        Riemannian CG on the torus: Fletcher–Reeves, Armijo, retraction
  pg_sanity(ch, ...)            plain projected-gradient twin (the AO >> PG reviewer comparison)
  run_ao_dataset(...)           batch runner used by scripts/run_gate2.py / run_ao_reference.py
  load_channel_npz(path)        loader for the frozen npz datasets (§6.1 save format)
  random_rates(ch, rng, n)      random-phase helper for AO-side scripts (canonical baselines
                                live in Ayansh's src/baselines.py — same RNG protocol)

Frozen-design compliance (runbook §2 G2, §4.2):
  * iters <= 200, stop ||rgrad||_F < 1e-4, Armijo backtracking, FR-CG with periodic resets
  * runs on SAVED channels only (npz in, theta/rate out — no channel generation here)
  * objective = SUM RATE under ZF (fix #1), autograd gradients (fix #5 discipline)
  * init theta = 0 — identical to the unfolding init (fair AO-vs-unfolding comparison)

dtype policy: float64 / complex128 throughout.  Finite-difference tests agree with
autograd to ~1e-6 relative, which float32 cannot sustain.  Cast to float32 only for
inference-latency experiments (documented in docs/ao_math.md).

NOTE on ch["P"]: the §6.1 channel dict carries no power key while §6.2's sum_rate
reads ch["P"]; we use ch.get("P", 1.0) — 1.0 W == 30 dBm == env.yaml p_tx_dbm.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

REAL = torch.float64
CPX = torch.complex128
TWO_PI = 2.0 * math.pi

__all__ = [
    "sum_rate_phi", "sum_rate_torch", "sum_rate_numpy",
    "riemannian_gradient", "project_tangent", "retract",
    "ao_riemannian", "pg_sanity", "run_ao_dataset", "load_channel_npz",
    "random_rates", "best_of_random", "no_ris_rate",
]


# --------------------------------------------------------------------------------------
# Channel handling (frozen §6.1 dict contract: h_d (K,M), G (M,N), h_r (K,N), sigma2, [P])
# --------------------------------------------------------------------------------------

def _to_ch_torch(ch: Dict, device: str = "cpu") -> Dict:
    """numpy channel dict -> torch complex128 dict (single channel, no batch dim)."""
    def cx(x):
        return torch.as_tensor(np.asarray(x), dtype=CPX, device=device)

    return {
        "h_d": cx(ch["h_d"]),
        "G": cx(ch["G"]),
        "h_r": cx(ch["h_r"]),
        "sigma2": float(ch["sigma2"]),
        "P": float(ch.get("P", 1.0)),          # 30 dBm -> 1 W (env.yaml p_tx_dbm)
    }


def load_channel_npz(path) -> List[Dict]:
    """Load the frozen dataset — supports object-array and stacked-array layouts."""
    z = np.load(path, allow_pickle=True)
    files = list(z.files)

    for k in ("chs", "channels"):                       # layout A: object array
        if k in files:
            return [{kk: (np.asarray(v) if kk not in ("sigma2", "P") else v)
                     for kk, v in ch.items()} for ch in z[k]]

    if all(k in files for k in ("h_d", "G", "h_r")):    # layout B: stacked arrays
        h_d, G, h_r = np.asarray(z["h_d"]), np.asarray(z["G"]), np.asarray(z["h_r"])
        n = h_d.shape[0]
        s2 = np.asarray(z["sigma2"]).reshape(-1) if "sigma2" in files else np.full(n, 1e-9)
        P = np.asarray(z["P"]).reshape(-1) if "P" in files else None
        out = []
        for i in range(n):
            ch = {"h_d": h_d[i], "G": G[i], "h_r": h_r[i], "sigma2": float(s2[i])}
            if P is not None:
                ch["P"] = float(P[i] if P.size > 1 else P[0])
            for k in files:                             # metadata (u_pos etc.)
                if k in ("h_d", "G", "h_r", "sigma2", "P"):
                    continue
                arr = np.asarray(z[k])
                ch[k] = arr.item() if arr.shape == () else arr[i]
            out.append(ch)
        return out

    raise KeyError(f"Unrecognized npz layout. Keys = {files}")


# --------------------------------------------------------------------------------------
# Differentiable sum rate (torch twin of runbook §6.2 — the frozen numerical heart)
# --------------------------------------------------------------------------------------

def sum_rate_phi(phi, ch_t: Dict):
    """Sum rate under ZF as a function of the complex unit-modulus phase vector phi.

    STRICT batching semantics (mismatched shapes raise ValueError):
      * phi (N,)   with a SINGLE-channel dict (h_d (K,M), G (M,N), h_r (K,N)) -> scalar
      * phi (B,N)  with a BATCHED dict ((B,K,M), (B,M,N), (B,K,N))            -> (B,)

    Formula (frozen §6.2, conjugations absorbed exactly as implemented there):
        Heff = h_d + (h_r * phi) @ G^H                     (K,M)
        W    = Heff^H (Heff Heff^H)^{-1},  ||W||_F^2 = P   (ZF; SINR == SNR)
        R    = sum_k log2(1 + |h~_k^H w_k|^2 / sigma2)
    The §6.2 skeleton's final pairing line `np.sum(Heff * W.conj(), axis=1)` is
    shape-broken for K != M; the correct pairing (Heff @ W = I  =>  user k owns
    column k of W) is the einsum below.  Logged as design decision B2.
    """
    h_d, G, h_r = ch_t["h_d"], ch_t["G"], ch_t["h_r"]
    sigma2, P = ch_t["sigma2"], ch_t["P"]

    if phi.dim() == 1:
        if h_d.dim() != 2:
            raise ValueError("phi (N,) requires a single-channel (unbatched) dict")
        Heff = h_d + (h_r * phi) @ G.conj().T                          # (K,M)
        Hh = Heff.conj().T                                             # (M,K)
        W = Hh @ torch.linalg.inv(Heff @ Hh)                           # ZF
        W = W * torch.sqrt(P / (W.abs() ** 2).sum())
        gain = torch.einsum("km,mk->k", Heff, W.conj())                # h~_k^H w_k
        snr = (gain.abs() ** 2) / sigma2
        return torch.log2(1.0 + snr).sum()                             # scalar

    B = phi.shape[0]
    if h_d.dim() != 3 or h_d.shape[0] != B:
        raise ValueError(
            f"phi batch {tuple(phi.shape)} misaligned with channel batch "
            f"{tuple(h_d.shape)} — stack channels with channels_to_tensors()")
    Heff = h_d + (h_r * phi.unsqueeze(1)) @ G.conj().transpose(1, 2)   # (B,K,M)
    Hh = Heff.conj().transpose(1, 2)                                   # (B,M,K)
    W = Hh @ torch.linalg.inv(Heff @ Hh)                               # (B,M,K) ZF
    p_norm = (W.abs() ** 2).sum(dim=(1, 2))                            # (B,)
    scale = P / p_norm if not (torch.is_tensor(P) and P.dim() > 0) else P / p_norm
    W = W * torch.sqrt(scale).reshape(B, 1, 1)
    gain = torch.einsum("bkm,bmk->bk", Heff, W.conj())                 # h~_k^H w_k
    s2 = sigma2.reshape(B, 1) if (torch.is_tensor(sigma2) and sigma2.dim() > 0) else sigma2
    snr = (gain.abs() ** 2) / s2                                       # ZF => SINR = SNR
    return torch.log2(1.0 + snr).sum(dim=1)                            # (B,)


def sum_rate_torch(theta, ch_t: Dict):
    """Angle-parametrization wrapper: theta (N,) real or (B,N) batched -> sum rate.

    Batching must match the channel dict exactly ((N,) with a single-channel
    dict, (B,N) with a batched dict) — same strict semantics as sum_rate_phi.
    This is the frozen differentiable API the unfolding (§6.7) and all gate
    scripts code against.  Autograd wrt theta is the real R^N gradient used by
    the unrolled projected-gradient steps.
    """
    theta = torch.as_tensor(theta, dtype=REAL)
    return sum_rate_phi(torch.exp(1j * theta), ch_t)


def sum_rate_numpy(theta, ch: Dict) -> float:
    """numpy reference twin of §6.2 (kept for tests / sanity; canonical numpy rate
    lives in Ayansh's src/precoder.py — the twin test keeps them in lock-step)."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    phi = np.exp(1j * theta)
    Heff = ch["h_d"] + (ch["h_r"] * phi) @ ch["G"].conj().T                 # (K,M)
    Hh = Heff.conj().T
    W = Hh @ np.linalg.inv(Heff @ Hh)
    W = W * np.sqrt(float(ch.get("P", 1.0)) / np.sum(np.abs(W) ** 2))
    gain = np.einsum("km,mk->k", Heff, W.conj())
    snr = np.abs(gain) ** 2 / ch["sigma2"]
    return float(np.sum(np.log2(1 + snr)))


# --------------------------------------------------------------------------------------
# Torus manifold machinery (runbook §4.2 math table — implemented verbatim)
# --------------------------------------------------------------------------------------

def project_tangent(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """P_T(z)(v) = v - Re(v ∘ conj(z)) ∘ z   (Euclidean metric of R^{2N})."""
    return v - (v.conj() * z).real * z


def retract(z: torch.Tensor, d: torch.Tensor, gamma: float) -> torch.Tensor:
    """R_z(gamma d) = (z + gamma d) / |z + gamma d|   — back onto the torus."""
    z_new = z + gamma * d
    return z_new / z_new.abs()


def _real_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Real inner product <u, v>_R = Re(sum conj(u) v) on C^N."""
    return (u.conj() * v).real.sum()


def riemannian_gradient(z: torch.Tensor, ch_t: Dict):
    """Euclidean (R^{2N}) gradient of the sum rate at z via autograd, then the
    §4.2 tangent projection.  PyTorch's complex-autograd convention returns
    dL/dRe(z) + j dL/dIm(z), i.e. exactly the Euclidean gradient — verified
    against central finite differences in tests/test_ao_grad.py.

    Returns (f, g, rgrad): value, Euclidean gradient, Riemannian gradient.
    """
    z = z.detach().clone().requires_grad_(True)
    f = sum_rate_phi(z, ch_t)
    g = torch.autograd.grad(f, z)[0]
    rgrad = project_tangent(g, z).detach()
    return f.detach(), g.detach(), rgrad


# --------------------------------------------------------------------------------------
# G2 owner: Riemannian conjugate-gradient ascent on the torus
# --------------------------------------------------------------------------------------

def ao_riemannian(ch: Dict, iters: int = 200, tol: float = 1e-4,
                  c1: float = 1e-4, gamma0: float = 1.0, rho: float = 0.5,
                  max_backtracks: int = 30, record_history: bool = True) -> Dict:
    """Riemannian CG (Fletcher–Reeves) on M = {|z_i| = 1} maximizing the sum rate.

    Frozen protocol (runbook §2 G2 / §4.2):
      * objective f(z) = sum_k log2(1 + SNR_k(z))  — autograd gradient
      * tangent projection P_T(z)(v) = v − Re(v∘conj(z))∘z
      * direction d = rgrad + β · P_T(z)(d_prev),  β_FR = ||r_new||²/||r_old||²
        with periodic resets (every N iters, on non-ascent, or on Armijo failure)
      * retraction z ← (z + γd) / |z + γd|, Armijo backtracking on γ
      * stop when ||rgrad||_F < tol (1e-4) OR iters (200)

    SIGN CONVENTION (viva note): runbook §4.2 writes the CG recursion in
    MINIMIZATION form, d ← −rgrad + β·P_T(d).  We MAXIMIZE the rate directly,
    so the steepest-ASCENT tangent direction is +rgrad — algebraically the
    same algorithm run on the objective −f.  (An earlier draft kept the
    minus sign and stalled exactly as the §8 playbook predicts: Armijo kept
    shrinking γ on a descent direction.  tests/test_ao_grad.py catches this
    class of bug — that is why FD tests are the day-1 deliverable.)

    Runs on a SAVED channel dict only.  Returns dict with theta (numpy, (N,)),
    final rate, iteration count, convergence flag, history, wall time.
    """
    t_start = time.perf_counter()
    ch_t = _to_ch_torch(ch)
    N = ch_t["h_r"].shape[1]

    z = torch.ones(N, dtype=CPX)                      # theta = 0 init (same as unfolding)
    f, _g, rgrad = riemannian_gradient(z, ch_t)
    rgrad_norm2 = _real_dot(rgrad, rgrad)

    d: Optional[torch.Tensor] = None
    d_prev_norm2 = rgrad_norm2.clone()
    history = [float(f)] if record_history else None
    it_used = 0
    converged = False
    stalled = False

    for t in range(iters):
        it_used = t + 1
        if torch.sqrt(rgrad_norm2) < tol:
            converged = True
            it_used = t                                # converged BEFORE this iteration
            break

        # --- CG direction (Fletcher–Reeves, transported by tangent projection) ---
        # We MAXIMIZE f, so the ascent direction is +rgrad (see SIGN CONVENTION above).
        if d is None or (t % N == 0):
            d = rgrad
        else:
            beta = rgrad_norm2 / d_prev_norm2
            d = rgrad + beta * project_tangent(d, z)
        slope = _real_dot(rgrad, d)
        if slope <= 0:                                 # non-ascent safeguard -> restart
            d = rgrad
            slope = _real_dot(rgrad, d)

        # --- Armijo backtracking on the retraction step size ---
        accepted = False
        gamma = gamma0
        f_new = None
        z_new = None
        for _ in range(max_backtracks):
            z_try = retract(z, d, float(gamma))
            f_try = sum_rate_phi(z_try, ch_t)
            f_try = torch.nan_to_num(f_try, nan=-1e30) if torch.isnan(f_try) else f_try
            if f_try >= f + c1 * gamma * slope:
                accepted = True
                z_new, f_new = z_try, f_try
                break
            gamma *= rho
        if not accepted:
            # one steepest-ascent retry, then declare stalled (numerically converged)
            d = rgrad
            slope = _real_dot(rgrad, d)
            if slope <= 0:
                stalled = True
                break
            for _ in range(max_backtracks):
                z_try = retract(z, d, float(gamma))
                f_try = sum_rate_phi(z_try, ch_t)
                if torch.isnan(f_try):
                    gamma *= rho
                    continue
                if f_try >= f + c1 * gamma * slope:
                    accepted, z_new, f_new = True, z_try, f_try
                    break
                gamma *= rho
            if not accepted:
                stalled = True
                break

        z, f = z_new, f_new
        d_prev_norm2 = rgrad_norm2
        f, _g, rgrad = riemannian_gradient(z, ch_t)
        rgrad_norm2 = _real_dot(rgrad, rgrad)
        if record_history:
            history.append(float(f))

    theta = torch.angle(z).detach().numpy()
    return {
        "theta": theta,
        "rate": float(f),
        "iters": int(it_used),
        "converged": bool(converged),
        "stalled": bool(stalled),
        "rgrad_norm": float(torch.sqrt(rgrad_norm2).detach()),
        "history": history,
        "time_s": time.perf_counter() - t_start,
    }


def pg_sanity(ch: Dict, iters: int = 200, step0: float = 0.5,
              tol: float = 1e-4, max_backtracks: int = 20) -> Dict:
    """Plain projected-gradient ASCENT twin in the angle domain (runbook §6.6 note:
    'reviewers love the AO >> PG comparison').  Same init, same stopping rule,
    no CG memory — expected to converge slower / to worse local points than AO."""
    t_start = time.perf_counter()
    ch_t = _to_ch_torch(ch)
    N = ch_t["h_r"].shape[1]

    theta = torch.zeros(N, dtype=REAL, requires_grad=True)
    f = sum_rate_torch(theta, ch_t).detach()
    history = [float(f)]
    it_used = 0
    converged = False

    for t in range(iters):
        it_used = t + 1
        g = torch.autograd.grad(sum_rate_torch(theta, ch_t), theta)[0]
        gnorm = g.norm()
        if gnorm < tol:
            converged = True
            it_used = t
            break
        step, accepted = step0, False
        for _ in range(max_backtracks):
            theta_try = theta.detach() + step * g
            f_try = float(sum_rate_torch(theta_try, ch_t))
            if f_try >= float(f) + 1e-4 * step * float(g @ g):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        # project back to the fundamental domain [0, 2pi) — rate is 2pi-periodic
        theta = (theta_try - TWO_PI * torch.floor(theta_try / TWO_PI)).detach().requires_grad_(True)
        f = torch.tensor(f_try, dtype=REAL)
        history.append(float(f))

    return {
        "theta": theta.detach().numpy() % TWO_PI,
        "rate": float(f),
        "iters": int(it_used),
        "converged": bool(converged),
        "history": history,
        "time_s": time.perf_counter() - t_start,
    }


# --------------------------------------------------------------------------------------
# AO-side reference helpers (canonical baselines live in Ayansh's src/baselines.py —
# identical RNG protocol: np.random.default_rng, uniform(0, 2pi, N), so numbers agree)
# --------------------------------------------------------------------------------------

def random_rates(ch: Dict, rng: np.random.Generator, n_draws: int = 200) -> np.ndarray:
    N = np.asarray(ch["h_r"]).shape[1]
    out = np.empty(n_draws)
    for i in range(n_draws):
        out[i] = sum_rate_numpy(rng.uniform(0.0, TWO_PI, N), ch)
    return out


def best_of_random(ch: Dict, rng: np.random.Generator, n_draws: int = 200) -> float:
    return float(random_rates(ch, rng, n_draws).max())


def no_ris_rate(ch: Dict) -> float:
    ch2 = dict(ch)
    ch2["h_r"] = np.zeros_like(np.asarray(ch["h_r"]))
    return sum_rate_numpy(np.zeros(np.asarray(ch["h_r"]).shape[1]), ch2)


# --------------------------------------------------------------------------------------
# Batch runner used by the gate / reference scripts
# --------------------------------------------------------------------------------------

def _ao_worker(ch, iters, tol):
    """Module-level worker (spawn-picklable); single-threaded torch per process."""
    torch.set_num_threads(1)
    return ao_riemannian(ch, iters=iters, tol=tol)


def run_ao_dataset(chs: List[Dict], iters: int = 200, tol: float = 1e-4,
                   workers: int = 1, progress_every: int = 25) -> List[Dict]:
    """Run ao_riemannian over a list of saved channels.

    workers=1 keeps everything in-process (deterministic, easiest to debug).
    workers>1 forks processes (channels are small numpy dicts — picklable);
    each worker runs single-threaded torch for throughput.
    """
    if workers <= 1:
        results = []
        for i, ch in enumerate(chs):
            results.append(ao_riemannian(ch, iters=iters, tol=tol))
            if progress_every and (i + 1) % progress_every == 0:
                part = np.mean([r["rate"] for r in results])
                print(f"  [run_ao_dataset] {i + 1}/{len(chs)} done | partial mean = {part:.3f} bps/Hz", flush=True)
        return results

    import multiprocessing as mp
    from functools import partial

    fn = partial(_ao_worker, iters=iters, tol=tol)
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        return pool.map(fn, chs, chunksize=max(1, len(chs) // (workers * 4)))
