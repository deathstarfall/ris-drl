"""
tests/test_unfolding_grad.py — Balraj Singh (24BCE0531) — runbook §3 tests/ ownership
====================================================================================

Finite-difference / behaviour tests for the deep unfolding (fix #5):
  1. theta-space gradient of the rate (the signal the layers ascend along)
     matches central finite differences,
  2. one unrolled step with a small step size increases the rate (ascent),
  3. second-order backprop through L layers produces finite alpha-gradients,
  4. the three losses are finite; circular/mse vanish exactly at the target,
  5. a short end-to-end training improves the frozen-test-set rate,
  6. inference latency is measured and finite (honest G3 table, no <1ms claim here).

Self-contained synthetic channels (M=4, N=8, K=2) — green on day 1, before
Ayansh's data/*.npz exist.
"""

import math
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import REAL, CPX, TWO_PI, sum_rate_torch  # noqa: E402
from unfolding import (  # noqa: E402
    UnrolledPG, loss_circular, loss_mse_angles, loss_rate,
    channels_to_tensors, evaluate_on_channels, measure_decision_latency, wrap_diff,
)


def _synth_stack(n: int = 8, seed: int = 7, M: int = 4, N: int = 8, K: int = 2):
    rng = np.random.default_rng(seed)
    cn = lambda s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / np.sqrt(2)
    chs = [{"h_d": 0.05 * cn((K, M)), "G": cn((M, N)), "h_r": 1.0 * cn((K, N)),
            "sigma2": 1e-2, "P": 1.0} for _ in range(n)]
    return chs, channels_to_tensors(chs)


def _theta_grad(theta: torch.Tensor, T: dict) -> torch.Tensor:
    theta = theta.detach().clone().requires_grad_(True)
    r = sum_rate_torch(theta, T)
    return torch.autograd.grad(r, theta)[0]


# --------------------------------------------------------------------------------------
# 1) theta-space gradient vs finite differences
# --------------------------------------------------------------------------------------

def test_theta_grad_matches_finite_differences():
    chs, _T = _synth_stack(n=2, seed=13)
    T1 = {k: (v[0] if torch.is_tensor(v) else v) for k, v in _T.items()}  # single channel
    rng = np.random.default_rng(3)
    theta0 = torch.as_tensor(rng.uniform(0, TWO_PI, 8), dtype=REAL)
    g = _theta_grad(theta0, T1)
    eps = 1e-6
    for _ in range(5):
        v = torch.as_tensor(rng.normal(size=8), dtype=REAL)
        v = v / v.norm()
        rp = float(sum_rate_torch(theta0 + eps * v, T1))
        rm = float(sum_rate_torch(theta0 - eps * v, T1))
        fd = (rp - rm) / (2 * eps)
        analytic = float(g @ v)
        assert abs(fd - analytic) / max(1.0, abs(fd)) < 1e-5, f"FD {fd} vs autograd {analytic}"


# --------------------------------------------------------------------------------------
# 2) one unrolled step ascends the rate
# --------------------------------------------------------------------------------------

def test_one_unrolled_step_is_ascent():
    chs, T = _synth_stack(n=8, seed=7)
    model = UnrolledPG(L=1, theta_init=0.0, wrap=True)
    with torch.no_grad():
        model.alphas.copy_(torch.log(torch.full((1,), 0.05, dtype=REAL)))  # small step
    r_before = sum_rate_torch(torch.zeros(len(chs), 8, dtype=REAL), T).detach()
    theta = model(T, create_graph=False)
    r_after = sum_rate_torch(theta, T).detach()
    frac = float((r_after > r_before).float().mean())
    assert frac >= 0.9, f"only {frac:.0%} of channels ascended after one small step"
    assert float(r_after.mean()) > float(r_before.mean())


# --------------------------------------------------------------------------------------
# 3) second-order backprop through the L layers reaches the step sizes
# --------------------------------------------------------------------------------------

def test_second_order_backward_reaches_alphas():
    chs, T = _synth_stack(n=4, seed=5)
    model = UnrolledPG(L=3, theta_init=0.0)
    theta = model(T, create_graph=True)
    loss = loss_rate(theta, T)
    loss.backward()
    grad = model.alphas.grad
    assert grad is not None, "no gradient reached the step sizes"
    assert torch.isfinite(grad).all(), f"non-finite alpha grads: {grad}"
    assert float(grad.abs().sum().detach()) > 0, "alpha grads are exactly zero — graph is severed"


# --------------------------------------------------------------------------------------
# 4) the three losses: finite, and zero exactly at the reference
# --------------------------------------------------------------------------------------

def test_losses_finite_and_zero_at_target():
    chs, T = _synth_stack(n=4, seed=17)
    torch.manual_seed(0)
    theta_ref = torch.as_tensor(np.random.default_rng(1).uniform(0, TWO_PI, (4, 8)), dtype=REAL)

    lr_ = loss_rate(theta_ref, T)
    assert torch.isfinite(lr_) and lr_ < 0          # = -mean rate

    lc = loss_circular(theta_ref, theta_ref)
    lm = loss_mse_angles(theta_ref, theta_ref)
    assert float(lc) < 1e-12 and float(lm) < 1e-12

    # circular is branch-cut free: target +pi vs -pi is the SAME phase
    tgt = torch.full((4, 8), math.pi, dtype=REAL)
    other = tgt.clone(); other[..., 0] = -math.pi
    assert float(loss_circular(other, tgt)) < 1e-9
    # ...while MSE-on-wrapped-angles (the old bug) explodes there:
    assert float(loss_mse_angles(other, tgt)) > 1.0

    # wrap_diff stays in (-pi, pi]
    d = wrap_diff(torch.tensor([0.0, 2 * math.pi - 1e-9, -2 * math.pi + 1e-9]),
                  torch.zeros(3, dtype=REAL))
    assert bool((d.abs() <= math.pi + 1e-6).all())


# --------------------------------------------------------------------------------------
# 5) short end-to-end training improves the frozen-test-set rate
# --------------------------------------------------------------------------------------

def test_short_training_improves_test_rate():
    chs, T = _synth_stack(n=8, seed=7)              # 8 "train" channels
    test_chs, T_test = _synth_stack(n=8, seed=99)   # disjoint "test" channels
    torch.manual_seed(0)

    model = UnrolledPG(L=4, theta_init=0.0)
    rate_init = evaluate_on_channels(model, T_test)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    rng = np.random.default_rng(0)
    for _ in range(120):
        idx = torch.as_tensor(rng.integers(0, len(chs), 8))
        Tb = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == len(chs) else v) for k, v in T.items()}
        theta = model(Tb, create_graph=True)
        loss = loss_rate(theta, Tb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    rate_trained = evaluate_on_channels(model, T_test)
    assert rate_trained > rate_init, (
        f"training did not help: init {rate_init:.4f} vs trained {rate_trained:.4f}")


# --------------------------------------------------------------------------------------
# 6) latency measurement machinery works (honest reporting; no <1 ms assertion)
# --------------------------------------------------------------------------------------

def test_latency_measured_and_finite():
    chs, T = _synth_stack(n=1, seed=7)
    model = UnrolledPG(L=6, theta_init=0.0)
    lat = measure_decision_latency(model, T, n_trials=30, warmup=5)
    assert lat["median_ms"] > 0 and np.isfinite(lat["median_ms"])
    assert lat["p95_ms"] >= lat["median_ms"]
    assert lat["median_ms"] < 1000.0                # sanity ceiling only (G3 reports truly)
