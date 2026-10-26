"""
src/unfolding.py — Balraj Singh (24BCE0531) — runbook §4.2 / §6.7 (fix #5)
==========================================================================

Deep unfolding: L unrolled projected-gradient ASCENT steps on the SUM RATE,
trained END-TO-END.  This is the direct fix for old cause #5 ("unfolding
trained with MSE on wrapped angles"):

  * forward pass  = L = 6 fixed gradient-ascent layers on R(theta) — the
    network IS a differentiable optimizer, not a phase regressor;
  * trainable parameters = ONE log-step-size per layer (log -> always > 0);
  * MAIN loss = -(mean sum rate of the final layer) — the rate itself;
  * MSE-on-wrapped-angles (the old bug) is kept ONLY as an ablation, and the
    circular loss 1 - cos(dtheta) is the discontinuity-free middle ablation.

Fairness notes (runbook §1.5):
  * init theta = 0 — identical to ao_riemannian's init;
  * same frozen train/test npz split (Ayansh is the only writer of data/);
  * identical evaluation entry point / channels as every other method.

dtype: float64/complex128 in training (FD-grade gradients); see ao_math.md.
Inference-latency lever (G3): channels_to_tensors(dtype=torch.complex64) or
cast_tensors(T, torch.complex64) + UnrolledPG(dtype=torch.float32) gives a
float32 inference path — forward() infers its dtypes from the INPUT channels,
so mixed model/channel dtypes are safe.  Defaults keep float64 behaviour
bit-identical; scripts/latency_float32.py reports both rows honestly.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:                                    # package mode once merged into the repo
    from .ao_riemannian import CPX, REAL, TWO_PI, sum_rate_torch, sum_rate_phi, load_channel_npz
except ImportError:                     # flat mode (tests / standalone dev on feat/opt-bound)
    from ao_riemannian import CPX, REAL, TWO_PI, sum_rate_torch, sum_rate_phi, load_channel_npz

__all__ = [
    "UnrolledPG", "loss_rate", "loss_circular", "loss_mse_angles",
    "wrap_diff", "channels_to_tensors", "cast_tensors", "batch_slice",
    "sum_rate_safe", "evaluate_on_channels", "measure_decision_latency",
    "train_from_config",
]


def _real_of(cpx_dtype: torch.dtype) -> torch.dtype:
    """Real dtype matching a complex dtype (complex128->float64, complex64->float32)."""
    return torch.float32 if cpx_dtype == torch.complex64 else torch.float64


def sum_rate_safe(theta: torch.Tensor, ch_t: Dict) -> torch.Tensor:
    """dtype-safe sum rate: same math as ao_riemannian.sum_rate_torch, but the
    phase vector phi is cast EXPLICITLY to the channel dtype before sum_rate_phi.

    Why this exists (live bug): sum_rate_torch computes torch.exp(1j * theta)
    internally, and in torch the Python-scalar promotion `1j * <float32 or
    complex64 tensor>` ALWAYS yields complex128 — no weak form for complex
    scalars.  Against complex64 channels that raises:
        RuntimeError: expected scalar type ComplexFloat but found ComplexDouble
        (ao_riemannian.py:150, via :172)
    We therefore build phi here and let ao_riemannian.sum_rate_phi (G2-SIGNED,
    READ-ONLY — never edited) consume a phi that already matches the channels.

    float64/complex128 path (training defaults): .to(complex128) is a no-op,
    so this is bit-identical to sum_rate_torch.  float32/complex64 path: the
    cast lands phi in complex64 and the rate flows through exp/cast with full
    autograd support.
    """
    hr = ch_t["h_r"]
    cdtype = hr.dtype if hr.is_complex() else CPX      # channels are complex in this pipeline
    phi = torch.exp((1j * theta).to(cdtype))           # (1j*theta) may upcast to complex128; cast pins it
    return sum_rate_phi(phi, ch_t)


# --------------------------------------------------------------------------------------
# The model (runbook §6.7, trainable step sizes only — frozen design)
# --------------------------------------------------------------------------------------

class UnrolledPG(nn.Module):
    """L unrolled projected-gradient ASCENT steps on the sum rate.

    theta_{l+1} = wrap( theta_l + alpha_l * dR/dtheta |_{theta_l} ),  alpha_l = exp(rho_l) > 0

    * ONE trainable scalar per layer (self.alphas, log-parametrized).
    * wrap projects theta back into [0, 2pi) each layer ("projected" PG; the
      rate is 2pi-periodic, so this only keeps numbers bounded).
    * create_graph=True during training so the loss gradient flows THROUGH the
      L internal gradient evaluations (second-order autograd, batched).
    * dtype: real dtype of the step sizes (default float64).  forward() takes
      the working dtype from the INPUT channel dict, so the same module can be
      trained in float64 and measured in float32 without code changes.
    """

    def __init__(self, L: int = 6, theta_init: float = 0.0, wrap: bool = True,
                 dtype: torch.dtype = REAL):
        super().__init__()
        self.L = int(L)
        self.theta_init = float(theta_init)
        self.wrap = bool(wrap)
        self.real_dtype = _real_of(dtype)
        self.alphas = nn.Parameter(torch.log(
            torch.full((self.L,), 0.5, dtype=self.real_dtype)))

    def forward(self, ch_t: Dict, theta: Optional[torch.Tensor] = None,
                L: Optional[int] = None, create_graph: bool = True) -> torch.Tensor:
        """Run the unrolled ascent for a BATCH of channels.

        ch_t   : batched torch channel dict (leading batch dim B)
        theta  : optional (B,N) starting phases (default: theta_init everywhere)
        returns theta_final (B, N)
        """
        B, N = ch_t["h_r"].shape[0], ch_t["h_r"].shape[2]
        rdtype = (ch_t["h_r"].real.dtype if ch_t["h_r"].is_complex()
                  else ch_t["h_r"].dtype)          # working dtype follows the input
        L = self.L if L is None else int(L)
        if theta is None:
            theta = torch.full((B, N), self.theta_init, dtype=rdtype, requires_grad=True)
        alphas = torch.exp(self.alphas.to(rdtype))

        for l in range(L):
            if not theta.requires_grad:            # eval mode: relink a fresh leaf
                theta = theta.detach().requires_grad_(True)
            rate = sum_rate_safe(theta, ch_t)      # (B,) batched rates; dtype-safe wrapper (phi pinned to the channel dtype — see sum_rate_safe)
            # grad_outputs=ones => per-element dR/dtheta of the TOTAL batch rate:
            # the exact ascent signal for simultaneous per-channel PG steps.
            g = torch.autograd.grad(rate, theta, grad_outputs=torch.ones_like(rate),
                                    create_graph=create_graph)[0]
            theta = theta + alphas[l] * g          # ASCENT (we maximize the rate)
            if self.wrap:
                theta = theta - TWO_PI * torch.floor(theta / TWO_PI)
        return theta


# --------------------------------------------------------------------------------------
# Losses: rate (main) / circular (ablation) / MSE-on-wrapped-angles (the old bug)
# --------------------------------------------------------------------------------------

def loss_rate(theta_final: torch.Tensor, ch_t: Dict) -> torch.Tensor:
    """MAIN objective (fix #5): maximize the achievable sum rate end-to-end."""
    return -sum_rate_safe(theta_final, ch_t).mean()


def wrap_diff(theta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Circular residual in (-pi, pi]: the metric angle differences actually live on."""
    return torch.remainder(theta - ref + math.pi, TWO_PI) - math.pi


def loss_circular(theta_final: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    """Ablation B: 1 - cos(dtheta) — smooth, branch-cut-free, in [0, 2]."""
    d = wrap_diff(theta_final, theta_ref)
    return (1.0 - torch.cos(d)).mean()


def loss_mse_angles(theta_final: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    """Ablation C (the OLD BUG, kept only to reproduce it): MSE on np.angle-wrapped
    values.  Discontinuous at the ±pi branch cut — a target of +pi vs -pi is the
    same phase but a huge MSE.  Never use as the main loss (runbook fix #5)."""
    a = torch.angle(torch.exp(1j * theta_final))
    b = torch.angle(torch.exp(1j * theta_ref))
    return ((a - b) ** 2).mean()


LOSSES = {"rate": loss_rate, "circular": loss_circular, "mse": loss_mse_angles}


# --------------------------------------------------------------------------------------
# Dataset helpers (frozen npz interface — Ayansh is the only writer of data/)
# --------------------------------------------------------------------------------------

def channels_to_tensors(chs: List[Dict], dtype: torch.dtype = CPX):
    """Stack a channel list into batched torch tensors.

    dtype: complex dtype of the channel arrays — complex128 (default,
    FD-grade training) or complex64 (float32 inference-latency lever).
    The real dtype (sigma2) is derived from it."""
    real = _real_of(dtype)
    h_d = torch.as_tensor(np.stack([np.asarray(c["h_d"]) for c in chs]), dtype=dtype)
    G = torch.as_tensor(np.stack([np.asarray(c["G"]) for c in chs]), dtype=dtype)
    h_r = torch.as_tensor(np.stack([np.asarray(c["h_r"]) for c in chs]), dtype=dtype)
    sigma2 = torch.as_tensor(np.array([float(c["sigma2"]) for c in chs]), dtype=real)
    P = float(chs[0].get("P", 1.0))
    return {"h_d": h_d, "G": G, "h_r": h_r, "sigma2": sigma2, "P": P}


def cast_tensors(T: Dict, dtype: torch.dtype) -> Dict:
    """Copy of a stacked tensor dict cast to `dtype` (complex tensors) / its real
    partner (float tensors).  Non-tensor values pass through unchanged."""
    real = _real_of(dtype)
    out = {}
    for k, v in T.items():
        if torch.is_tensor(v):
            out[k] = v.to(dtype) if v.is_complex() else v.to(real)
        else:
            out[k] = v
    return out


def batch_slice(T: Dict, idx: np.ndarray) -> Dict:
    n = int(T["h_r"].shape[0])
    return {k: (v[idx] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n else v) for k, v in T.items()}


def evaluate_on_channels(model: UnrolledPG, T: Dict, chunk: int = 128) -> float:
    """Mean sum rate of the model's final-layer phases over a stacked dataset.

    NOTE: the forward pass IS a gradient-based optimizer, so it must run under
    torch.enable_grad() even at eval time; only the final rate readout is
    detached (and no model-parameter backward is ever taken here)."""
    model.eval()
    rates = []
    n = T["h_r"].shape[0]
    for s in range(0, n, chunk):
        idx = torch.arange(s, min(s + chunk, n))
        chb = batch_slice(T, idx)
        with torch.enable_grad():                      # the layers need dR/dtheta
            theta = model(chb, create_graph=False)
        with torch.no_grad():
            rates.append(sum_rate_safe(theta, chb).detach())
    model.train()
    return float(torch.cat(rates).mean())


def measure_decision_latency(model: UnrolledPG, T: Dict, n_trials: int = 200,
                             warmup: int = 20) -> Dict:
    """Wall-clock per decision (one channel, L layers) — the G3 <1 ms claim table.

    Measured as-is on the dtype of T (float64 training dtype, or float32 via
    cast_tensors — the documented lever); reported honestly either way."""
    one = batch_slice(T, torch.arange(1))
    for _ in range(warmup):
        model(one, create_graph=False)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.enable_grad():                      # inference still runs PG steps
            model(one, create_graph=False)
        times.append(time.perf_counter() - t0)
    times = np.sort(np.asarray(times) * 1e3)           # ms
    return {"median_ms": float(np.median(times)), "p95_ms": float(times[int(0.95 * len(times))]),
            "mean_ms": float(times.mean()), "dtype": str(one["h_r"].dtype), "n_trials": n_trials}


# --------------------------------------------------------------------------------------
# Trainer (D6–D10 deliverable; entry point: scripts/train_unfolding.py)
# --------------------------------------------------------------------------------------

def train_from_config(cfg: Dict, loss_override: Optional[str] = None,
                      max_iters_override: Optional[int] = None,
                      out_suffix: str = "") -> Dict:
    """Train UnrolledPG from a parsed configs/unfolding.yaml (DD-041 protocol).

    Protocol (audit-revised): the 1,500 training channels are split deterministically
    into TRAIN (first 80%) and VALIDATION (last 20%).  Mid-training monitoring and any
    selection happens on VALIDATION only; the frozen 500-channel test set is evaluated
    exactly ONCE, on the validation-best checkpoint, after training completes.

    loss = rate      : end-to-end sum-rate loss (main method, fix #5)
    loss = circular  : needs AO reference thetas (results/ao/train/ao_thetas.npy)
    loss = mse       : the old bug, ablation only (same AO reference requirement)

    Returns the metrics dict; writes ckpt + metrics.json under out.root.
    """
    loss_name = (loss_override or cfg["train"]["loss"]).lower()
    assert loss_name in LOSSES, f"unknown loss {loss_name}"
    L = int(cfg["method_L"]) if "method_L" in cfg else int(cfg.get("L", 6))
    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    root = Path(cfg["out"]["root"])
    tag = f"{cfg['out']['ckpt_stem']}_L{L}_{loss_name}{out_suffix}"
    (root / f"{tag}").mkdir(parents=True, exist_ok=True)

    train_chs = load_channel_npz(cfg["data"]["train_channels"])
    test_chs = load_channel_npz(cfg["data"]["test_channels"])
    # DD-041 protocol: deterministic train/val split (last val_frac slice = validation)
    val_frac = float(cfg["train"].get("val_frac", 0.2))
    n_val = max(1, int(round(val_frac * len(train_chs))))
    train_chs, val_chs = train_chs[:-n_val], train_chs[-n_val:]
    T_train = channels_to_tensors(train_chs)
    T_val = channels_to_tensors(val_chs)
    T_test = channels_to_tensors(test_chs)
    N = T_train["h_r"].shape[2]

    theta_ref = None
    if loss_name in ("circular", "mse"):
        ref_path = Path(cfg["data"]["ao_train_refs"])
        if not ref_path.exists():
            raise FileNotFoundError(
                f"loss={loss_name} needs AO reference thetas. Run first:\n"
                f"  python scripts/run_ao_reference.py --channels {cfg['data']['train_channels']}"
                f" --out results/ao/train")
        theta_ref_all = torch.as_tensor(np.load(ref_path), dtype=REAL)
        # refs align with the ORIGINAL train npz; slice to the split's train part
        assert len(theta_ref_all) >= len(train_chs), "AO refs misaligned with train npz"
        theta_ref_all = theta_ref_all[:len(train_chs)]

    model = UnrolledPG(L=L, theta_init=float(cfg["train"]["theta_init"]),
                       wrap=bool(cfg["train"]["wrap_theta"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    B = int(cfg["train"]["batch_channels"])
    iters = int(max_iters_override or cfg["train"]["iters"])
    eval_every = int(cfg["train"]["eval_every"])

    base_rate = float(sum_rate_safe(torch.zeros(len(test_chs), N, dtype=REAL),
                                    T_test).mean())     # all-zero phases reference (float64 defaults — bit-identical to sum_rate_torch)
    init_model_rate = evaluate_on_channels(model, T_val)  # model before any training (validated on VAL per DD-041)
    hist = []
    best = {"val_rate": -float("inf"), "iter": -1, "state": None}
    t_start = time.perf_counter()

    for it in range(iters + 1):
        # batch WITH replacement, clamped to the dataset size (keeps theta/ref/channel
        # batch dims aligned even when batch_channels > n_train, e.g. in smoke runs)
        idx = rng.integers(0, len(train_chs), min(B, len(train_chs)))
        chb = batch_slice(T_train, torch.as_tensor(idx))
        theta = model(chb, create_graph=True)
        if loss_name == "rate":
            loss = loss_rate(theta, chb)
        else:
            ref = theta_ref_all[torch.as_tensor(idx)]
            loss = LOSSES[loss_name](theta, ref)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % eval_every == 0 or it == iters:
            val_rate = evaluate_on_channels(model, T_val)
            if val_rate > best["val_rate"]:                      # selection on VAL only (DD-041)
                best = {"val_rate": val_rate, "iter": it,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            rec = {"iter": it, "train_loss": float(loss.detach()),
                   "val_mean_rate": val_rate,
                   "gain_vs_zero_init": val_rate - base_rate,
                   "alphas": [float(a) for a in torch.exp(model.alphas).detach()],
                   "wall_s": time.perf_counter() - t_start}
            hist.append(rec)
            print(f"  [unfolding:{loss_name}] iter {it:5d}  loss {rec['train_loss']: .4f}"
                  f"  val rate {val_rate: .4f} (best {best['val_rate']: .4f} @ {best['iter']})"
                  f"  alphas {np.round(rec['alphas'], 3).tolist()}", flush=True)

    # restore the VALIDATION-BEST weights, then evaluate the frozen test set once (DD-041)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test_rate = evaluate_on_channels(model, T_test)

    latency = measure_decision_latency(model, T_test, n_trials=int(cfg["eval"]["latency_trials"]))
    metrics = {
        "tag": tag, "L": L, "loss": loss_name, "seed": seed,
        "iters": iters, "batch_channels": B, "lr": float(cfg["train"]["lr"]),
        "N": int(N), "n_train": len(train_chs), "n_val": len(val_chs),
        "n_test": len(test_chs), "val_frac": val_frac,
        "protocol": "DD-041: train/val split, val-selected best, single final test eval",
        "zero_phase_test_rate": base_rate,
        "init_model_val_rate": init_model_rate,
        "best_iter": best["iter"],
        "best_val_rate": best["val_rate"],
        "final_test_rate": test_rate,                      # test eval of the val-best checkpoint
        "latency_ms": latency,
        "history": hist,
    }
    torch.save(model.state_dict(), str(root / f"{tag}.pt"))
    with open(root / f"{tag}" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [unfolding:{loss_name}] saved {root / tag}.pt  |  final test rate"
          f" {metrics['final_test_rate']:.4f}  |  median decision {latency['median_ms']:.3f} ms")
    return metrics