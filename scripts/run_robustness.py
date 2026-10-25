#!/usr/bin/env python
"""Robustness sweeps: B-bit quantization and CSI-error (runbook s4.2, D6-D10).

Balraj Singh (24BCE0531). Two sweeps over the FROZEN test set:

  --which bits   R vs bits B in {2,3,4}: quantize the CONTINUOUS solutions of
                 AO / PG / trained unfolding (if a ckpt exists) / random onto the
                 uniform 2^B-per-element codebook and re-score. NOTE: this is
                 post-hoc output quantization — distinct from Shreyansh's
                 trained-codebook DQN sweep (his agent CHOOSES from the codebook).

  --which csi    R vs eps in {0, 0.01, 0.02, 0.05, 0.1}: optimize with the
                 perturbed channel, SCORE on the true channel (runbook s4.2:
                 "evaluate with h-hat, score on true h"). Perturbation model
                 (docs/ao_math.md s8): per block (h_r, G, h_d),
                 Delta = sigma * CN(0,1) per entry with sigma chosen so that
                 E||(Delta)||_F^2 = eps^2 ||h||_F^2 exactly.

Outputs (script-generated only; runbook s9):
    results/unfold/quantization_bits.json
    results/unfold/csi_robustness.json
Exit codes: 0 ok, 3 blocked on section-4.4 handoff artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPS_LIST = (0.0, 0.01, 0.02, 0.05, 0.1)    # frozen in the runbook s4.2
BITS_LIST = (2, 3, 4)                       # frozen in the runbook s4.2


def _blocked(cfg):
    checks = [
        (ROOT / cfg["paths"]["test_channels"], "Ayansh, D2 - frozen test set"),
        (ROOT / "src" / "precoder.py", "Ayansh, D1 - frozen rate contract"),
    ]
    return [f"missing: {p.relative_to(ROOT)}  (owes it: {owe})" for p, owe in checks
            if not p.exists()]


def perturb_channel(ch, eps: float, rng: np.random.Generator):
    """Block-wise multiplicative-scale CSI error with EXACT second moment:

        Delta_ij = sigma * g_ij,  g_ij ~ CN(0,1) (i.e. +j part iid),
        sigma = eps * ||h||_F / sqrt(n),  E||Delta||_F^2 = n*sigma^2 = eps^2||h||_F^2.
    """
    def _emb(arr):
        arr = np.asarray(arr)
        n = arr.size
        sigma = eps * np.linalg.norm(arr) / math.sqrt(n)
        noise = sigma * (rng.normal(size=arr.shape)
                         + 1j * rng.normal(size=arr.shape))
        return arr + noise

    if eps == 0.0:
        return dict(ch)
    out = dict(ch)
    out["h_r"] = _emb(ch["h_r"])
    out["G"] = _emb(ch["G"])
    out["h_d"] = _emb(ch["h_d"])
    return out


def rate_np(theta, ch):
    import src.precoder as prec

    return prec.sum_rate(theta, ch)


def load_unfold_ckpt(unfold_root):
    """Trained rate-loss stepper, if the D6-D10 training already ran."""
    import src.unfolding as un

    ckpts = sorted(unfold_root.glob("ckpts/unfolding_rate_seed*.pt"))
    if not ckpts:
        return None
    ck = torch.load(str(ckpts[-1]), map_location="cpu", weights_only=False)
    L = int(ck["meta"]["L"])
    model = un.UnrolledPG(L=L)
    model.load_state_dict(ck["state_dict"])
    print(f"unfolding ckpt loaded: {ckpts[-1].name} (loss={ck['meta']['loss']})")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["bits", "csi", "both"], default="both")
    ap.add_argument("--n", type=int, default=None, help="subset of frozen test channels")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "unfolding.yaml").read_text())
    missing = _blocked(cfg)
    if missing:
        print("Robustness sweeps BLOCKED (section 4.4):")
        for m in missing:
            print("  -", m)
        return 3

    import src.ao_riemannian as ao
    import src.unfolding as un

    from src.channel import load_dataset

    channels = load_dataset(str(ROOT / cfg["paths"]["test_channels"]))
    n = args.n or len(channels)
    if args.smoke:
        n = min(n, 8)
    channels = channels[:n]

    dt = torch.float32 if cfg["unfolding"]["dtype"] == "float32" else torch.float64
    dof = torch.float64 if cfg["ao"]["dtype"] == "float64" else torch.float32
    max_iter, tol = int(cfg["ao"]["max_iter"]), float(cfg["ao"]["tol"])
    unfold_root = ROOT / cfg["paths"]["unfold"]
    unfold_root.mkdir(parents=True, exist_ok=True)
    unfold_model = load_unfold_ckpt(unfold_root)
    if unfold_model is None:
        print("NOTE: no unfolding ckpt under results/unfold/ckpts — run "
              "scripts/train_unfolding.py first; sweep proceeds with AO / PG / random.")

    # ----------------------------------------------------------------------- #
    # Sweep 1: output quantization (R vs B)                                    #
    # ----------------------------------------------------------------------- #
    if args.which in ("bits", "both"):
        methods = ["ao", "pg", "random"] + (["unfold"] if unfold_model else [])
        results = {"meta": {
            "owner": "Balraj Singh (24BCE0531)",
            "sweep": ("post-hoc quantization of CONTINUOUS solutions onto the "
                      "uniform per-element 2^B codebook (B in 2/3/4); "
                      "SEPARATE from Shreyansh's trained-codebook DQN sweep"),
            "n_channels": len(channels), "bits": list(BITS_LIST),
        }, "methods": {}}
        for method in methods:
            by_bits = {}
            for b in BITS_LIST:
                rates = []
                rng = np.random.default_rng(3000)  # SAME seed per method/bits (paired)
                for i, ch in enumerate(channels):
                    th = _theta_of(method, ch, rng, ao, dof, max_iter, tol,
                                   unfold_model, dt)
                    q = un.quantize_phases(th, b)
                    rates.append(rate_np(q, ch))
                by_bits[str(b)] = [float(x) for x in rates]
            m = [f"{np.mean(by_bits[str(b)]):.3f}" for b in BITS_LIST]
            print(f"[bits] {method}: rates at B=2/3/4 = {m}")
            results["methods"][method] = by_bits
        (unfold_root / "quantization_bits.json").write_text(json.dumps(results, indent=2))
        print(f"saved {unfold_root / 'quantization_bits.json'}")

    # ----------------------------------------------------------------------- #
    # Sweep 2: CSI-error (R vs eps); optimize on h-hat, score on true h        #
    # ----------------------------------------------------------------------- #
    if args.which in ("csi", "both"):
        eps_results = {"meta": {
            "owner": "Balraj Singh (24BCE0531)",
            "sweep": "optimize with perturbed CSI, score on the TRUE channel",
            "eps_list": list(EPS_LIST),
            "model": "per-block Delta_ij = sigma*CN(0,1), sigma = eps*||h||_F/sqrt(n); "
                     "E||Delta||_F^2 = eps^2||h||_F^2 exactly",
            "n_channels": len(channels),
            "unfolding_included": bool(unfold_model),
        }, "eps": {}}
        for k, eps in enumerate(EPS_LIST):
            rows = {"ao": [], "unfold": [], "random": []}
            for i, ch in enumerate(channels):
                rates = _csi_row(eps, i, ch, ao, un, dof, dt,
                                 max_iter, tol, unfold_model)
                rows["ao"].append(rates["ao"])
                if unfold_model:
                    rows["unfold"].append(rates["unfold"])
                rows["random"].append(rates["random"])
            entry = {
                "ao_mean": float(np.mean(rows["ao"])),
                "ao_sd": float(np.std(rows["ao"], ddof=1)),
                "random_mean": float(np.mean(rows["random"])),
                "ao_per_channel": [float(x) for x in rows["ao"]],
            }
            if unfold_model:
                entry["unfold_mean"] = float(np.mean(rows["unfold"]))
                entry["unfold_sd"] = float(np.std(rows["unfold"], ddof=1))
            eps_results["eps"][str(eps)] = entry
            print(f"[csi] eps={eps}: AO={entry['ao_mean']:.3f} "
                  f"random={entry['random_mean']:.3f}"
                  + (f" unfold={entry.get('unfold_mean', float('nan')):.3f}"),
                  flush=True)
        (unfold_root / "csi_robustness.json").write_text(json.dumps(eps_results, indent=2))
        print(f"saved {unfold_root / 'csi_robustness.json'}")
    return 0


def _theta_of(method, ch, rng, ao, dtype, max_iter, tol, unfold_model, dt2):
    """Continuous solution per method (single channel, frozen numerics)."""
    if method == "random":
        return rng.uniform(0.0, 2 * math.pi, ch["h_r"].shape[1])
    if method == "unfold":
        import torch as T

        from src.unfolding import prep_batch

        ctx = prep_batch([ch], device="cpu", dtype=dt2)
        th0 = T.zeros(1, ch["h_r"].shape[1])
        return unfold_model(th0, ctx)[0].detach().numpy()
    ct = ao.prep_channel(ch, device="cpu", dtype=dtype)
    if method == "pg":
        return ao.pg_descent(ct, max_iter=max_iter, tol=tol).theta
    return ao.ao_riemannian(ct, max_iter=max_iter, tol=tol).theta


def _csi_row(eps, i, ch, ao, un, ao_dtype, un_dtype, max_iter, tol, unfold_model):
    """One channel's rates under eps-level CSI error (scored on the TRUE channel)."""
    if eps == 0.0:
        theta = ao.ao_riemannian(ao.prep_channel(ch, device="cpu", dtype=ao_dtype),
                                 max_iter=max_iter, tol=tol).theta
        out = {"ao": rate_np(theta, ch), "random": rate_np(
            np.random.default_rng(3000 + i).uniform(
                0.0, 2 * math.pi, ch["h_r"].shape[1]), ch)}
        if unfold_model is not None:
            ctx = un.prep_batch([ch], device="cpu", dtype=un_dtype)
            th0 = torch.zeros(1, ch["h_r"].shape[1])
            theta_un = unfold_model(th0, ctx)[0].detach().numpy()
            out["unfold"] = rate_np(theta_un, ch)
        return out

    rng_h = np.random.default_rng(10_000 + 137 * i)     # perturbations, deterministic
    ch_hat = perturb_channel(ch, eps, rng_h)
    theta = ao.ao_riemannian(ao.prep_channel(ch_hat, device="cpu", dtype=ao_dtype),
                             max_iter=max_iter, tol=tol).theta
    out = {"ao": rate_np(theta, ch), "random": rate_np(
        np.random.default_rng(3000 + i).uniform(
            0.0, 2 * math.pi, ch["h_r"].shape[1]), ch)}
    if unfold_model is not None:
        ctx_hat = un.prep_batch([ch_hat], device="cpu", dtype=un_dtype)
        th0 = torch.zeros(1, ch_hat["h_r"].shape[1])
        theta_un = unfold_model(th0, ctx_hat)[0].detach().numpy()
        out["unfold"] = rate_np(theta_un, ch)
    return out


if __name__ == "__main__":
    sys.exit(main())
