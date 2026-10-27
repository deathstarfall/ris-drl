import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.precoder import zf_precoder, effective_channel, sum_rate

def _mk_ch(rng, K=4, M=8, N=16):
    cn  = lambda s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / np.sqrt(2)
    return dict(h_d=cn((K, M)), G=cn((M, N)), h_r=cn((K, N)), sigma2=2e-14, P=1.0)

def test_zf_power_exact():
    rng, ch = np.random.default_rng(0), None
    ch = _mk_ch(rng)
    Heff = effective_channel(rng.uniform(0, 2*np.pi, 16), ch)
    W = zf_precoder(Heff, ch["P"])
    assert np.isclose(np.sum(np.abs(W)**2), ch["P"], rtol=1e-9)

def test_zf_is_zero_forcing():
    rng = np.random.default_rng(1); ch = _mk_ch(rng)
    Heff = effective_channel(rng.uniform(0, 2*np.pi, 16), ch)
    W = zf_precoder(Heff, ch["P"])
    HW = Heff @ W
    c = HW[0, 0]
    assert np.allclose(HW, c * np.eye(4), atol=1e-5)
    assert abs(c) > 0            # ZF actually equalizes something

def test_rate_matches_bruteforce():
    """vectorized effective channel == per-element brute-force loop (DD-002 cross-check)"""
    rng = np.random.default_rng(2); ch = _mk_ch(rng)
    theta = rng.uniform(0, 2*np.pi, 16); phi = np.exp(1j*theta)
    Heff_vec = effective_channel(theta, ch)
    Heff_bf = np.zeros_like(Heff_vec)
    for k in range(4):
        Heff_bf[k] = ch["h_d"][k] + sum(phi[n] * ch["h_r"][k, n] * np.conj(ch["G"][:, n])
                                        for n in range(16))
    assert np.allclose(Heff_vec, Heff_bf, atol=1e-12)
    assert np.isfinite(sum_rate(theta, ch))

def test_sum_rate_is_true_zf():
    """DD-045 (external review round 3): the shared rate function must equal the
    textbook post-ZF sum rate |<h_k, w_k>|^2/sigma2 with the *non-conjugated*
    inner product. The line below is deliberately written WITHOUT using
    src.sum_rate, so a future conjugation regression cannot hide behind a twin:
    the pre-DD-043 bug produced |<h, w*>|^2 which this test fails on."""
    rng = np.random.default_rng(5); ch = _mk_ch(rng)
    th = rng.uniform(0, 2 * np.pi, 16)
    H = effective_channel(th, ch)
    W = zf_precoder(H, ch["P"])
    exp = np.sum(np.log2(1 + np.abs(np.diag(H @ W))**2 / ch["sigma2"]))
    assert abs(sum_rate(th, ch) - exp) < 1e-9

def test_direct_link_alone_rates_positive():
    """no-RIS path must still give a positive, finite rate (sanity of the whole pipeline)"""
    rng = np.random.default_rng(3); ch = _mk_ch(rng); ch["h_r"] = np.zeros_like(ch["h_r"])
    assert sum_rate(np.zeros(16), ch) > 0.0