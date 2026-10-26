"""ZF precoding + sum rate. Convention DD-002:
   H_eff = h_d + (h_r ∘ e^{jθ}) G^H ,  y_k = H_eff[k,:] @ w_k + n_k
"""
import numpy as np

def zf_precoder(H_eff, p_total):
    """W: (M,K), Heff @ W == I_K (zero forcing), sum |W|² == p_total."""
    Hh = H_eff.conj().T                                   # (M,K)
    W  = Hh @ np.linalg.pinv(H_eff @ Hh)                  # ZF (pinv = numerical safety)
    return W * np.sqrt(p_total / np.sum(np.abs(W) ** 2))

def effective_channel(theta, ch):
    """theta: (N,) radians -> H_eff (K,M). SAME formula in Balraj's torch twin."""
    phi = np.exp(1j * np.asarray(theta).reshape(-1))
    return ch["h_d"] + (ch["h_r"] * phi) @ ch["G"].conj().T

def sum_rate(theta, ch):
    Heff = effective_channel(theta, ch)                   # (K,M)
    W    = zf_precoder(Heff, ch["P"])                     # (M,K)
    gain = np.sum(Heff * W.T, axis=1)                     # h_k . w_k (NO conj: y_k = H_eff[k,:] @ w_k; DD-043 fix)
    snr  = np.abs(gain) ** 2 / ch["sigma2"]               # ZF ⇒ SINR = SNR (noise-limited)
    return float(np.sum(np.log2(1.0 + snr)))