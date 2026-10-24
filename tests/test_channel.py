import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.channel import save_dataset, load_dataset, noise_sigma2, _cfg

def test_noise_matches_hand_derivation():
    """-174 + 10log10(1e6) + 7 = -107 dBm -> 10^(-13.7) W. docs/link_budget.md cross-check."""
    assert abs(noise_sigma2(_cfg()["power"]) - 10 ** (-13.7)) < 1e-16

def test_npz_roundtrip_deterministic(tmp_path):
    p = tmp_path / "t.npz"
    n = save_dataset("t.npz" if False else p, n=5, seed=7)
    a = load_dataset(p); b = load_dataset(p)
    assert n == 5 and len(a) == 5
    for i in range(5):
        assert np.allclose(a[i]["h_r"], b[i]["h_r"])        # same file -> same channels
        assert a[i]["h_r"].shape == (4, 64)