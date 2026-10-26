"""DD-043 triage: re-score the 30 existing DRL checkpoints under the CORRECTED
ZF pairing, without touching anything else.

Run AFTER the 4-line conjugation fix in src/precoder.py + src/ao_riemannian.py.
Freezes: checkpoints, frozen test npz, obs normalization, episode structure,
deterministic fixed-order eval. ONLY the rate function differs -> any delta is
attributable to the precoder correction.

Verdict logic printed at the end (Scenario A = null survives / Scenario B =
retrain required). Also re-scores the corrected random/zero/best-200/no-RIS
baselines on the same channels for reference.

Usage (repo root):
    python scripts/triage_zf_rescore.py                 # all agents, seeds 0-4
    python scripts/triage_zf_rescore.py --agents sac td3
    python scripts/triage_zf_rescore.py --agents sac --seeds 0 1 2

Requires ckpt/{agent}_seed{seed}.pt to exist (missing weights -> clean skip
with an explicit marker, NOT a silent pass).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

AGENTS = ["sac", "td3", "ddpg", "dqn_b2", "dqn_b3", "dqn_b4"]
TEST_CHANNELS = ROOT / "data" / "test_channels.npz"
OUT_DIR = ROOT / "results" / "baselines" / "zf_rescore"


def zf_fix_is_in_place() -> bool:
    """Assert the corrected pairing equals diag(H @ W) on a real channel."""
    from src.channel import load_dataset
    from src.precoder import effective_channel, zf_precoder, sum_rate
    ch = load_dataset(str(TEST_CHANNELS))[0]
    th = np.random.default_rng(0).uniform(0, 2 * np.pi, ch["h_r"].shape[1] if "h_r" in ch else 64)
    H = effective_channel(th, ch)
    W = zf_precoder(H, ch["P"])
    good = float(sum_rate(th, ch))
    true = float(np.sum(np.log2(1 + np.abs(np.diag(H @ W)) ** 2 / ch["sigma2"])))
    ok = abs(good - true) < 1e-6 * max(1.0, abs(true))
    print("[triage] ZF-pairing sanity: repo %.4f vs true-diag %.4f -> %s"
          % (good, true, "FIXED" if ok else "STILL BUGGED - aborting"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=AGENTS, choices=AGENTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    if not zf_fix_is_in_place():
        return 2

    from src.channel import load_dataset
    from src.precoder import sum_rate
    from src.agents.common import build_policy_fn
    from src.stats import paired_vs_random

    chs = load_dataset(str(TEST_CHANNELS))
    N = int(chs[0]["h_r"].shape[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # corrected-rate reference baselines (same frozen channels; fixes itself
    # because src.baselines calls the corrected sum_rate)
    from src.baselines import run_baselines
    base = run_baselines(chs, seed=123)
    rnd = np.asarray(base["random"], dtype=float)
    ref = {"random": float(rnd.mean()),
           "zero": float(np.mean(base["zero"])),
           "best200": float(np.mean(base["best_of_200"])) if "best_of_200" in base else None,
           "no_ris": float(np.mean(base["no_ris"]))}
    rnd_comp = json.loads(json.dumps({"rates": list(map(float, rnd))}))
    (OUT_DIR / "baseline_rescored.json").write_text(json.dumps({"refs": ref, "detail": rnd_comp}, indent=1))
    print("[triage] corrected baselines: " + ", ".join(f"{k}={v:.3f}" for k, v in ref.items()))

    rows, verdicts = [], []
    for agent in a.agents:
        for seed in a.seeds:
            ckpt = ROOT / "ckpt" / f"{agent}_seed{seed}.pt"
            key = f"{agent}_seed{seed}"
            if not ckpt.exists():
                print(f"[triage] {key}: ckpt MISSING -> weights required for triage "
                      f"(push ckpt/{agent}_seed{seed}.pt, or Scenario B: retrain)")
                rows.append({"key": key, "status": "ckpt_missing"})
                verdicts.append("MISSING")
                continue
            policy = build_policy_fn(agent, seed, N)
            rates = [float(sum_rate(policy(ch), ch)) for ch in chs]
            t = paired_vs_random(rates, list(rnd), agent.upper())
            row = {"key": key, "mean": float(np.mean(rates)), "sd": float(np.std(rates, ddof=1)),
                   "vs_random_mean": t["mean_gain"], "p_wilcoxon": t["p_wilcoxon"],
                   "ci95": t["ci95"], "frac_positive": 100.0 * t["frac_channels_positive"],
                   "status": "rescored"}
            (OUT_DIR / f"{key}.json").write_text(json.dumps(row | {"rates": rates}, indent=1))
            beats = (t["mean_gain"] > 0 and t["p_wilcoxon"] < 0.05)
            verdicts.append("BEATS_RANDOM" if beats else "NULL_OK")
            rows.append(row)
            print(f"[triage] {key}: mean {row['mean']:.3f} vs corrected-random {ref['random']:.3f} "
                  f"(delta {t['mean_gain']:+.3f}, p={t['p_wilcoxon']:.2e})")

    Path(OUT_DIR / "triage_summary.json").write_text(json.dumps({"rows": rows}, indent=1))
    rescored = [v for v in verdicts if v != "MISSING"]
    if not rescored:
        print("\n[triage] VERDICT: no checkpoints available -> Scenario B (retrain under "
              "corrected objective) is the path.\n")
        return 1
    if any(v == "BEATS_RANDOM" for v in rescored):
        print("\n[triage] VERDICT: SCENARIO B - at least one policy beats the corrected random "
              "baseline -> full retraining campaign under the corrected objective required.\n")
        return 3
    print("\n[triage] VERDICT: SCENARIO A - all rescored policies remain <= corrected random; "
          "the certified null survives at re-evaluation (disclose: retraining under the "
          "corrected objective can be a v1.1 item).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
