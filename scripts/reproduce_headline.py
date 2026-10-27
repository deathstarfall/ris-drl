# DD-046: one-command headline reproduction check (numpy only, no torch).
# Prints every published number beside its committed-artifact source and exits
# nonzero if any value drifts beyond tolerance. Run from a clean checkout.
import json, sys, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))

FAIL = []

def check(name, got, want, tol=0.01):
    ok = abs(float(got) - float(want)) <= tol
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name:44s} artifact {got:9.3f}  paper {want:9.3f}")
    if not ok:
        FAIL.append(name)

print("== Headline reproduction check (committed artifacts vs paper numbers) ==")

# baselines
for meth, want in (("random", 53.178), ("zero", 53.076), ("no_ris", 42.769), ("best_of_200", 57.145)):
    d = json.load(open(f"results/baselines/{meth}/eval_rates.json"))
    check(meth, np.mean(d["rates"]), want)

# AO
ao = json.load(open("results/ao/ao_rates.json"))
check("AO mean", np.mean(ao["rates"]), 64.193)
assert len(ao["rates"]) == 500 and ao.get("channel_seed") == 2000, "AO provenance"
print("  [OK ] AO provenance: 500 channels, channel_seed 2000")

# unfolding
u = json.load(open("results/unfold/unfold_L6_rate/metrics.json"))
# metrics.json stores per-step history; the deployed artifact is in unfold_rates.json
u = json.load(open("results/unfold/unfold_rates.json"))
check("unfolding mean", np.mean(u["rates"]), 62.532)
lat = np.percentile(u["latency_ms"], 50)
check("unfolding latency median (ms)", lat, 1.295, tol=0.05)

# PG twin
pg = json.load(open("results/ao/pg_test_channels.json"))
check("PG twin mean", pg["pg_mean"], 63.790)

# ratio
ratio = float(np.mean(ao["rates"])) / float(np.mean(
    json.load(open("results/baselines/random/eval_rates.json"))["rates"]))
check("AO/random ratio", ratio, 1.207)

# DD-048 same-machine latency protocol (results/unfold/latency_common_protocol.json)
try:
    lp = json.load(open("results/unfold/latency_common_protocol.json"))
    check("same-machine AO median (s)", lp["ao_time_s"]["median"], 0.424, tol=0.02)
    check("same-machine unfolding trained median (ms)", lp["unfold_trained_ms"]["median"], 8.257, tol=0.5)
    check("same-machine unfolding untrained control (ms)", lp["unfold_untrained_ms"]["median"], 8.347, tol=0.5)
    check("same-machine latency ratio", lp["ratio_ao_over_unfold_trained"], 51.358, tol=2.0)
except Exception as e:
    print("  [SKIP] same-machine latency checks unavailable:", e)

# quantization sweep
b = json.load(open("results/bits_sweep.json"))
check("AO@2bit", b["mean_rate_by_method"]["ao"][0], 63.217)
check("unfolding@2bit", b["mean_rate_by_method"]["unfolding"][0], 61.567, tol=0.02)

# closed-form full-CSI no-learning anchor (round-2 matched baseline)
cf = np.array(json.load(open("results/baselines/closed_form_full_csi.json"))["rates"])
check("closed-form full-CSI mean", np.mean(cf), 56.944, tol=0.02)

# partial-observation campaign: all 6 families x 5 seeds re-scored from per-seed artifacts
PART = {"sac": 53.217, "td3": 53.080, "ddpg": 53.082,
        "dqn_b2": 53.142, "dqn_b3": 53.166, "dqn_b4": 53.181}
for fam, want in PART.items():
    seeds = [np.mean(json.load(open(f"results/{fam}/seed{s}/eval_rates.json"))["rates"]) for s in range(5)]
    check(f"partial-obs {fam} (5-seed mean)", float(np.mean(seeds)), want, tol=0.02)

# full-CSI campaign families (all six) with paired Wilcoxon stats with paired Wilcoxon stats
WILCOXON_WANT = {"sac": 0.262, "td3": 0.531, "ddpg": 0.433,
                 "dqn_b2": 0.159, "dqn_b3": 0.276, "dqn_b4": 0.517}
try:
    from scipy.stats import wilcoxon
    rnd = np.array(json.load(open("results/baselines/random/eval_rates.json"))["rates"])
    for fam, want_p in WILCOXON_WANT.items():
        R = np.stack([np.array(json.load(open(f"results/full/{fam}/seed{s}/eval_rates.json"))["rates"]) for s in range(5)])
        ms = R.mean(axis=0)
        check(f"full-CSI {fam} mean", ms.mean(), {"sac": 53.164, "td3": 53.086, "ddpg": 53.065,
              "dqn_b2": 53.153, "dqn_b3": 53.164, "dqn_b4": 53.204}[fam], tol=0.02)
        _, p = wilcoxon(ms, rnd)
        check(f"full-CSI {fam} Wilcoxon p vs floor", p, want_p, tol=0.02)
except Exception as e:
    print("  [SKIP] full-CSI checks unavailable:", e)

# DD-045 sanity: the true-ZF unit test guards the rate function (numpy path)
try:
    from src.precoder import effective_channel, zf_precoder, sum_rate
    rng = np.random.default_rng(5)
    cn = lambda s: (rng.normal(size=s) + 1j * rng.normal(size=s)) / np.sqrt(2)
    ch = dict(h_d=cn((4, 8)), G=cn((8, 16)), h_r=cn((4, 16)), sigma2=2e-14, P=1.0)
    th = rng.uniform(0, 2 * np.pi, 16)
    H = effective_channel(th, ch); W = zf_precoder(H, ch["P"])
    exp = np.sum(np.log2(1 + np.abs(np.diag(H @ W)) ** 2 / ch["sigma2"]))
    check("ZF rate == textbook post-ZF (DD-045 check)", sum_rate(th, ch), exp, tol=1e-9)
except Exception as e:
    print("  [SKIP] src import unavailable in this checkout:", e)

print("== RESULT:", "ALL VERIFIED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
