"""Single evaluation harness: baselines + any RL checkpoint + AO/unfolding stats.
   Everyone's test numbers must come out of THIS file (DD-008)."""
import sys, pathlib, json, argparse, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.channel import load_dataset
from src.baselines import run_baselines
from src.precoder import sum_rate

ROOT = pathlib.Path(__file__).resolve().parents[1]

def store(name, rates, extra=None):
    d = {"method": name, "rates": [float(x) for x in rates],
         "n": len(rates), "channel_seed": 2000}
    if extra: d.update(extra)
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_rates.json").with_suffix(".json").write_text(json.dumps(d, indent=2))
    print(f"[store] results/{name}/eval_rates.json  mean={np.mean(rates):.3f}")

def eval_rl(agent, seed, N):
    """Uses Shreyansh's loader (DD-008). Fails cleanly until his agents PR is merged."""
    from src.agents.common import build_policy_fn        # lazy import
    policy = build_policy_fn(agent, seed, N)
    chs = load_dataset("data/test_channels.npz")
    rates = [sum_rate(policy(ch), ch) for ch in chs]     # NO env RNG here: eval is deterministic
    return np.asarray(rates)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baselines", "rl", "ao"])
    ap.add_argument("--agent", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=None)
    args = ap.parse_args()
    chs = load_dataset("data/test_channels.npz")

    if args.method == "baselines":
        t0 = time.time()
        res = run_baselines(chs, callback=lambda i, m:
                            print(f"\r  {i+1}/{len(chs)}", end="") if i % 50 == 0 else None)
        print(f" [{time.time()-t0:.0f}s]")
        for k, v in res.items():
            store(f"baselines/{k}", v, {"description": k})
    elif args.method == "rl":
        rates = eval_rl(args.agent, args.seed, args.N or chs[0]["h_r"].shape[1])
        store(f"{args.agent}/seed{args.seed}", rates, {"agent": args.agent,
              "seed": args.seed, "ckpt": f"ckpt/{args.agent}_seed{args.seed}.pt"})
    elif args.method == "ao":
        # AO rates come from Balraj's own runner; this only re-validates provenance (DD-006)
        f = ROOT / "results" / "ao" / "ao_rates.json"
        d = json.loads(f.read_text())
        assert d["channel_seed"] == 2000 and len(d["rates"]) == len(chs), "AO provenance FAIL"
        print("[ao] provenance OK vs frozen test set:", float(np.mean(d["rates"])))