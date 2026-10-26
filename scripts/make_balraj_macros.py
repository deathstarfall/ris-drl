#!/usr/bin/env python3
"""
scripts/make_balraj_macros.py -- Balraj Singh (24BCE0531)
=========================================================

Fills paper/sections/balraj_macros.tex from FROZEN ARTIFACTS ONLY.
Honesty rule: a number that cannot be parsed from an artifact stays
TBD -- nothing is ever invented here. (Internally the marker is
__TBD__, but the tex emits bold TBD: bare underscores are illegal in
LaTeX text mode and would break compilation.)

Sources (read-only; produced by gated runs):
  * results/ao/gate2_verdict.txt        -> Ao* macros + GateTwoVerdict
    (written by scripts/run_gate2.py; the report JSON is NOT required --
     this fixes the "__TBD__ after a successful rerun" bug, because the
     verdict txt is what the gate actually writes)
  * results/unfold/*/metrics.json       -> UnfoldGapToAo
    (every rate-loss run is collected and grouped by lr; the LARGEST
     group is the 5-seed protocol set, so lr-grid dirs (lr 0.03 / 0.003)
     and the circular/mse ablations are excluded automatically)
  * results/unfold/latency_float32.json -> UnfoldLatencyMs, UnfoldSpeedup
    (float64 per-channel MEDIAN latency, like-for-like with the AO median;
     UnfoldSpeedup = AoWallMeanMs / UnfoldLatencyMs, e.g. 475/6.732 -> 70.6.
     Parser is SCHEMA-AGNOSTIC: walks the whole JSON and picks the float64
     median-ms leaf)
  * results/bits_sweep.json             -> QaoQuantGapBthree
    (AO-vs-AO-quantized mean-rate gap at B=3, reported as dB-equivalent
     via the standard high-SNR convention 1 bps/Hz == 10*log10(2) ==
     3.0103 dB, i.e. doubling the SNR buys exactly 1 bps/Hz)
  * results/unfold/csi_sweep.json       -> CsiAoZero, CsiAoPone,
    CsiRandomZero, CsiRandomPone
    (eps=0 vs eps=0.1 rows; protocol: decide on h_hat, score on true h)
  * results/ao/generalization.json      -> GenAo, GenPg, GenUnfold, GenRandom
    (unseen-zone means; PG stays TBD until a PG generalization pass
     actually runs -- it never has)
  * results/ao/pg_test_channels.json    -> PgRandomRatio
    (PG test mean / random test mean, the test-set twin of AoRandomRatio;
     guarded: the pg file's ao_mean must match the canonical AO test mean
     within 0.01 or the ratio is not like-for-like and stays TBD)
  * results/ao/gate2_report.json        -> AoWallPNinetyFiveMs (best
     effort; the report has never been produced, so this stays TBD
     until an instrumented AO re-run -- lead item)
  * any pre-existing macros not in the known set are preserved verbatim

Usage:
    python scripts/make_balraj_macros.py [--root REPO_ROOT] [--out TEX_PATH]

Exit code 0 iff every known macro is filled; 1 otherwise (CI-safe).
Parser is encoding-robust: it matches ASCII labels and digits only, so
cp1252/utf-8 mangling of +/- or em-dashes in the verdict cannot break it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

KNOWN = ["AoRandomRatio", "AoRateMean", "AoSD", "AoIterMean",
         "AoWallMeanMs", "GateTwoVerdict", "UnfoldLatencyMs",
         "UnfoldGapToAo", "UnfoldSpeedup",
         "UnfoldRateMean", "UnfoldRateSD",
         "UnfoldLatencyMeanMs", "UnfoldLatencyP95Ms",
         "QaoQuantGapBthree",
         "CsiAoZero", "CsiAoPone", "CsiRandomZero", "CsiRandomPone",
         "NoRisMeanRate", "NoRisSD",
         "GenAo", "GenUnfold", "GenRandom", "PgRandomRatio"]

# why a macro can still be TBD today (printed with the outstanding list)
# NOTE (DD-036): AoWallPNinetyFiveMs and GenPg are NOT emitted at all - removed
# macros whose artifacts do not exist; the paper text hands off those claims
# to v1.1 instead of printing a TBD.
TBD_HINTS = {}

FLOAT_RE = re.compile(r"[0-9]*\.?[0-9]+")
NEWCMD_RE = re.compile(r"^\s*\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")
F64_RE = re.compile(r"f64|float64|fp64|double|complex128", re.I)
F32_RE = re.compile(r"f32|float32|fp32|single|complex64", re.I)
# median first: the AO side (gate2 verdict) reports the MEDIAN wall time,
# so the unfold side must also be a median for a like-for-like ratio.
_MS_PREF = ("median", "p50", "avg", "mean", "latency",
            "forward", "infer", "wall")


# ----------------------------------------------------------------------
# gate2 verdict parser (label -> floats on that line; separator-agnostic)
# ----------------------------------------------------------------------

def _line_floats(text: str, label: str):
    for raw in text.splitlines():
        if label in raw:
            return FLOAT_RE.findall(raw)
    return None


def parse_unfold_rates(path: Path):
    """DD-041 canonical unfold stats + latency from results/unfold/unfold_rates.json
    (DD-009: per-channel rates + per-channel latency on the frozen test set, produced
    by scripts/eval_unfolding.py from the committed checkpoint)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    import numpy as np
    r = np.asarray(d["rates"], dtype=float)
    l = np.asarray(d["latency_ms"], dtype=float)
    out = {
        "UnfoldRateMean": f"{r.mean():.3f}",
        "UnfoldRateSD": f"{r.std(ddof=1):.3f}",
        "UnfoldLatencyMs": f"{np.median(l):.3f}",
        "UnfoldLatencyMeanMs": f"{l.mean():.3f}",
        "UnfoldLatencyP95Ms": f"{np.percentile(l, 95):.3f}",
    }
    print(f"[macros] DD-009 unfold: mean {r.mean():.3f} +/- {r.std(ddof=1):.3f}, "
          f"latency median {np.median(l):.3f} / mean {l.mean():.3f} / p95 "
          f"{np.percentile(l, 95):.3f} ms (n={len(r)}, dtype {d['detail'].get('dtype')})")
    return out


def parse_ratios_frozen(ao_path: Path, rnd_path: Path, pg_path: Path):
    """DD-006 frozen-baseline policy: published ratios use results/baselines/random
    (53.178, n=500, DD-043 corrected pairing), never gate2_verdict's internal
    200-draw random."""
    import numpy as np
    try:
        import numpy as np
        ao = np.asarray(json.loads(ao_path.read_text())["rates"], dtype=float)
        rnd = np.asarray(json.loads(rnd_path.read_text())["rates"], dtype=float)
        out = {"AoRandomRatio": f"{ao.mean() / rnd.mean():.3f}"}
        print(f"[macros] AO/random (frozen baseline): {ao.mean():.3f}/{rnd.mean():.3f} "
              f"= {out['AoRandomRatio']}")
        if pg_path.exists():
            pg = np.asarray(json.loads(pg_path.read_text())["pg_mean"]) \
                if "pg_mean" in json.loads(pg_path.read_text()) else None
            if pg is not None:
                out["PgRandomRatio"] = f"{float(pg) / rnd.mean():.3f}"
                print(f"[macros] PG/random (frozen baseline): {out['PgRandomRatio']}")
        return out
    except Exception as e:
        print(f"[macros] WARNING frozen ratio parse failed: {e}")
        return None


def parse_verdict(path: Path) -> dict:
    out = {}
    txt = path.read_text(encoding="utf-8", errors="replace")

    f = _line_floats(txt, "AO mean rate")
    if f and len(f) >= 2:
        out["AoRateMean"], out["AoSD"] = f[0], f[1]

    f = _line_floats(txt, "AO / random mean ratio")
    if f:
        out["AoRandomRatio"] = f[0]

    f = _line_floats(txt, "iters to terminate")
    if f:
        out["AoIterMean"] = f[0]            # median (== mean: all runs at cap)

    m = re.search(r"median\s+([0-9]*\.?[0-9]+)\s*s/channel", txt)
    if m:
        out["AoWallMeanMs"] = f"{float(m.group(1)) * 1000:.0f}"
    else:
        m = re.search(r"median\s+([0-9]*\.?[0-9]+)\s*ms/channel", txt)
        if m:
            out["AoWallMeanMs"] = f"{float(m.group(1)):.0f}"

    m = re.search(r"G2 VERDICT:\s*([A-Za-z]+)", txt)
    if m:
        out["GateTwoVerdict"] = m.group(1).upper()

    # internal consistency check (warn only -- never silently trust)
    f = _line_floats(txt, "random mean rate")
    if ("AoRandomRatio" in out and "AoRateMean" in out and f):
        recomputed = float(out["AoRateMean"]) / float(f[0])
        if abs(recomputed - float(out["AoRandomRatio"])) > 0.005:
            print(f"[macros] WARNING ratio mismatch: verdict says "
                  f"{out['AoRandomRatio']}, AO/random recomputes to "
                  f"{recomputed:.3f} -- check the verdict artifact")
    return out


# ----------------------------------------------------------------------
# unfolding 5-seed protocol finder (group-by-lr excludes grid + ablations)
# ----------------------------------------------------------------------

def unfold_protocol(root: Path):
    base = root / "results" / "unfold"
    if not base.is_dir():
        return None, []
    runs = []
    for mp in sorted(base.glob("*/metrics.json")):
        try:
            d = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            print(f"[macros] WARNING unreadable metrics: {mp}")
            continue
        if str(d.get("loss", "")).lower() != "rate":
            continue                        # circular / mse ablations out
        runs.append({"lr": float(d.get("lr", 0.01)),
                     "rate": float(d["final_test_rate"]),
                     "path": str(mp)})
    if not runs:
        return None, []
    by_lr = {}
    for r in runs:
        by_lr.setdefault(r["lr"], []).append(r)
    best_lr = max(by_lr, key=lambda k: len(by_lr[k]))   # largest group = protocol
    group = by_lr[best_lr]
    rates = [r["rate"] for r in group]
    mean = sum(rates) / len(rates)
    sd = (sum((x - mean) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 \
        if len(rates) > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": len(rates), "lr": best_lr,
            "winner": max(rates)}, runs


# ----------------------------------------------------------------------
# latency-lever parser (schema-agnostic walk; picks the float64 median-ms)
# ----------------------------------------------------------------------

def _walk_ms(node, keys, tags, out):
    """Recursively collect numeric leaves whose key mentions ms or seconds."""
    if isinstance(node, dict):
        # does this dict itself declare a dtype (e.g. {"dtype": "float64", ...})?
        tag = None
        for dk, dv in node.items():
            if isinstance(dv, str) and re.search(r"dtype|precision|type", dk, re.I):
                if F64_RE.search(dv):
                    tag = "f64"
                elif F32_RE.search(dv):
                    tag = "f32"
        child_tags = tags + ([tag] if tag else [])
        for k, v in node.items():
            kl = str(k).lower()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                joined = "_".join(keys + [kl]).lower()
                ms_hit = "ms" in kl or any("ms" in str(k2).lower() for k2 in keys)
                sec_hit = (not ms_hit) and (joined.endswith("_s")
                                            or kl in ("s", "sec", "seconds"))
                if ms_hit or sec_hit:
                    out.append({"key": kl, "path": joined, "val": float(v),
                                "f64": tag == "f64" or bool(F64_RE.search(joined)),
                                "f32": tag == "f32" or bool(F32_RE.search(joined)),
                                "sec": sec_hit})
            _walk_ms(v, keys + [kl], child_tags, out)
    elif isinstance(node, list):
        for v in node:
            _walk_ms(v, keys, tags, out)


def parse_latency(path: Path):
    """
    Return the float64 per-channel MEDIAN latency in MILLISECONDS from
    results/unfold/latency_float32.json (falls back to mean when no median
    key exists), tolerant to the exact schema.
    Returns None when nothing plausible is found (honesty rule -> TBD).
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    leaves = []
    _walk_ms(d, [], [], leaves)
    cands = [l for l in leaves if not l["f32"]]     # f32 row is portability, not speed
    f64 = [l for l in cands if l["f64"]]
    pool = f64 or cands
    if not pool:
        print(f"[macros] WARNING no ms-valued leaf found in {path}")
        return None
    for pref in _MS_PREF:                            # median beats mean beats p95
        hits = [l for l in pool if pref in l["key"] or pref in l["path"]]
        if hits:
            pool = hits
            break
    leaf = pool[0]
    ms = leaf["val"] * (1000.0 if leaf["sec"] else 1.0)
    if leaf["sec"]:
        print(f"[macros] NOTE latency stored in seconds ('{leaf['path']}'); converted to ms")
    if not 0.2 <= ms <= 5000.0:
        print(f"[macros] WARNING implausible latency {ms} ms from "
              f"'{leaf['path']}' -- ignoring (honesty rule)")
        return None
    print(f"[macros] latency source key '{leaf['path']}' -> {ms:.3f} ms (float64)")
    return ms


# ----------------------------------------------------------------------
# bits sweep: AO-vs-AO-quantized gap at B=3, in dB-equivalent
# ----------------------------------------------------------------------

def parse_bits_sweep(path: Path):
    """AO (infinite precision) minus AO @ 3-bit codebook, as dB-equivalent.
    Convention: 1 bps/Hz == 10*log10(2) == 3.0103 dB (high-SNR regime:
    doubling the SNR buys exactly 1 bps/Hz). Returns None when the
    artifact lacks the expected fields (honesty rule -> TBD)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    ao = ((d.get("detail") or {}).get("methods") or {}).get("ao") or {}
    inf_mean = (ao.get("inf") or {}).get("mean")
    per = ao.get("per_bits") or {}
    b3 = per.get("3") or per.get(3) or {}
    b3_mean = b3.get("mean")
    if inf_mean is None or b3_mean is None:
        print("[macros] WARNING bits_sweep.json lacks detail.methods.ao.inf / per_bits['3']")
        return None
    gap = float(inf_mean) - float(b3_mean)
    if gap < 0:
        print("[macros] WARNING negative quantization gap -- ignoring (honesty rule)")
        return None
    db = gap * (10.0 * math.log10(2.0))
    print(f"[macros] quant gap B=3: {inf_mean:.4f} - {b3_mean:.4f} = {gap:.4f} "
          f"bps/Hz == {db:.3f} dB-equivalent")
    return db


# ----------------------------------------------------------------------
# CSI robustness sweep: eps 0 vs 0.1
# ----------------------------------------------------------------------

def parse_csi_sweep(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    mrm = d.get("mean_rate_by_method") or {}
    eps = d.get("eps") or []
    ao, ra = mrm.get("ao"), mrm.get("random")
    if not ao or not ra or len(ao) != len(eps) or len(ra) != len(eps):
        print("[macros] WARNING csi_sweep.json missing expected fields")
        return None

    def idx(target):
        for i, e in enumerate(eps):
            if abs(float(e) - target) < 1e-9:
                return i
        return None

    i0, i1 = idx(0.0), idx(0.1)
    if i0 is None or i1 is None:
        print("[macros] WARNING csi_sweep.json lacks eps=0 / eps=0.1 rows")
        return None
    print(f"[macros] csi sweep: ao {ao[i0]:.3f} -> {ao[i1]:.3f}, "
          f"random {ra[i0]:.3f} -> {ra[i1]:.3f} (eps 0 -> 0.1)")
    return {"CsiAoZero": f"{ao[i0]:.3f}", "CsiAoPone": f"{ao[i1]:.3f}",
            "CsiRandomZero": f"{ra[i0]:.3f}", "CsiRandomPone": f"{ra[i1]:.3f}"}


# ----------------------------------------------------------------------
# generalization: unseen-zone means (+ PG/random test ratio)
# ----------------------------------------------------------------------

def parse_generalization(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    m = d.get("methods") or {}
    out = {}
    for name, key in (("GenAo", "ao"), ("GenPg", "pg"),
                      ("GenUnfold", "unfolding"), ("GenRandom", "random")):
        try:
            out[name] = f"{float(m[key]['unseen']['mean']):.3f}"
        except (KeyError, TypeError):
            pass                        # e.g. PG unseen never produced
    print(f"[macros] generalization unseen means: {out}")
    return out


def parse_pg_ratio(gen_path: Path, pg_path: Path):
    """PG test mean / random test mean -- the test-set twin of AoRandomRatio.
    DD-006/036 policy: published ratios use the FROZEN random baseline
    (results/baselines/random/eval_rates.json, 53.178 n=500 under the DD-043
    corrected pairing), never an internal random draw. Guard: pg file's
    ao_mean must match the canonical
    AO test mean (from generalization.json) within 0.01, else the channels
    differ and the ratio is not like-for-like -> None."""
    try:
        g = json.loads(gen_path.read_text(encoding="utf-8"))
        p = json.loads(pg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING cannot read gen/pg artifacts: {e}")
        return None
    try:
        ao_ref = float(g["methods"]["ao"]["test"]["mean"])
    except (KeyError, TypeError):
        print("[macros] WARNING generalization.json lacks the AO test mean")
        return None
    rnd_path = gen_path.parents[1] / "baselines" / "random" / "eval_rates.json"
    try:
        import numpy as np
        rnd = float(np.mean(json.loads(rnd_path.read_text(encoding="utf-8"))["rates"]))
    except Exception as e:
        print(f"[macros] WARNING frozen random baseline unreadable ({rnd_path}): {e}")
        return None
    ao_pg = p.get("ao_mean")
    if ao_pg is None or abs(float(ao_pg) - ao_ref) > 0.01:
        print(f"[macros] WARNING pg file ao_mean {ao_pg} != test AO {ao_ref:.3f} "
              f"-- different channel vintage; no PG ratio emitted")
        return None
    ratio = float(p["pg_mean"]) / rnd
    print(f"[macros] PgRandomRatio (frozen baseline): "
          f"{float(p['pg_mean']):.3f} / {rnd:.3f} = {ratio:.3f}")
    return ratio


# ----------------------------------------------------------------------
# gate2 report: AO wall-time p95 (artifact not produced yet -> TBD)
# ----------------------------------------------------------------------

def parse_gate2_report(path: Path):
    """Best-effort p95 wall time (ms) from results/ao/gate2_report.json.
    The file has never been produced, so the exact schema is unknown; we
    walk for ms/second-valued leaves tagged p95 and sanity-check magnitude.
    Returns None (-> TBD) on any doubt."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[macros] WARNING unreadable {path}: {e}")
        return None
    leaves = []
    _walk_ms(d, [], [], leaves)
    cands = [l for l in leaves if "p95" in l["key"] or "p95" in l["path"]]
    wall = [l for l in cands if "wall" in l["path"]] or cands
    if not wall:
        print(f"[macros] WARNING no p95 wall-time leaf found in {path}")
        return None
    leaf = wall[0]
    ms = leaf["val"] * (1000.0 if leaf["sec"] else 1.0)
    if not 0.1 <= ms <= 600000.0:
        print(f"[macros] WARNING implausible p95 {ms} ms from "
              f"'{leaf['path']}' -- ignoring (honesty rule)")
        return None
    print(f"[macros] AO wall p95 source key '{leaf['path']}' -> {ms:.1f} ms")
    return ms


# ----------------------------------------------------------------------
# existing tex: preserve unknown macros (the pasted stub may not be complete)
# ----------------------------------------------------------------------

def load_extra_macros(path: Path):
    extra = []
    REMOVED = {"AoWallPNinetyFiveMs", "GenPg"}  # DD-036: no artifact -> macro deleted, never TBD
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = NEWCMD_RE.match(line)
            if m and m.group(1) not in KNOWN and m.group(1) not in REMOVED:
                extra.append(line.rstrip())
    return extra


def main() -> int:
    ap = argparse.ArgumentParser(description="fill balraj_macros.tex from artifacts")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                    help="repo root (default: parent of scripts/)")
    ap.add_argument("--out", default=None,
                    help="override output tex path (e.g. the paper repo copy)")
    a = ap.parse_args()
    root = Path(a.root)
    out_path = Path(a.out) if a.out else root / "paper" / "sections" / "balraj_macros.tex"

    values = {k: "__TBD__" for k in KNOWN}

    vpath = root / "results" / "ao" / "gate2_verdict.txt"
    if vpath.exists():
        parsed = parse_verdict(vpath)
        values.update(parsed)
        print(f"[macros] gate2_verdict.txt -> {sorted(parsed)}")
    else:
        print(f"[macros] WARNING missing {vpath}")

    proto, runs = unfold_protocol(root)
    if proto:
        print(f"[macros] unfolding rate runs: {len(runs)} total | protocol group: "
              f"n={proto['n']} @ lr={proto['lr']:.4g} | "
              f"mean {proto['mean']:.3f} +/- {proto['sd']:.3f} | "
              f"winner {proto['winner']:.3f}")
    else:
        print("[macros] WARNING no rate-loss unfold metrics found")

    # canonical unfold stats/latency: the DD-009 per-channel artifact (DD-041 protocol)
    dd009 = parse_unfold_rates(root / "results" / "unfold" / "unfold_rates.json")
    if dd009:
        values.update(dd009)

    if values["UnfoldRateMean"] != "__TBD__" and values["AoRateMean"] != "__TBD__":
        values["UnfoldGapToAo"] = f"{100.0 * float(values['UnfoldRateMean']) / float(values['AoRateMean']):.2f}"

    # frozen-baseline ratios (DD-006 policy; overrides gate2's internal random)
    ratios = parse_ratios_frozen(root / "results" / "ao" / "ao_rates.json",
                                 root / "results" / "baselines" / "random" / "eval_rates.json",
                                 root / "results" / "ao" / "pg_test_channels.json")
    if ratios:
        values.update(ratios)

    # bench latency_float32 is RETIRED from the paper (DD-037): keep on disk only

    # ---- no-RIS baseline (DD-log round-3 reviewer request; committed artifact) -
    npath = root / "results" / "baselines" / "no_ris" / "eval_rates.json"
    if npath.exists():
        import numpy as np
        nr = np.asarray(json.loads(npath.read_text(encoding="utf-8"))["rates"], dtype=float)
        values["NoRisMeanRate"] = f"{nr.mean():.3f}"
        values["NoRisSD"] = f"{nr.std(ddof=1):.3f}"
        print(f"[macros] no-RIS baseline: {nr.mean():.3f} +/- {nr.std(ddof=1):.3f} (SD, n={len(nr)})")
    else:
        print(f"[macros] WARNING missing {npath}")

    if values["UnfoldLatencyMs"] != "__TBD__" and values["AoWallMeanMs"] != "__TBD__":
        values["UnfoldSpeedup"] = f"{float(values['AoWallMeanMs']) / float(values['UnfoldLatencyMs']):.1f}"

    # ---- B=3 quantization gap (dB-equivalent) ------------------------------
    bpath = root / "results" / "bits_sweep.json"
    if bpath.exists():
        db = parse_bits_sweep(bpath)
        if db is not None:
            values["QaoQuantGapBthree"] = f"{db:.3f}"
    else:
        print(f"[macros] WARNING missing {bpath}")

    # ---- CSI robustness (eps 0 vs 0.1) --------------------------------------
    cpath = root / "results" / "unfold" / "csi_sweep.json"
    if cpath.exists():
        csi = parse_csi_sweep(cpath)
        if csi:
            values.update(csi)
    else:
        print(f"[macros] WARNING missing {cpath}")

    # ---- generalization means + PG/random ratio -----------------------------
    gpath = root / "results" / "ao" / "generalization.json"
    ppath = root / "results" / "ao" / "pg_test_channels.json"
    if gpath.exists():
        gen = parse_generalization(gpath)
        if gen:
            values.update(gen)
        if ppath.exists():
            ratio = parse_pg_ratio(gpath, ppath)
            if ratio is not None:
                values["PgRandomRatio"] = f"{ratio:.3f}"
        else:
            print(f"[macros] WARNING missing {ppath}")
    else:
        print(f"[macros] WARNING missing {gpath}")

    # ---- AO wall-time p95: artifact has never been produced; the macro is
    # REMOVED from the paper (DD-036). No block here (v1.1 lead).

    extras = load_extra_macros(out_path)
    if extras:
        print(f"[macros] preserving {len(extras)} non-standard macro(s) from the existing tex")

    lines = [
        "% Auto-generated by scripts/make_balraj_macros.py -- DO NOT hand-edit.",
        "% DD-036/037/041 policy: published ratios use the FROZEN random baseline",
        "% (results/baselines/random, 53.178 n=500, DD-043 corrected pairing;",
        "% DD-009 per-channel artifact results/unfold/unfold_rates.json (DD-041",
        "% val-protocol checkpoint; latency measured per-channel, complex128).",
        "% latency_float32.json is retired from the paper (bench context only).",
        "% Sources: results/ao/gate2_verdict.txt (AO side)",
        "%          + results/unfold/unfold_rates.json (DD-009 canonical)",
        "%          + results/bits_sweep.json + csi_sweep.json",
        "%          + results/ao/generalization.json + pg_test_channels.json",
        "% Ayansh, Balraj Singh (24BCE0531); consumed by paper/sections/sec5_2_*.tex.",
    ]
    # LaTeX-safe emission: unfilled macros print bold TBD (never __TBD__ --
    # bare underscores are illegal in text mode and break compilation).
    lines += [f"\\newcommand{{\\{k}}}{{\\textbf{{{values[k] if values[k] != '__TBD__' else 'TBD'}}}}}"
              for k in KNOWN]
    lines += extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("-" * 56)
    for k in KNOWN:
        print(f"  {k:20s} = {values[k]}")
    print("-" * 56)

    outstanding = [k for k in KNOWN if values[k] == "__TBD__"]
    if outstanding:
        print("WARNING - TBD macros outstanding (honesty rule: no invented numbers):")
        for k in outstanding:
            print(f"  - {k}")
            if k in TBD_HINTS:
                print(f"      ({TBD_HINTS[k]})")
        print(f"[macros] wrote {out_path} (incomplete)")
        return 1
    print(f"[macros] wrote {out_path} -- all {len(KNOWN)} macros filled")
    return 0


if __name__ == "__main__":
    sys.exit(main())