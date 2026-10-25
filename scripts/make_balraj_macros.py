#!/usr/bin/env python3
"""
scripts/make_balraj_macros.py — Balraj Singh (24BCE0531)
=========================================================

Fills paper/sections/balraj_macros.tex from FROZEN ARTIFACTS ONLY.
Honesty rule: a number that cannot be parsed from an artifact stays
__TBD__ — nothing is ever invented here.

Sources (read-only; produced by gated runs):
  * results/ao/gate2_verdict.txt        -> Ao* macros + GateTwoVerdict
    (written by scripts/run_gate2.py; the report JSON is NOT required —
     this fixes the "__TBD__ after a successful rerun" bug, because the
     verdict txt is what the gate actually writes)
  * results/unfold/*/metrics.json       -> UnfoldGapToAo
    (every rate-loss run is collected and grouped by lr; the LARGEST
     group is the 5-seed protocol set, so lr-grid dirs (lr 0.03 / 0.003)
     and the circular/mse ablations are excluded automatically)
  * results/unfold/latency_float32.json -> UnfoldLatencyMs, UnfoldSpeedup
    (float64 per-channel MEDIAN latency from the frozen latency-lever run —
     median, because the AO side (gate2 verdict) is also a median, so the
     ratio is like-for-like; mean/p95 are disclosed in DD-029 instead.
     UnfoldSpeedup = AoWallMeanMs / UnfoldLatencyMs, e.g. 475/6.732 -> 70.6.
     The parser is SCHEMA-AGNOSTIC: it walks the whole JSON and picks the
     float64 median-ms leaf, so the exact key layout does not matter)
  * any pre-existing macros not in the known set are preserved verbatim

Usage:
    python scripts/make_balraj_macros.py [--root REPO_ROOT] [--out TEX_PATH]

Exit code 0 iff every known macro is filled; 1 otherwise (CI-safe).
Parser is encoding-robust: it matches ASCII labels and digits only, so
cp1252/utf-8 mangling of ± or em-dashes in the verdict cannot break it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KNOWN = ["AoRandomRatio", "AoRateMean", "AoSD", "AoIterMean",
         "AoWallMeanMs", "GateTwoVerdict", "UnfoldLatencyMs",
         "UnfoldGapToAo", "UnfoldSpeedup"]

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

    # internal consistency check (warn only — never silently trust)
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
    Returns None when nothing plausible is found (honesty rule -> __TBD__).
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
# existing tex: preserve unknown macros (the pasted stub may not be complete)
# ----------------------------------------------------------------------

def load_extra_macros(path: Path):
    extra = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = NEWCMD_RE.match(line)
            if m and m.group(1) not in KNOWN:
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

    if proto and values["AoRateMean"] != "__TBD__":
        values["UnfoldGapToAo"] = f"{100.0 * proto['mean'] / float(values['AoRateMean']):.1f}"

    lpath = root / "results" / "unfold" / "latency_float32.json"
    if lpath.exists():
        ms = parse_latency(lpath)
        if ms is not None:
            values["UnfoldLatencyMs"] = f"{ms:.3f}"
    else:
        print(f"[macros] WARNING missing {lpath}")

    if values["UnfoldLatencyMs"] != "__TBD__" and values["AoWallMeanMs"] != "__TBD__":
        values["UnfoldSpeedup"] = f"{float(values['AoWallMeanMs']) / float(values['UnfoldLatencyMs']):.1f}"

    extras = load_extra_macros(out_path)
    if extras:
        print(f"[macros] preserving {len(extras)} non-standard macro(s) from the existing tex")

    lines = [
        "% Auto-generated by scripts/make_balraj_macros.py -- DO NOT hand-edit.",
        "% Sources: results/ao/gate2_verdict.txt + results/unfold/*/metrics.json",
        "%          + results/unfold/latency_float32.json (f64 MEDIAN, matching",
        "%            the AO median in gate2_verdict.txt), schema-agnostic",
        "% Balraj Singh (24BCE0531); consumed by paper/sections/sec5_2_*.tex.",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{\\textbf{{{values[k]}}}}}" for k in KNOWN]
    lines += extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("-" * 56)
    for k in KNOWN:
        print(f"  {k:16s} = {values[k]}")
    print("-" * 56)

    outstanding = [k for k in KNOWN if values[k] == "__TBD__"]
    if outstanding:
        print("WARNING - __TBD__ macros outstanding (honesty rule: no invented numbers):")
        for k in outstanding:
            print(f"  - {k}")
        print(f"[macros] wrote {out_path} (incomplete)")
        return 1
    print(f"[macros] wrote {out_path} -- all {len(KNOWN)} macros filled")
    return 0


if __name__ == "__main__":
    sys.exit(main())