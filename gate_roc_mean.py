"""Gate-ROC for the AMENDED Rule-Shift estimator (two-sample mean contrast).

The 2026-07-17 G3 amendment replaced query-embedding cPCA with a mean contrast
because a location difference between failures and successes is invisible to
covariance contrasts. This sweep calibrates that amended estimator in the
PILOT'S EXACT kernel configuration (window=60, consecutive_detections=3,
stability_cos=0.5) the same way gate_roc.py calibrated the static-task gate —
so the amendment is a measured operating characteristic, not a hand-wave.

Planted signal: failure EMBEDDINGS mean-shifted +beta/2 along a fixed axis,
successes -beta/2 (the polarity split the shift creates). beta^2 = snr x the
permutation edge calibrated on null data. Each trial feeds 4 rounds with a
check per round, so the 3-consecutive-detections streak can fire as it does
live. A trial "fires" when an axiom INSERT happens (the _NullLLM returns a
clean RULE line, which also exercises the new no-scaffolding guard).

Usage: uv run python gate_roc_mean.py     (writes gate_roc_mean.json)
"""

import contextlib
import io
import json

import numpy as np

from gate_roc import _NullConn, _NullLLM
from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel

D = 768
SNRS = [0.0, 0.5, 1.0, 2.0, 4.0]
N_FAILS = [25, 30, 45]          # per-window failure counts around the pilot's ~30
REPLICATES = 20
EDGE_CAL_REPS = 10
N_BATCHES = 16                  # live cadence: 16 batches of 10, check per batch
BATCH = 10
KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}


def _make_kernel(rng_seed):
    conn = _NullConn()
    return conn, EigenMemoryKernel(conn, _NullLLM(), rng_seed=rng_seed, **KCFG)


def _observe(kernel, rng, is_fail, beta, v, tag):
    sign = 1.0 if is_fail else -1.0
    kernel.observe(embedding=rng.normal(size=D) + sign * (beta / 2) * v,
                   residual=rng.normal(size=D), was_correct=not is_fail,
                   context_input=tag, prediction="X", actual="Y" if is_fail else "X")


def run_trial(rng, n_fail, beta, fire_seed):
    """Live cadence: 16 batches of 10 with a check per batch; the window keeps
    the expected per-window failure count at n_fail. Streaks, floors, and
    window trimming all operate exactly as in the pilot."""
    conn, kernel = _make_kernel(fire_seed)
    v = np.zeros(D)
    v[0] = 1.0
    p_fail = n_fail / KCFG["window"]
    with contextlib.redirect_stdout(io.StringIO()):
        for b in range(N_BATCHES):
            for i in range(BATCH):
                _observe(kernel, rng, rng.random() < p_fail, beta, v, f"{b}-{i}")
            kernel.check_and_crystallize()
    if not kernel.detectability_history:
        return False, False
    lam, edge = kernel.detectability_history[-1]
    return conn.inserts > 0, lam > edge


def calibrate_edge(rng, n_fail):
    """Mean permutation edge on NULL data at this window composition."""
    edges = []
    for rep in range(EDGE_CAL_REPS):
        _, kernel = _make_kernel(1000 + rep)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(n_fail):
                _observe(kernel, rng, True, 0.0, np.zeros(D), f"cf{i}")
            for i in range(KCFG["window"] - n_fail):
                _observe(kernel, rng, False, 0.0, np.zeros(D), f"cs{i}")
            kernel.check_and_crystallize()
        edges.extend(e for _, e in kernel.detectability_history)
    return float(np.mean(edges))


def main():
    rng = np.random.default_rng(0)
    results = {"snrs": SNRS, "n_fails": N_FAILS, "replicates": REPLICATES,
               "kernel_cfg": KCFG, "cells": {}}
    print(f"{'n_fail':>7} | " + " | ".join(f"snr={s:<7}" for s in SNRS)
          + f"   (fire rate / final-check detect rate, {REPLICATES} replicates)")
    print("-" * 90)
    for n_fail in N_FAILS:
        edge = calibrate_edge(rng, n_fail)
        row = []
        for snr in SNRS:
            beta = float(np.sqrt(snr * edge))
            fires = detects = 0
            for rep in range(REPLICATES):
                fired, detectable = run_trial(rng, n_fail, beta, fire_seed=rep)
                fires += fired
                detects += detectable
            row.append(f"{fires / REPLICATES:4.2f}/{detects / REPLICATES:4.2f}")
            results["cells"][f"n{n_fail}_snr{snr}"] = {
                "fire_rate": fires / REPLICATES,
                "detect_rate": detects / REPLICATES,
                "edge_used": edge, "beta": beta,
            }
        print(f"{n_fail:>7} | " + " | ".join(row), flush=True)

    spec = [results["cells"][f"n{n}_snr0.0"]["fire_rate"] for n in N_FAILS]
    results["specificity_pass"] = all(s <= 0.05 for s in spec)
    print(f"\nspecificity (snr=0 fire rate): {spec} -> "
          f"{'PASS (<= 0.05 everywhere)' if results['specificity_pass'] else 'FAIL'}")
    with open("gate_roc_mean.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote gate_roc_mean.json")


if __name__ == "__main__":
    main()
