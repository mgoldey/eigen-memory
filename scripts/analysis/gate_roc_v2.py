"""Gate v2 calibration: evidence accumulation instead of streak-over-max-edge.

Motivation (2026-07-25, seeds 2/18 of the Rule-Shift replication): the v1 gate
requires lam1 to beat the MAX of 20 permutation draws on 3 consecutive checks.
Both gate-shut seeds showed a persistent, direction-stable contrast at 0.6-1.1x
the edge across 3-4 checks — real signal, structurally invisible to a binary
streak that discards every sub-max near-miss. The v1 rule's two free choices
(max-of-20 edge, streak=3) were never derived from a stated error budget.

The v2 rule replaces both with one calibrated quantity:
  - per check, a permutation p-value  p_t = (1 + #{perm lam >= lam_obs}) / (n+1)
    (rank statistic, 200 perms — not a max)
  - evidence  E_t = gamma * E_{t-1} + (-ln p_t),  gamma = 0.8
  - fire when  E_t >= theta  AND stable AND novel
  - theta is chosen HERE, empirically: the smallest value with noise
    (snr=0) full-rule fire rate <= 0.05 at live cadence, across window
    compositions. Overlapping-window dependence between checks (consecutive
    windows share ~50/60 residuals, which correlates even noise directions and
    inflates the stable flag) is baked into the simulation, so theta absorbs it.

V1 and v2 are run PAIRED on identical observation streams, so the comparison
is apples-to-apples per replicate. Cells report fire rate, median checks-to-
fire, and mean lam/edge ratio (the latter locates the live seeds on the curve:
seeds 2/18 logged ratios 0.60-1.10).

This script does not modify src/ — the live replication continues under v1.
Usage: uv run python gate_roc_v2.py     (writes gate_roc_v2.json)
"""

import contextlib
import io
import json

import numpy as np

from gate_roc import _NullConn, _NullLLM
from src.eigen_memory_agent.memory_kernel import (
    EigenMemoryKernel, _mean_contrast, _unit,
)
from src import paths

D = 768
SNRS = [0.0, 0.15, 0.25, 0.35, 0.5, 1.0, 2.0]
N_FAILS = [25, 30, 45]
REPLICATES = 20
NOISE_CAL_REPS = 40             # extra noise replicates for choosing theta
N_BATCHES = 16
BATCH = 10
GAMMA = 0.8
N_PERM_V2 = 200
FP_BUDGET = 0.05                # run-level false-fire budget the threshold buys
KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}


class V2Kernel(EigenMemoryKernel):
    """v1 kernel with check_and_crystallize swapped for the evidence rule.

    Mirrors the parent's flow (floors -> reduce -> contrast -> stability/
    novelty bookkeeping) as of 2026-07-25; only the firing decision differs.
    """

    def __init__(self, *args, gamma=GAMMA, theta=np.inf, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.theta = theta
        self.evidence = 0.0
        self.evidence_history = []  # (E_t, p_t, stable, novel) per check

    def _perm_pvalue(self, F_red, S_red, lam_obs):
        pooled = np.vstack([F_red, S_red])
        n_fail = len(F_red)
        ge = 0
        for _ in range(N_PERM_V2):
            idx = self.rng.permutation(len(pooled))
            lam, _ = _mean_contrast(pooled[idx[:n_fail]], pooled[idx[n_fail:]])
            ge += lam >= lam_obs
        return (1 + ge) / (N_PERM_V2 + 1)

    def check_and_crystallize(self):
        if (len(self.fail_records) < self.min_fail_residuals
                or len(self.succ_records) < self.min_succ_residuals):
            return
        F_red, S_red, basis = self._reduced_residuals()
        lam1, v_red = _mean_contrast(F_red, S_red)
        v_full = _unit(basis.T @ v_red)
        p = self._perm_pvalue(F_red, S_red, lam1)
        self.evidence = self.gamma * self.evidence + (-np.log(p))
        stable = self.prev_direction is not None and (
            abs(float(v_full @ self.prev_direction)) > self.stability_cos
        )
        novel = all(
            abs(float(v_full @ c)) < self.novelty_cos
            for c in self.consumed_directions
        )
        self.prev_direction = v_full
        self.evidence_history.append((self.evidence, p, stable, novel))
        if self.evidence >= self.theta and stable and novel:
            self._crystallize(v_full, strength=self.evidence / self.theta)


def _gen_stream(rng, n_fail, beta):
    """Pre-generate one observation stream so v1/v2 see identical data."""
    v = np.zeros(D)
    v[0] = 1.0
    p_fail = n_fail / KCFG["window"]
    stream = []
    for b in range(N_BATCHES):
        batch = []
        for i in range(BATCH):
            is_fail = rng.random() < p_fail
            sign = 1.0 if is_fail else -1.0
            batch.append((rng.normal(size=D) + sign * (beta / 2) * v,
                          rng.normal(size=D), is_fail, f"{b}-{i}"))
        stream.append(batch)
    return stream


def _feed(kernel, stream):
    with contextlib.redirect_stdout(io.StringIO()):
        for batch in stream:
            for emb, res, is_fail, tag in batch:
                kernel.observe(embedding=emb, residual=res,
                               was_correct=not is_fail, context_input=tag,
                               prediction="X", actual="Y" if is_fail else "X")
            kernel.check_and_crystallize()


def _run_pair(stream, rep, theta):
    """Feed the same stream to a v1 and a v2 kernel; return per-rule outcome."""
    conn1 = _NullConn()
    k1 = EigenMemoryKernel(conn1, _NullLLM(), rng_seed=rep, **KCFG)
    _feed(k1, stream)
    conn2 = _NullConn()
    k2 = V2Kernel(conn2, _NullLLM(), rng_seed=rep, theta=theta, **KCFG)
    _feed(k2, stream)
    checks_to_fire = next(
        (i + 1 for i, (e, _, s, n) in enumerate(k2.evidence_history)
         if e >= theta and s and n), None)
    ratio = (np.mean([l / e for l, e in k1.detectability_history if e > 0])
             if k1.detectability_history else None)
    return conn1.inserts > 0, conn2.inserts > 0, checks_to_fire, ratio


def calibrate_theta(rng):
    """Smallest theta with noise full-rule fire rate <= FP_BUDGET everywhere.

    For each noise replicate, record the max evidence over checks where
    stable AND novel also held (the level a threshold must exceed for that
    run to fire) — the (1 - FP_BUDGET) quantile of the worst n_fail cell is
    the calibrated theta.
    """
    worst = 0.0
    per_cell = {}
    for n_fail in N_FAILS:
        maxes = []
        for rep in range(NOISE_CAL_REPS):
            stream = _gen_stream(rng, n_fail, 0.0)
            k = V2Kernel(_NullConn(), _NullLLM(), rng_seed=10_000 + rep,
                         theta=np.inf, **KCFG)
            _feed(k, stream)
            armed = [e for e, _, s, n in k.evidence_history if s and n]
            maxes.append(max(armed) if armed else 0.0)
        q = float(np.quantile(maxes, 1 - FP_BUDGET))
        per_cell[n_fail] = q
        worst = max(worst, q)
        print(f"  theta cal n_fail={n_fail}: q{100 * (1 - FP_BUDGET):.0f}"
              f"(armed max-E)={q:.2f}", flush=True)
    return worst * 1.001, per_cell


def main():
    rng = np.random.default_rng(0)
    print(f"calibrating theta on noise ({NOISE_CAL_REPS} reps/cell, "
          f"budget {FP_BUDGET})...")
    theta, theta_cells = calibrate_theta(rng)
    print(f"theta = {theta:.2f}  (gamma={GAMMA}, null stationary mean "
          f"~= {1 / (1 - GAMMA):.1f})\n")

    # beta from snr via a null edge reference, mirroring gate_roc_mean.py
    results = {"theta": theta, "theta_cells": theta_cells, "gamma": GAMMA,
               "n_perm_v2": N_PERM_V2, "fp_budget": FP_BUDGET,
               "snrs": SNRS, "n_fails": N_FAILS, "replicates": REPLICATES,
               "kernel_cfg": KCFG, "cells": {}}
    edge_ref = json.load(open(paths.calibration("gate_roc_mean.json")))["cells"]
    print(f"{'n_fail':>7} | " + " | ".join(f"snr={s:<5}" for s in SNRS)
          + "   (v1 fire / v2 fire / lam-edge ratio)")
    print("-" * 110)
    for n_fail in N_FAILS:
        edge = edge_ref[f"n{n_fail}_snr0.0"]["edge_used"]
        row = []
        for snr in SNRS:
            beta = float(np.sqrt(snr * edge))
            v1 = v2 = 0
            ctf, ratios = [], []
            for rep in range(REPLICATES):
                stream = _gen_stream(rng, n_fail, beta)
                f1, f2, c, r = _run_pair(stream, rep, theta)
                v1 += f1
                v2 += f2
                if c is not None:
                    ctf.append(c)
                if r is not None:
                    ratios.append(r)
            cell = {"v1_fire": v1 / REPLICATES, "v2_fire": v2 / REPLICATES,
                    "median_checks_to_fire": float(np.median(ctf)) if ctf else None,
                    "mean_lam_edge_ratio": float(np.mean(ratios)) if ratios else None,
                    "beta": beta}
            results["cells"][f"n{n_fail}_snr{snr}"] = cell
            row.append(f"{cell['v1_fire']:.2f}/{cell['v2_fire']:.2f}/"
                       f"{cell['mean_lam_edge_ratio']:.2f}")
            print(f"{n_fail:>7} | " + " | ".join(row), flush=True)
        print()

    spec = [results["cells"][f"n{n}_snr0.0"]["v2_fire"] for n in N_FAILS]
    results["v2_specificity_pass"] = all(s <= FP_BUDGET for s in spec)
    print(f"v2 specificity (snr=0 fire rate): {spec} -> "
          f"{'PASS' if results['v2_specificity_pass'] else 'FAIL'}")
    with open(paths.calibration("gate_roc_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote gate_roc_v2.json")


if __name__ == "__main__":
    main()
