"""S0 gate-calibration sweep (docs/NEXT_EXPERIMENT.md): ROC of the REAL
crystallization gate on synthetic residuals. No LLM, no Postgres, no Ollama.

For each cell (snr, n_fail) the sweep runs REPLICATES independent trials of
the actual EigenMemoryKernel (fake DB/LLM, exactly as the unit tests drive it):
feed half the residuals, check, feed the rest, check again — so the stability
gate operates as it does live. A trial "fires" when an axiom INSERT happens.

Planted signal: failure residuals get a +/-beta spike along a fixed axis;
successes are pure noise. beta is scaled so the spike's eigenvalue is
snr x the permutation edge (edge calibrated per n on null data first), i.e.
snr = 1.0 sits exactly at the gate's noise floor.

Claims this calibrates (pre-registered in NEXT_EXPERIMENT.md):
  - specificity: fire rate at snr=0 should be ~0 (the gate refuses noise);
  - sensitivity: fire rate at snr=2 should be >= 0.9 (the gate is not
    vacuously specific — it fires when real structure exists).

Usage: uv run python gate_roc.py          (writes gate_roc.json, ~2-4 min)
"""

import json

import numpy as np

from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from src import paths

D = 768
SNRS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]  # spike eigenvalue as a multiple of the noise edge
N_FAILS = [50, 100, 200]             # failure residuals per trial (succ = half)
REPLICATES = 20
EDGE_CAL_REPS = 10                   # null replicates used to calibrate the edge per n


class _NullConn:
    """Records semantic_core INSERTs; everything else is a no-op."""

    def __init__(self):
        self.inserts = 0

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                if "INSERT INTO semantic_core" in sql:
                    conn.inserts += 1

            def fetchall(self):
                return []

        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


class _NullLLM:
    class _Msg:
        content = "RULE: synthetic."

    def __init__(self):
        outer = self

        class _Completions:
            @staticmethod
            def create(**kw):
                class _Resp:
                    choices = [type("C", (), {"message": outer._Msg()})()]

                return _Resp()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def run_trial(rng, n_fail, beta, fire_seed):
    """One kernel lifecycle: feed half, check, feed rest, check. Returns
    (fired, detectable_at_final_check, cross_check_cos, max lambda1/edge).

    The decomposition matters: the compound gate is detectability AND
    stability, and under realistic noise the stability term (direction
    reproducibility across checks at |cos| > 0.95) binds much harder than
    the permutation edge."""
    conn = _NullConn()
    kernel = EigenMemoryKernel(conn, _NullLLM(), rng_seed=fire_seed)
    v = np.zeros(D)
    v[0] = 1.0
    n_succ = max(15, n_fail // 2)

    def feed(nf, ns, tag):
        for i in range(nf):
            sign = 1 if i % 2 == 0 else -1
            kernel.observe(
                embedding=rng.normal(size=D),
                residual=sign * beta * v + rng.normal(size=D),
                was_correct=False,
                context_input=f"f{tag}-{i}", prediction="X", actual="Y",
            )
        for i in range(ns):
            kernel.observe(
                embedding=rng.normal(size=D),
                residual=rng.normal(size=D),
                was_correct=True,
                context_input=f"s{tag}-{i}", prediction="Y", actual="Y",
            )

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        feed(n_fail // 2, n_succ // 2, "a")
        kernel.check_and_crystallize()
        d1 = kernel.prev_direction.copy() if kernel.prev_direction is not None else None
        feed(n_fail - n_fail // 2, n_succ - n_succ // 2, "b")
        kernel.check_and_crystallize()
        d2 = kernel.prev_direction.copy() if kernel.prev_direction is not None else None

    lam, edge = kernel.detectability_history[-1]
    margins = [l / e for l, e in kernel.detectability_history if e > 0]
    cos = abs(float(d1 @ d2)) if d1 is not None and d2 is not None else 0.0
    return conn.inserts > 0, lam > edge, cos, (max(margins) if margins else 0.0)


def calibrate_edge(rng, n_fail):
    """Mean permutation edge on NULL data at this n (beta=0)."""
    edges = []
    for rep in range(EDGE_CAL_REPS):
        conn = _NullConn()
        kernel = EigenMemoryKernel(conn, _NullLLM(), rng_seed=1000 + rep)
        n_succ = max(15, n_fail // 2)
        for i in range(n_fail):
            kernel.observe(embedding=rng.normal(size=D), residual=rng.normal(size=D),
                           was_correct=False, context_input=f"f{i}", prediction="X", actual="Y")
        for i in range(n_succ):
            kernel.observe(embedding=rng.normal(size=D), residual=rng.normal(size=D),
                           was_correct=True, context_input=f"s{i}", prediction="Y", actual="Y")
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.check_and_crystallize()
        edges.extend(edge for _, edge in kernel.detectability_history)
    return float(np.mean(edges))


def main():
    rng = np.random.default_rng(0)
    results = {"snrs": SNRS, "n_fails": N_FAILS, "replicates": REPLICATES, "cells": {}}

    print(f"{'n_fail':>7} | " + " | ".join(f"snr={s:<9}" for s in SNRS)
          + "   (full-gate fire rate / detectability-only rate, "
          + str(REPLICATES) + " replicates)")
    print("-" * 100)
    for n_fail in N_FAILS:
        edge = calibrate_edge(rng, n_fail)
        # The planted spike adds ~beta^2 to the failure covariance along v, so
        # the contrast eigenvalue is ~beta^2; set beta^2 = snr * edge.
        row = []
        for snr in SNRS:
            beta = float(np.sqrt(snr * edge))
            fires, detects, coss, margins = 0, 0, [], []
            for rep in range(REPLICATES):
                fired, detectable, cos, margin = run_trial(rng, n_fail, beta, fire_seed=rep)
                fires += fired
                detects += detectable
                coss.append(cos)
                margins.append(margin)
            rate = fires / REPLICATES
            row.append(f"{rate:4.2f}/{detects / REPLICATES:4.2f}")
            results["cells"][f"n{n_fail}_snr{snr}"] = {
                "fire_rate": rate,
                "detect_rate": detects / REPLICATES,
                "mean_cross_check_cos": float(np.mean(coss)),
                "mean_margin": float(np.mean(margins)),
                "edge_used": edge,
                "beta": beta,
            }
        print(f"{n_fail:>7} | " + " | ".join(row), flush=True)

    spec = [results["cells"][f"n{n}_snr0.0"]["fire_rate"] for n in N_FAILS]
    results["specificity_pass"] = all(s <= 0.05 for s in spec)
    print(f"\nspecificity (snr=0 full-gate fire rate): {spec} -> "
          f"{'PASS (<= 0.05 everywhere)' if results['specificity_pass'] else 'FAIL'}")
    for target, name in [(2.0, "snr=2"), (8.0, "snr=8")]:
        sens = [results["cells"][f"n{n}_snr{target}"]["fire_rate"] for n in N_FAILS]
        print(f"sensitivity ({name} full-gate fire rate): {sens}")
    # Where does each stage turn on? The gap between detect_rate and fire_rate
    # is the stability gate's contribution (its cost in detection latency).
    print("\nstability is the binding constraint wherever detect_rate >> fire_rate; "
          "see mean_cross_check_cos per cell in gate_roc.json (threshold 0.95)")

    with open(paths.calibration("gate_roc.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote gate_roc.json")


if __name__ == "__main__":
    main()
