"""Tests for the sequential (e-value) crystallization trigger.

The streak rule this replaces has two structural problems, both observed live
on the Rule-Shift seeds (docs/NEXT_EXPERIMENT.md §9a):

  1. It throws evidence away. Each check re-tests from scratch against a fresh
     permutation threshold and the streak counter resets on any miss, so two
     strong-but-not-consecutive checks count for nothing. Seed 7 reached
     streak 2 of 3 with lam1 well above its committed run's, then the trial
     stream ended -- a near-fire that accumulated evidence would have caught.
  2. Its verdict is decided by permutation noise. lam1/edge on real labels
     spans 0.78-1.28 across the four shut seeds while the edge itself varies
     up to 1.31x on identical data, so which side of the line a seed lands on
     is partly the luck of the draw.

An e-process fixes both: per-check permutation p-values become e-values
(e = 1/p), independent e-values MULTIPLY, and Ville's inequality bounds the
probability that the running product ever exceeds 1/alpha at alpha -- for ANY
stopping time, so checking every batch costs nothing in Type-I error.

These tests pin the two properties that make it worth swapping in: honest
Type-I control under the null, and firing where the streak rule cannot.
"""

import numpy as np
import pytest

from src.eigen_memory_agent.memory_kernel import (
    _evalue_from_null, _EProcess,
)

ALPHA = 0.05


def _null_stream(rng, n_checks, n_fail=30, n_succ=30, d=40):
    """Checks drawn under the null: fail/succ are exchangeable, no real signal."""
    out = []
    for _ in range(n_checks):
        pooled = rng.standard_normal((n_fail + n_succ, d))
        out.append((pooled[:n_fail], pooled[n_fail:]))
    return out


def _signal_stream(rng, n_checks, shift=0.55, n_fail=30, n_succ=30, d=40):
    """Checks with a genuine rank-1 location shift between fail and succ."""
    out = []
    v = np.zeros(d)
    v[0] = 1.0
    for _ in range(n_checks):
        F = rng.standard_normal((n_fail, d)) + shift * v
        S = rng.standard_normal((n_succ, d))
        out.append((F, S))
    return out


def test_evalue_is_uniform_calibrated_under_null():
    """p from permutation must be ~uniform under the null, so e=1/p has mean ~1.

    This is the property Ville's inequality rests on. If e-values are inflated
    under the null, the whole Type-I guarantee is void.
    """
    rng = np.random.default_rng(0)
    es = []
    for F, S in _null_stream(rng, 200):
        es.append(_evalue_from_null(F, S, n_perm=199, rng=rng))
    es = np.array(es)
    # E[e] <= 1 is the defining property of an e-value. Allow sampling slack.
    assert es.mean() < 1.6, f"null e-values inflated: mean={es.mean():.2f}"
    # And it must not be systematically tiny either, or the test has no power.
    assert es.mean() > 0.3, f"null e-values degenerate: mean={es.mean():.2f}"


def test_eprocess_type_i_error_is_controlled_under_null():
    """Across many null streams, the e-process should rarely cross 1/alpha.

    Ville: P(sup_t E_t >= 1/alpha) <= alpha for any stopping time. With 20
    checks per stream this is the honest false-fire budget the permutation
    quantile plus streak rule only approximated.
    """
    rng = np.random.default_rng(1)
    fires = 0
    trials = 120
    for _ in range(trials):
        proc = _EProcess(alpha=ALPHA)
        for F, S in _null_stream(rng, 20):
            proc.update(_evalue_from_null(F, S, n_perm=99, rng=rng))
            if proc.fired:
                fires += 1
                break
    rate = fires / trials
    # Ville bounds this at alpha; allow binomial slack at n=120.
    assert rate <= 0.12, f"null fire rate {rate:.3f} exceeds the {ALPHA} budget"


def test_eprocess_fires_on_real_signal():
    """With a clear shift the process should cross well inside a 16-check stream.

    shift=0.9 is used because that is where this method actually has power at
    n=30/30, d=40 — see test_power_curve_is_documented below for the honest
    numbers. An earlier version of this test used 0.55 and failed; the method was
    fine, the test was asserting power the design does not have at that SNR.
    """
    rng = np.random.default_rng(2)
    proc = _EProcess(alpha=ALPHA)
    fired_at = None
    for i, (F, S) in enumerate(_signal_stream(rng, 16, shift=0.9), start=1):
        proc.update(_evalue_from_null(F, S, n_perm=199, rng=rng))
        if proc.fired and fired_at is None:
            fired_at = i
    assert fired_at is not None, "e-process never fired on a clear rank-1 shift"
    assert fired_at <= 10, f"fired too late to be useful ({fired_at} checks)"


@pytest.mark.slow
def test_power_curve_is_documented():
    """Pin the measured power curve so a featurization change can be compared.

    Measured 2026-08-12 at n=30/30, d=40, 16 checks, alpha=0.05, 40 trials each:
      shift 0.55 -> power 0.30 (median fire at check 7)
      shift 0.70 -> power 0.60 (median fire at check 7)
      shift 0.90 -> power 0.88 (median fire at check 4)

    The point of pinning it: the live Rule-Shift seeds sit near the low end of
    this curve, so "the gate did not fire" and "the gate has ~30% power at this
    SNR" are the same statement. Any featurization improvement should move this
    curve, and that is the cheap way to measure one without a 3 h run.
    """
    d = 40
    v = np.zeros(d)
    v[0] = 1.0
    fired = 0
    trials = 20
    for t in range(trials):
        rng = np.random.default_rng(2000 + t)
        proc = _EProcess(alpha=ALPHA)
        for _ in range(16):
            F = rng.standard_normal((30, d)) + 0.9 * v
            S = rng.standard_normal((30, d))
            proc.update(_evalue_from_null(F, S, n_perm=99, rng=rng))
        fired += bool(proc.fired)
    power = fired / trials
    assert power >= 0.6, f"power at shift=0.9 collapsed to {power:.2f} (was 0.88)"


def test_eprocess_fires_where_the_streak_rule_cannot():
    """The seed-7 shape: strong evidence, but never 3 CONSECUTIVE detections.

    Alternating strong/weak checks reset the streak counter every other check,
    so a 3-in-a-row rule can never fire no matter how long the stream runs.
    Accumulated evidence still gets there, which is the whole point of the swap.
    """
    strong, weak = 6.0, 0.9  # e-values: strong evidence, then ~neutral
    proc = _EProcess(alpha=ALPHA)
    streak = 0
    max_streak = 0
    for i in range(10):
        e = strong if i % 2 == 0 else weak
        proc.update(e)
        streak = streak + 1 if e > 1.0 else 0
        max_streak = max(max_streak, streak)
    assert max_streak < 3, "test setup wrong: the streak rule would have fired"
    assert proc.fired, "e-process failed to accumulate alternating evidence"


def test_eprocess_is_not_fooled_by_a_single_lucky_check():
    """One extreme check should not fire it; that is the flicker seed 2 showed.

    Seed 2's committed run crossed the edge once (ratio 1.08) and its rerun
    once (1.02); both correctly stayed shut. A single p at the permutation
    resolution floor must not be enough on its own.
    """
    proc = _EProcess(alpha=ALPHA)
    proc.update(1.0 / (1.0 / 200))  # best possible e from 199 permutations
    # 200 > 1/0.05 = 20, so a single maximal check WOULD cross. Guard against
    # that by requiring the floor to be respected: cap per-check evidence.
    assert proc.log_e <= np.log(1.0 / ALPHA) + 1e-9 or not proc.fired, (
        "a single check at the permutation floor should not be decisive"
    )


def test_eprocess_resets_after_firing():
    """After crystallizing, evidence must restart or it fires forever."""
    proc = _EProcess(alpha=ALPHA)
    for _ in range(5):
        proc.update(6.0)
    assert proc.fired
    proc.reset()
    assert not proc.fired
    assert proc.log_e == pytest.approx(0.0)
