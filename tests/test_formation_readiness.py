"""Detection marks THAT a change happened; readiness decides WHEN to write.

The v3 seed-42 run failed on this seam. The outcome trigger fired at the shift
-- correctly, on the first DEFER the agent ever saw -- but crystallization then
ran two batches later against a 60-trial window still dominated by PRE-shift
trials, and wrote a rule whose `request` branch was the pre-shift label. By
batch 15 the window had filled with post-change evidence and lam1 had risen to
0.070, its highest of the run, but the mechanism had already spent its one shot.

Readiness separates the two: once a change is detected, hold crystallization
until enough of the contrast window comes from AFTER the change. Detection stays
immediate (it is what stops the agent trusting stale exemplars); only rule
WRITING waits for evidence.
"""

import numpy as np

from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from conftest import NullConn


def _kernel(**kw):
    return EigenMemoryKernel(NullConn(), None, model="m", rng_seed=0,
                             outcome_trigger=True, window=60, **kw)


def _feed(k, n, actual, correct, rng):
    for _ in range(n):
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=correct,
                  context_input="x", prediction=actual, actual=actual)


def test_not_ready_immediately_after_detection():
    """At the moment of detection the window is almost entirely pre-change."""
    k = _kernel(formation_min_post_change=20)
    rng = np.random.default_rng(0)
    _feed(k, 50, "A", True, rng)          # pre-change regime
    _feed(k, 1, "B", False, rng)          # the change: unseen label
    assert k.outcome_change_detected, "setup wrong: should have detected"
    assert not k.formation_ready, (
        "ready to crystallize with 1 post-change trial in a 60-trial window"
    )


def test_ready_once_enough_post_change_evidence_accumulates():
    k = _kernel(formation_min_post_change=20)
    rng = np.random.default_rng(1)
    _feed(k, 50, "A", True, rng)
    _feed(k, 25, "B", False, rng)         # change plus 24 more post-change
    assert k.outcome_change_detected
    assert k.formation_ready, "still not ready after 25 post-change trials"


def test_readiness_is_off_when_the_trigger_is_off():
    """Without outcome detection there is no change point, so no gating.

    The streak-rule and sequential paths must behave exactly as before.
    """
    k = EigenMemoryKernel(NullConn(), None, model="m", rng_seed=0, window=60)
    rng = np.random.default_rng(2)
    _feed(k, 50, "A", True, rng)
    assert k.formation_ready, "readiness gated a path that has no change point"


def test_readiness_does_not_delay_detection_itself():
    """Detection must stay immediate: it is what stops trusting stale exemplars.

    Only rule WRITING waits. If readiness also delayed detection, the agent
    would keep copying stale labels while waiting for evidence.
    """
    k = _kernel(formation_min_post_change=20)
    rng = np.random.default_rng(3)
    _feed(k, 50, "A", True, rng)
    _feed(k, 1, "B", False, rng)
    assert k.outcome_change_detected, "detection was delayed by readiness"
    assert k.outcome_change_at is not None
