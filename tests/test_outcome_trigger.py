"""The outcome trigger: detect the rule shift from correctness, not geometry.

Measured on the five persisted Rule-Shift streams (outcome_detector.py):

  detector            fires  pre-shift false fires  median delay
  spectral (streak)   0/5*   0                      --
  outcome per-label   5/5    0                      1 trial

  * recomputed from the committed detectability traces; two seeds had only
    2 eligible checks against a 3-consecutive-detection rule and so could
    not have fired at any signal strength.

Two signals, both anytime-valid so checking every trial is free:
  - a betting e-process on the correctness stream, and
  - an unseen label, which under a stable rule cannot occur (the label set is
    closed), so its appearance IS the change.

On this task the second carries most of the detection, because the shift
introduces a label absent pre-shift. A shift that PERMUTES existing labels
would fall back to the e-process alone. These tests pin both paths.
"""

import numpy as np

from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel


class _Conn:
    def __init__(self):
        self.inserts = 0

    def cursor(self):
        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                pass

            def fetchall(self):
                return []

        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


def _kernel(**kw):
    return EigenMemoryKernel(_Conn(), None, model="m", rng_seed=0,
                             outcome_trigger=True, **kw)


def _observe(k, actual, was_correct, n, rng):
    for _ in range(n):
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=was_correct,
                  context_input="x", prediction=actual, actual=actual)


def test_stable_stream_does_not_trigger():
    """A steady error rate must not fire, however long the stream."""
    k = _kernel()
    rng = np.random.default_rng(0)
    for i in range(120):
        ok = (i % 3) != 0          # steady ~67% correct
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=ok,
                  context_input="x", prediction="A", actual="A")
    assert not k.outcome_change_detected, "fired on a stable stream"


def test_accuracy_collapse_triggers():
    """The e-process path: same labels, accuracy falls off a cliff."""
    k = _kernel()
    rng = np.random.default_rng(1)
    _observe(k, "A", True, 40, rng)      # healthy
    for _ in range(30):                  # collapse, no new label
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=False,
                  context_input="x", prediction="A", actual="A")
    assert k.outcome_change_detected, "did not fire on an accuracy collapse"


def test_novel_label_triggers():
    """The label path: under a stable rule the label set is closed."""
    k = _kernel()
    rng = np.random.default_rng(2)
    _observe(k, "A", True, 40, rng)
    k.observe(embedding=rng.standard_normal(8), residual=rng.standard_normal(8),
              was_correct=False, context_input="x", prediction="A", actual="B")
    assert k.outcome_change_detected, "did not fire on a previously unseen label"


def test_trigger_needs_a_burn_in():
    """Early trials establish the baseline; firing there is meaningless."""
    k = _kernel()
    rng = np.random.default_rng(3)
    for _ in range(5):
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=False,
                  context_input="x", prediction="A", actual="A")
    assert not k.outcome_change_detected, "fired inside the burn-in window"


def test_disabled_by_default():
    """Off unless asked for, so committed results stay reproducible."""
    k = EigenMemoryKernel(_Conn(), None, model="m", rng_seed=0)
    rng = np.random.default_rng(4)
    _observe(k, "A", True, 40, rng)
    k.observe(embedding=rng.standard_normal(8), residual=rng.standard_normal(8),
              was_correct=False, context_input="x", prediction="A", actual="B")
    assert not k.outcome_change_detected


def test_trigger_latches():
    """Once the rule has changed it has changed; recovery must not un-fire it."""
    k = _kernel()
    rng = np.random.default_rng(5)
    _observe(k, "A", True, 40, rng)
    k.observe(embedding=rng.standard_normal(8), residual=rng.standard_normal(8),
              was_correct=False, context_input="x", prediction="A", actual="B")
    assert k.outcome_change_detected
    _observe(k, "B", True, 20, rng)      # agent adapts, accuracy recovers
    assert k.outcome_change_detected, "un-fired after the agent recovered"
