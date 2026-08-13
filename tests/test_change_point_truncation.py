"""Drop pre-change records from the contrast window once a change is detected.

The window composition is what produced every stale axiom so far. On seed 42 the
shift lands at trial 100 and the window holds 60 trials, so at the moment
crystallization becomes ready the buffer is still MAJORITY pre-shift:

    readiness   crystallize at   window holds
    20          ~trial 120       40 pre-shift / 20 post-shift
    40          ~trial 140       20 pre-shift / 40 post-shift

The contrast is between failures and successes, and pre-change successes are
successes under the OLD rule -- they pull the success mean toward the old
regime's polarity and are precisely what lets the model write the pre-shift
rule back out. Truncation removes them.

This is not the offline truncation experiment, which crystallized at END of
stream with a naturally full post-shift buffer and found `full` and `truncated`
tied at 5/5. Mid-stream, with the window still half old-regime, truncation is
the difference.
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
                             outcome_trigger=True, window=60, **kw)


def _feed(k, n, actual, correct, rng):
    for _ in range(n):
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8), was_correct=correct,
                  context_input="x", prediction=actual, actual=actual)


def test_pre_change_records_are_dropped_on_detection():
    """Once the change is detected, the old regime's evidence must go."""
    k = _kernel(truncate_at_change=True)
    rng = np.random.default_rng(0)
    # A steady mixed rate, NOT a run of failures -- ten consecutive failures is
    # itself a detectable collapse and would fire the trigger early, truncating
    # before the intended change point.
    for i in range(50):
        _feed(k, 1, "A", (i % 4) != 0, rng)
    n_before = len(k.fail_records) + len(k.succ_records)
    assert n_before > 10, "setup wrong: nothing buffered"
    assert not k.outcome_change_detected, "setup wrong: fired before the change"
    _feed(k, 1, "B", False, rng)          # the change
    assert k.outcome_change_detected
    total = len(k.fail_records) + len(k.succ_records)
    assert total <= 1, (
        f"kept {total} records from before the change; truncation did not fire"
    )


def test_post_change_records_accumulate_normally_after_truncation():
    k = _kernel(truncate_at_change=True)
    rng = np.random.default_rng(1)
    _feed(k, 40, "A", True, rng)
    _feed(k, 1, "B", False, rng)          # change -> truncate
    _feed(k, 25, "B", False, rng)         # post-change evidence
    total = len(k.fail_records) + len(k.succ_records)
    assert total >= 25, f"post-change records not accumulating (have {total})"


def test_truncation_is_off_by_default():
    """Every committed result was produced without it."""
    k = _kernel()
    rng = np.random.default_rng(2)
    _feed(k, 40, "A", True, rng)
    _feed(k, 1, "B", False, rng)
    total = len(k.fail_records) + len(k.succ_records)
    assert total > 10, "truncated despite truncate_at_change=False"


def test_truncation_needs_the_outcome_trigger():
    """Without a detected change point there is nothing to truncate at."""
    k = EigenMemoryKernel(_Conn(), None, model="m", rng_seed=0, window=60,
                          truncate_at_change=True)
    rng = np.random.default_rng(3)
    _feed(k, 40, "A", True, rng)
    total = len(k.fail_records) + len(k.succ_records)
    assert total > 10, "truncated with no change point available"
