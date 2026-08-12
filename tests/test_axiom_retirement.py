"""Tests for re-validating stored axioms and retiring the ones that go stale.

This closes the one gap validation-at-write cannot: seed 42's axiom
"pending -> FILE" was written at batch 7 for a shift landing at batch 11, and
validated at batch 7 it scored 0.80 vs 0.70 HONESTLY, because it was still true.
No accept threshold catches that -- the evidence does not exist yet.

What does catch it is asking the same question later. A rule that described the
previous regime fails on trials drawn from the new one, so re-scoring stored
axioms against the recent tail and retiring the losers turns "correct when
written" into "correct now" as a maintained property.

This is the project's own thesis applied to its own memory: stale entries should
be retired, and until now that applied to exemplars but not to the axioms that
replaced them.
"""

import numpy as np
import pytest

from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel


class _Conn:
    """Captures INSERTs and UPDATEs against semantic_core."""

    def __init__(self, rows=()):
        self.rows = list(rows)          # (id, axiom_content, eigen_vector)
        self.retired = []
        self.inserts = 0

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                s = " ".join(sql.split())
                if "INSERT INTO semantic_core" in s:
                    conn.inserts += 1
                elif "UPDATE semantic_core" in s and "retired" in s:
                    conn.retired.append(params[0] if params else None)
                self._last = s

            def fetchall(self):
                if "FROM semantic_core" in getattr(self, "_last", ""):
                    return [(r[0], r[1]) for r in conn.rows]
                return []

        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


class _Client:
    """Answers validation prompts with a fixed label."""

    def __init__(self, answer):
        self.answer = answer
        self.chat = self
        self.completions = self
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": self.answer})()})()]})()


def _kernel(conn, client, **kw):
    k = EigenMemoryKernel(conn, client, model="m", labels=["DEFER", "FILE"],
                          rng_seed=0, retire_stale_axioms=True, **kw)
    return k


def _feed(k, actual, was_correct, n=10, start=0):
    rng = np.random.default_rng(0)
    for i in range(n):
        k.observe(embedding=rng.standard_normal(8),
                  residual=rng.standard_normal(8),
                  was_correct=was_correct, context_input=f"item-{start+i}",
                  prediction=actual, actual=actual)


def test_axiom_that_went_stale_is_retired():
    """The seed-42 case: true when written, false after the rule changed."""
    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    client = _Client("FILE")          # the stored rule still answers FILE
    k = _kernel(conn, client)
    _feed(k, actual="DEFER", was_correct=False)   # regime is now DEFER
    k.revalidate_axioms()
    assert conn.retired == ["ax1"], "a stale axiom was not retired"


def test_axiom_that_is_still_true_is_kept():
    """Re-validation must not churn through rules that are still earning."""
    conn = _Conn([("ax1", "RULE: pending -> DEFER")])
    client = _Client("DEFER")
    k = _kernel(conn, client)
    _feed(k, actual="DEFER", was_correct=False)   # agent wrong, rule right
    k.revalidate_axioms()
    assert conn.retired == [], "a correct axiom was retired"


def test_no_retirement_without_enough_recent_evidence():
    """Thin evidence must not retire a rule; absence of proof is not proof."""
    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    client = _Client("FILE")
    k = _kernel(conn, client)
    _feed(k, actual="DEFER", was_correct=False, n=3)   # below min_items
    k.revalidate_axioms()
    assert conn.retired == [], "retired on 3 items of evidence"


def test_revalidation_is_a_noop_when_disabled():
    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    client = _Client("FILE")
    k = EigenMemoryKernel(conn, client, model="m", labels=["DEFER", "FILE"],
                          rng_seed=0, retire_stale_axioms=False)
    _feed(k, actual="DEFER", was_correct=False)
    k.revalidate_axioms()
    assert conn.retired == []
    assert client.calls == 0, "made LLM calls despite retirement being off"


def test_llm_failure_does_not_retire():
    """A failed call must not be read as evidence the rule is stale.

    Retiring on error would delete working memory whenever the model blips --
    the opposite of the write path, where failing open is the dangerous
    direction and failing closed is safe.
    """
    class _Boom:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("model unavailable")

    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    k = _kernel(conn, _Boom())
    _feed(k, actual="DEFER", was_correct=False)
    k.revalidate_axioms()
    assert conn.retired == [], "retired an axiom because the LLM call failed"


def test_retiring_frees_the_direction_for_recrystallization():
    """Retirement must release the axis, or the fix can never be written.

    The novelty gate blocks a new axiom whose direction is close to an
    already-consumed one. After a rule is retired for going stale, the CORRECT
    rule for that same failure structure lies along essentially the same axis --
    so leaving the direction consumed would retire the wrong rule and then
    forbid the right one, which is worse than not retiring at all.
    """
    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    client = _Client("FILE")
    k = _kernel(conn, client)
    v = np.zeros(8)
    v[0] = 1.0
    k.consumed_directions.append(v)
    k.axiom_directions = {"ax1": v}
    _feed(k, actual="DEFER", was_correct=False)
    k.revalidate_axioms()
    assert conn.retired == ["ax1"]
    assert not any(abs(float(v @ c)) > 0.9 for c in k.consumed_directions), (
        "the retired axiom's direction is still consumed, so the corrected "
        "rule can never crystallize on that axis"
    )


def test_retired_axioms_are_not_reinjected():
    """Retirement must actually remove the rule from the injection path."""
    conn = _Conn([("ax1", "RULE: pending -> FILE")])
    client = _Client("FILE")
    k = _kernel(conn, client)
    _feed(k, actual="DEFER", was_correct=False)
    k.revalidate_axioms()
    # The SELECT the agent uses must filter on retired.
    assert "retired" in k.axiom_select_sql().lower(), (
        "the injection query does not exclude retired axioms"
    )
