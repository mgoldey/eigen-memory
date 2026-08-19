"""Tests for validating a candidate axiom before it is stored and injected.

§9b showed the structural gap this closes: the crystallizer has no notion of
WHEN a rule stopped being true. On the Rule-Shift task the sequential trigger
fired at batch 7 for a shift that lands at batch 11, and wrote an accurate
statement of the PRE-shift rule -- which was false four batches later. Three
such axioms across two seeds, every one with a wrong branch.

Validation makes early firing safe instead of preventing it: score the candidate
rule against the most RECENT trials, and refuse to store it unless it beats what
the agent would have done without it. A rule describing a superseded regime fails
on recent data by construction, which is exactly the signal that separates
"correct when written" from "correct now".

The accept bar is deliberately "beats the status quo on recent evidence", not
"is the true rule" -- the agent has no oracle. What it does have is its own
recent hit rate, which is the honest baseline.
"""

import numpy as np
import pytest

from src.eigen_memory_agent.memory_kernel import _validate_axiom
from conftest import NullConn


class _StubClient:
    """LLM stub: answers each validation item from a fixed label map."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.chat = self  # so .chat.completions.create resolves
        self.completions = self

    def create(self, **kw):
        label = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": label})()})()]})()


def _records(labels, correct):
    """Recent trials: (input, actual label, whether the agent got it right)."""
    return [{"input": f"item-{i}", "actual": a, "was_correct": c}
            for i, (a, c) in enumerate(zip(labels, correct))]


def test_axiom_that_beats_recent_baseline_is_accepted():
    """A rule matching recent ground truth should pass."""
    recent = _records(["DEFER"] * 10, [False] * 10)  # agent got all 10 wrong
    client = _StubClient(["DEFER"])                  # rule gets all 10 right
    ok, acc, base = _validate_axiom(
        "RULE: pending -> DEFER", recent, client, model="m", labels=["DEFER", "FILE"])
    assert ok, f"accuracy {acc} vs baseline {base} should have been accepted"
    assert acc == pytest.approx(1.0)
    # Baseline is the agent's 0.0 hit rate floored at chance (2 labels -> 0.5).
    assert base == pytest.approx(0.5)


def test_stale_axiom_is_rejected():
    """The §9b failure: a rule correct for the PREVIOUS regime.

    Recent trials are all DEFER (post-shift). The candidate answers FILE -- the
    pre-shift label, and precisely what the sequential gate wrote at batch 7.
    It must not be stored.
    """
    recent = _records(["DEFER"] * 10, [True] * 6 + [False] * 4)  # baseline 0.6
    client = _StubClient(["FILE"])                               # rule scores 0.0
    ok, acc, base = _validate_axiom(
        "RULE: pending -> FILE", recent, client, model="m", labels=["DEFER", "FILE"])
    assert not ok, "a rule scoring 0.0 against a 0.6 baseline was accepted"
    assert acc == pytest.approx(0.0)
    assert base == pytest.approx(0.6)


def test_axiom_no_better_than_baseline_is_rejected():
    """Ties go to the status quo: an axiom costs context, so it must earn it."""
    recent = _records(["DEFER"] * 5 + ["FILE"] * 5, [True] * 5 + [False] * 5)
    client = _StubClient(["DEFER"])  # right on 5 of 10 == baseline 0.5
    ok, acc, base = _validate_axiom(
        "RULE: everything -> DEFER", recent, client, model="m",
        labels=["DEFER", "FILE"])
    assert not ok
    assert acc == pytest.approx(base)


def test_validation_uses_only_recent_trials():
    """Only the tail is scored — that is what makes staleness detectable.

    If validation scored the whole history, a rule that was right for most of
    the run would pass even after the regime changed. The window must be the
    recent tail.
    """
    old = _records(["FILE"] * 40, [True] * 40)    # long pre-shift stretch
    new = _records(["DEFER"] * 10, [False] * 10)  # short post-shift tail
    client = _StubClient(["FILE"])                # the stale rule
    ok, acc, _ = _validate_axiom(
        "RULE: pending -> FILE", old + new, client, model="m",
        labels=["DEFER", "FILE"], window=10)
    assert not ok, "stale rule passed because validation looked too far back"
    assert acc == pytest.approx(0.0)


def test_insufficient_recent_data_rejects_rather_than_guesses():
    """With too little recent evidence, refuse rather than accept on noise."""
    recent = _records(["DEFER"] * 2, [False] * 2)
    client = _StubClient(["DEFER"])
    ok, _, _ = _validate_axiom(
        "RULE: x", recent, client, model="m", labels=["DEFER", "FILE"],
        window=10, min_items=5)
    assert not ok, "accepted an axiom on 2 items of evidence"


def test_axiom_must_beat_chance_not_just_the_baseline():
    """Seed 7's real failure: accepted at 0.30 vs a 0.20 baseline, 3 labels.

    Both numbers sit at or below the 0.33 chance line, so the margin was noise.
    An axiom costs context on every future call; clearing a floor-level baseline
    is not evidence it carries information.
    """
    recent = _records(["DEFER"] * 2 + ["FILE"] * 4 + ["ESCALATE"] * 4,
                      [True] * 2 + [False] * 8)          # baseline 0.2
    client = _StubClient(["DEFER", "DEFER", "DEFER", "FILE"])  # ~0.3 accuracy
    ok, acc, base = _validate_axiom(
        "RULE: weak", recent, client, model="m",
        labels=["DEFER", "FILE", "ESCALATE"])
    assert not ok, f"accepted at {acc:.2f} vs {base:.2f} on a 3-label task"
    assert base >= 1 / 3 - 1e-9, "baseline should be floored at chance"


def test_kernel_rejects_a_stale_axiom_end_to_end():
    """Exercise the wiring, not just the scorer.

    The unit tests above call _validate_axiom directly, so they would not catch a
    kernel that never invokes it, or references an attribute it never set. This
    drives a real kernel with validate_axioms=True and asserts nothing reaches
    the store when the candidate rule loses to the recent baseline.
    """
    from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel

    class _Client:
        """Crystallizes a rule that is stale, then answers validation with it."""

        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            content = kw["messages"][-1]["content"]
            # Validation prompts end in "Answer:"; crystallization does not.
            out = "FILE" if content.rstrip().endswith("Answer:") \
                else "RULE: pending -> FILE"
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {"content": out})()})()]})()

    conn, client = NullConn(), _Client()
    k = EigenMemoryKernel(conn, client, model="m", validate_axioms=True,
                          labels=["DEFER", "FILE"], rng_seed=0)
    rng = np.random.default_rng(0)
    # Recent regime is DEFER and the agent is getting them right: baseline 1.0,
    # so the stale FILE rule scores 0.0 and must be rejected.
    for i in range(12):
        v = rng.standard_normal(16)
        k.observe(embedding=v, residual=rng.standard_normal(16), was_correct=True,
                  context_input=f"item-{i}", prediction="DEFER", actual="DEFER")
    k._crystallize(np.ones(16) / 4.0)
    assert conn.inserts == 0, "a stale axiom was stored despite validation"
    assert k.validation_history and not k.validation_history[-1]["accepted"]


def test_llm_failure_rejects_rather_than_storing_unvalidated():
    """If the validation call errors, the axiom must NOT be stored.

    Failing open here would silently restore the pre-validation behaviour, which
    is the bug this whole mechanism exists to prevent.
    """
    class _Boom:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("model unavailable")

    recent = _records(["DEFER"] * 10, [False] * 10)
    ok, _, _ = _validate_axiom(
        "RULE: x", recent, _Boom(), model="m", labels=["DEFER", "FILE"])
    assert not ok, "an axiom was accepted despite the validation call failing"
