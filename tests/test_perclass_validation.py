"""Per-class scoring and a real accept margin, from the v3 seed-42 failure.

v3 stored a half-stale axiom and scored 0.500 held-out, worse than the 0.556
you get by injecting nothing. The per-class split is the diagnosis:

    class      no axiom   stale axiom
    request       0.359         0.077   <- the class whose rule changed
    report        0.706         0.824

A half-correct rule does not merely fail to help. It destroys the class whose
rule changed while boosting the other, and the two average to something
unremarkable -- which is exactly why validation accepted it at 0.40 against a
0.33 baseline and why retirement never flagged it across four checks.

Two changes tested here:

  per-class   a candidate must not be materially WORSE than baseline on any
              class it makes claims about, so 0.077-vs-0.359 is disqualifying
              however good the other branch looks.
  margin      beating the baseline by one item on a 10-item tail is noise. The
              accept bar requires a margin that scales with the tail size.
"""

import pytest

from src.eigen_memory_agent.memory_kernel import _validate_axiom


class _Stub:
    """Answers each validation item from a label map keyed on the item's input."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.chat = self
        self.completions = self

    def create(self, **kw):
        content = kw["messages"][-1]["content"]
        item = content.rsplit("Input: ", 1)[-1].split("\n")[0].strip()
        out = self.answer_for(item)
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": out})()})()]})()


def _recent(spec):
    """spec: list of (input, actual, was_correct)."""
    return [{"input": i, "actual": a, "was_correct": c} for i, a, c in spec]


def test_half_stale_axiom_is_rejected_on_the_class_it_breaks():
    """The v3 failure: right on `report`, catastrophically wrong on `request`.

    Aggregate accuracy clears the baseline, so only per-class scoring rejects it.
    """
    # These are v3's real numbers: aggregate 0.50 against a 0.40 baseline, which
    # the shipped code ACCEPTED, while `request` scored 0.00.
    spec = ([(f"req{i}", "DEFER", False) for i in range(5)] +
            [(f"rep{i}", "ESCALATE", True) for i in range(4)] +
            [("rep4", "ESCALATE", False)])
    # The stale rule: requests -> FILE (pre-shift label), reports -> ESCALATE.
    stub = _Stub(lambda item: "FILE" if item.startswith("req") else "ESCALATE")
    ok, acc, base = _validate_axiom(
        "RULE: pending -> FILE; completed -> ESCALATE", _recent(spec), stub,
        model="m", labels=["DEFER", "FILE", "ESCALATE"])
    assert acc == pytest.approx(0.5), "setup wrong: aggregate should look fine"
    assert base == pytest.approx(0.4), "setup wrong: should match v3's baseline"
    assert not ok, (
        f"accepted a rule scoring 0.0 on `request` because its aggregate "
        f"({acc:.2f}) beat the baseline ({base:.2f})"
    )


def test_per_class_bar_is_floored_at_chance_not_the_agents_collapse():
    """v3b's failure: the agent has ALREADY collapsed on the changed class.

    Comparing the candidate only to the agent's own per-class rate puts the bar
    on the floor exactly where it matters. Measured on seed 42, the agent scores
    0.00 on DEFER through the whole validation region, so a rule that is also
    0.00 on DEFER counts as "not materially worse" and gets stored -- which is
    what happened, at 0.60 aggregate against a 0.40 baseline.
    """
    # 6 request items the agent gets wrong (as live), 4 report items it gets
    # right. The candidate is stale on request, correct on report.
    spec = ([(f"req{i}", "DEFER", False) for i in range(6)] +
            [(f"rep{i}", "ESCALATE", True) for i in range(4)])
    stub = _Stub(lambda item: "FILE" if item.startswith("req") else "ESCALATE")
    ok, acc, base = _validate_axiom(
        "RULE: pending -> FILE; completed -> ESCALATE", _recent(spec), stub,
        model="m", labels=["DEFER", "FILE", "ESCALATE"])
    assert not ok, (
        f"accepted a rule scoring 0.00 on the changed class because the agent "
        f"also scores 0.00 there (aggregate {acc:.2f} vs {base:.2f})"
    )


def test_uniformly_correct_axiom_still_accepted():
    """Per-class scoring must not block a genuinely good rule."""
    spec = ([(f"req{i}", "DEFER", False) for i in range(5)] +
            [(f"rep{i}", "ESCALATE", False) for i in range(5)])
    stub = _Stub(lambda item: "DEFER" if item.startswith("req") else "ESCALATE")
    ok, acc, base = _validate_axiom(
        "RULE: pending -> DEFER; completed -> ESCALATE", _recent(spec), stub,
        model="m", labels=["DEFER", "FILE", "ESCALATE"])
    assert acc == pytest.approx(1.0)
    assert ok, f"rejected a fully correct rule ({acc:.2f} vs {base:.2f})"


def test_noise_level_margin_is_rejected():
    """v3 accepted 0.40 vs 0.33 on ten items -- one item of difference.

    The bar must scale with the evidence: on a 10-item tail, a single item is
    not grounds for putting a rule into every future prompt.
    """
    # 3 labels -> chance 0.333. Rule gets 4 of 10, uniformly spread so per-class
    # scoring is not what rejects it; the margin is.
    spec = [(f"x{i}", "DEFER", False) for i in range(10)]
    answers = {f"x{i}": ("DEFER" if i < 4 else "FILE") for i in range(10)}
    stub = _Stub(lambda item: answers[item])
    ok, acc, base = _validate_axiom(
        "RULE: weak", _recent(spec), stub, model="m",
        labels=["DEFER", "FILE", "ESCALATE"])
    assert acc == pytest.approx(0.4)
    assert not ok, f"accepted a one-item margin ({acc:.2f} vs {base:.2f})"


def test_clear_margin_is_accepted():
    """A rule that is decisively better must still get through."""
    spec = [(f"x{i}", "DEFER", False) for i in range(10)]
    stub = _Stub(lambda item: "DEFER")
    ok, acc, base = _validate_axiom(
        "RULE: strong", _recent(spec), stub, model="m",
        labels=["DEFER", "FILE", "ESCALATE"])
    assert acc == pytest.approx(1.0)
    assert ok, f"rejected a decisive rule ({acc:.2f} vs {base:.2f})"


def test_class_with_too_few_items_does_not_veto():
    """A class with one or two examples cannot support a per-class judgement.

    Otherwise a single unlucky item in a rare class blocks every candidate.
    """
    spec = ([(f"req{i}", "DEFER", False) for i in range(9)] +
            [("rare0", "ESCALATE", False)])
    stub = _Stub(lambda item: "DEFER" if item.startswith("req") else "FILE")
    ok, acc, base = _validate_axiom(
        "RULE: pending -> DEFER", _recent(spec), stub, model="m",
        labels=["DEFER", "FILE", "ESCALATE"])
    assert acc == pytest.approx(0.9)
    assert ok, "a single-item class vetoed an otherwise strong rule"
