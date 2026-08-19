"""Unit tests for the residual-error consolidation kernel (memory_kernel.py).

Drives the real observe/check_and_crystallize path on synthetic residuals with a
planted axis, using fake DB and LLM clients — no Postgres, no Ollama. Verifies each
gate of the corrected mechanism (THEORY.md sections 3-5):

- no crystallization below the sample floor;
- no crystallization when fail/success residuals are statistically identical
  (the permutation edge as a false-positive control);
- crystallization fires on a real planted axis, stores a direction aligned with it,
  and the prompt is task-neutral with contrast examples from both extremes;
- the same axis is not crystallized twice (novelty gate);
- axiom scoring is sign-invariant and ranks on-axis queries first.

Also covers the agent-side helpers that changed: clean_prediction's
earliest-occurrence fallback and the memory-conditional surprise prompt.
"""

import numpy as np
import pytest

from src.eigen_memory_agent.agent import _surprise_messages, clean_prediction
from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from conftest import FakeConn, FakeLLM

D = 128
BETA = 1.0
NOISE = 0.05  # per-dim residual noise


@pytest.fixture()
def rig():
    rng = np.random.default_rng(0)
    v_B = np.zeros(D)
    v_B[0] = 1.0  # planted axis (unit vector; noise is isotropic so this is WLOG)
    conn, llm = FakeConn(), FakeLLM()
    kernel = EigenMemoryKernel(conn, llm, min_fail_residuals=25, min_succ_residuals=10)
    return dict(rng=rng, v_B=v_B, conn=conn, llm=llm, kernel=kernel)


def _feed(kernel, rng, v_B, n_fail, n_succ, planted=True, tag=""):
    """Feed synthetic observations: failure residuals carry +-2*beta on v_B when
    planted; success residuals never do."""
    for i in range(n_fail):
        sign = 1.0 if i % 2 == 0 else -1.0
        spike = sign * 2 * BETA * v_B if planted else 0.0
        kernel.observe(
            embedding=rng.normal(size=D),
            residual=spike + rng.normal(scale=NOISE, size=D),
            was_correct=False,
            context_input=f"fail{tag}-{i}-{'A' if sign > 0 else 'B'}",
            prediction="X",
            actual="Y",
        )
    for i in range(n_succ):
        kernel.observe(
            embedding=rng.normal(size=D),
            residual=rng.normal(scale=NOISE, size=D),
            was_correct=True,
            context_input=f"succ{tag}-{i}",
            prediction="Y",
            actual="Y",
        )


def _axiom_inserts(conn):
    return [e for e in conn.executed if "INSERT INTO semantic_core" in e[0]]


def test_no_crystallization_below_sample_floor(rig):
    k = rig["kernel"]
    _feed(k, rig["rng"], rig["v_B"], n_fail=10, n_succ=5)
    k.check_and_crystallize()
    assert not _axiom_inserts(rig["conn"])
    assert not rig["llm"].prompts  # spectral machinery must not even have run
    assert k.detectability_history == []


def test_no_crystallization_when_no_planted_axis(rig):
    """False-positive control: fail and success residuals identically distributed.
    The permutation edge must keep the gate closed across repeated checks."""
    k = rig["kernel"]
    for _ in range(4):
        _feed(k, rig["rng"], rig["v_B"], n_fail=20, n_succ=15, planted=False)
        k.check_and_crystallize()
    assert not _axiom_inserts(rig["conn"])
    # And the telemetry must show why: lambda1 never cleared the edge by the
    # stability-eligible second check.
    assert len(k.detectability_history) >= 2


def test_crystallizes_planted_axis_with_aligned_direction(rig):
    k, v_B, conn, llm = rig["kernel"], rig["v_B"], rig["conn"], rig["llm"]

    # First check: detectable but not yet stable (no previous direction).
    _feed(k, rig["rng"], v_B, n_fail=30, n_succ=15)
    k.check_and_crystallize()
    assert not _axiom_inserts(conn), "stability gate must hold on the first check"

    # Second check: same axis again -> stable -> crystallize.
    _feed(k, rig["rng"], v_B, n_fail=10, n_succ=5, tag="-b2")
    k.check_and_crystallize()

    inserts = _axiom_inserts(conn)
    assert len(inserts) == 1, "exactly one axiom must crystallize"
    axiom_content, stored_vec, strength = inserts[0][1]
    cos = abs(float(np.array(stored_vec) @ v_B))
    assert cos > 0.9, f"stored direction must align with the planted axis (cos={cos:.3f})"
    assert strength > 1.0  # detectability margin lambda1/edge
    assert "RULE:" in axiom_content
    # Only the final RULE: line is stored — the <thought> scaffolding must be
    # stripped, or it gets injected into every future context (review finding).
    assert "<thought>" not in axiom_content
    assert axiom_content.startswith("RULE:")

    # The prompt must be task-neutral and built from both extremes of the axis.
    prompt = llm.prompts[-1]
    assert "arithmetic" not in prompt.lower()
    assert "prime" not in prompt.lower()
    assert "side A" in prompt and "side B" in prompt
    # Each side must be HOMOGENEOUS in planted sign, and the two sides opposite.
    # (Which sign lands on "side A" is arbitrary — an eigenvector's sign is.)
    a_block, _, b_block = prompt.partition("Failures (side B")
    a_tags = {t for t in ("-A |", "-B |") if t in a_block}
    b_tags = {t for t in ("-A |", "-B |") if t in b_block}
    assert len(a_tags) == 1 and len(b_tags) == 1 and a_tags != b_tags, \
        "contrast sides must be homogeneous and opposite extremes"
    assert "succ" in prompt, "matched successes must be included"


def test_only_the_rule_line_is_stored_when_cot_trails_it(rig):
    """Trailing chain-of-thought after the RULE: line must not be stored.

    Regression for the fourth constant/CoT bug class. The guard used to be
    `raw.rpartition("RULE:")[2]`, which keeps the whole *suffix* — so a reply
    that puts the rule first and then keeps rambling stored ~250 chars of CoT
    and injected it into every context the axiom was selected for. This is the
    exact shape the seed-42 Rule-Shift pilot produced
    (results/shift/comparison_results.shift.42.json); the pre-existing test
    missed it because its fake reply ended at the rule.
    """
    k, conn, llm = rig["kernel"], rig["conn"], rig["llm"]
    llm.reply = (
        "<thought>weighing the two sides</thought>\n"
        "RULE: requests and reports route oppositely.\n\n"
        "Wait, let me refine that based on the core distinction between A and B.\n"
        "Side A: Resolved/Awaiting = High urgency; Side B: Pending = Deferred.\n"
        "Side A (Resolved/Awaiting) -> ESCALATE or"
    )

    _feed(k, rig["rng"], rig["v_B"], n_fail=30, n_succ=15)
    k.check_and_crystallize()
    _feed(k, rig["rng"], rig["v_B"], n_fail=10, n_succ=5, tag="-b2")
    k.check_and_crystallize()

    inserts = _axiom_inserts(conn)
    assert len(inserts) == 1, "exactly one axiom must crystallize"
    axiom = inserts[0][1][0]
    assert axiom == "RULE: requests and reports route oppositely.", axiom
    for leak in ("Wait, let me refine", "Side A", "<thought>", "->"):
        assert leak not in axiom, f"CoT scaffolding leaked into the axiom: {leak!r}"


def test_same_axis_not_crystallized_twice(rig):
    k, conn = rig["kernel"], rig["conn"]
    for tag in ("-1", "-2", "-3", "-4"):
        _feed(k, rig["rng"], rig["v_B"], n_fail=15, n_succ=10, tag=tag)
        k.check_and_crystallize()
    assert len(_axiom_inserts(conn)) == 1, "novelty gate must block re-crystallization"


def test_axiom_scoring_is_sign_invariant_and_centered(rig):
    k = rig["kernel"]
    v = np.zeros(D)
    v[3] = 1.0
    rows = [("on-axis rule", v.tolist()), ("off-axis rule", np.roll(v, 1).tolist())]

    q = np.zeros(D)
    q[3] = -2.0  # strongly expressed, NEGATIVE side of the axis
    scored = k.score_axioms(q, rows)
    assert scored[0][1] == "on-axis rule", "negative-side queries must still rank the axis first"
    assert scored[0][0] > scored[1][0]

    # pgvector text literals must parse too.
    scored2 = k.score_axioms(q, [("textual", "[" + ",".join(["0"] * 3 + ["1"] + ["0"] * (D - 4)) + "]")])
    assert scored2[0][0] == pytest.approx(2.0)


def test_clean_prediction_last_line_and_last_occurrence_fallback():
    labels = ["RED", "BLUE", "GREEN"]
    assert clean_prediction("<thought>hmm</thought>\nBLUE", labels) == "BLUE"
    assert clean_prediction("blue.", labels) == "BLUE"
    assert clean_prediction("**GREEN**", labels) == "GREEN"  # markdown-wrapped
    # Truncated CoT with no final label line: pick the LAST label mentioned —
    # the model's most recent hypothesis. Earliest-occurrence re-inherited
    # label-list order whenever the response restated its options first.
    restated = "<thought>The options are RED, BLUE, GREEN. This looks like BLUE bec"
    assert clean_prediction(restated, labels) == "BLUE"
    # Word-boundary: a label inside a longer word must not match.
    assert clean_prediction("I keep a PROFILE of cases", ["FILE"]) != "FILE"
    assert clean_prediction("no labels here", labels) not in labels


def test_surprise_probe_is_memory_conditional():
    msgs = _surprise_messages(47, ["RED", "BLUE", "GREEN"], context="Similar Past Events:\n- Input: 12 -> GREEN")
    assert msgs[0]["role"] == "system"
    assert "Similar Past Events" in msgs[1]["content"]
    assert msgs[1]["content"].rstrip().endswith("Label:"), "label must stay the first generated token"
    # Without context the probe must be unchanged (back-compat).
    bare = _surprise_messages(47)
    assert "Similar" not in bare[1]["content"]
    assert bare[1]["content"] == "Input: 47\nLabel:"
