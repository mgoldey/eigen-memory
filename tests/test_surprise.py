import os

import pytest

from src.eigen_memory_agent.agent import (
    _extract_nll,
    _surprise_messages,
    MISSING_TOKEN_NLL,
)


class _TL:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


def test_nll_found_token_returns_negative_logprob():
    tops = [_TL("RED", -0.1), _TL("BLUE", -2.0)]
    assert _extract_nll(tops, "RED") == 0.1


def test_nll_case_insensitive_substring_match():
    tops = [_TL(" Blue", -1.5)]
    assert _extract_nll(tops, "BLUE") == 1.5


def test_nll_missing_token_returns_high_finite():
    # The bug case: true label absent from top-k must not crash or collapse to a
    # flat fallback. It returns a high, finite, well-defined surprise.
    tops = [_TL("GREEN", -0.1), _TL("RED", -2.0)]
    s = _extract_nll(tops, "BLUE")
    assert s == MISSING_TOKEN_NLL
    assert 0 < s < 1e9


def test_surprise_messages_constrain_to_single_label():
    # The probe must force a one-word label answer (so the label is the first
    # token). A system message must pin the output to RED/BLUE/GREEN.
    msgs = _surprise_messages(47)
    assert msgs[0]["role"] == "system"
    assert all(c in msgs[0]["content"] for c in ("RED", "BLUE", "GREEN"))
    assert "47" in msgs[1]["content"]


def _ollama_up():
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
def test_predictive_surprise_varies_with_constrained_prompt():
    # Regression test for the chat-vs-completion bug: NLL must NOT be a constant
    # (the old prompt returned MISSING_TOKEN_NLL for every item). With the
    # constrained prompt, the true label appears in top-k and NLL varies.
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    nlls = []
    for query, true in [(47, "RED"), (50, "BLUE"), (12, "GREEN"), (7, "RED")]:
        res = client.chat.completions.create(
            model="gemma3:4b",
            messages=_surprise_messages(query),
            logprobs=True,
            top_logprobs=10,
            max_tokens=1,
        )
        tops = res.choices[0].logprobs.content[0].top_logprobs
        nlls.append(_extract_nll(tops, true))
    # Not all identical, and none collapsed to the missing-token fallback.
    assert len(set(round(n, 2) for n in nlls)) > 1, f"NLLs are constant: {nlls}"
    assert not all(n == MISSING_TOKEN_NLL for n in nlls), f"all missing: {nlls}"
