from src.eigen_memory_agent.agent import _extract_nll, MISSING_TOKEN_NLL


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
