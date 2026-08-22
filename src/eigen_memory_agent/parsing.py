import re


def parse_prediction(raw, labels):
    """Extract the predicted label from a raw CoT response.

    Returns (label, used_fallback). Take the last non-empty line stripped of
    punctuation and markdown; if it isn't a valid label, fall back to the LAST
    word-boundary occurrence of any label in the text. Last, not first: CoT
    responses often restate the option list before deciding, so the earliest
    occurrence just re-inherits label-list order (the old RED bias in a new
    coat). Word-boundary, so "FILE" cannot match "PROFILE". Callers should
    track the fallback rate — it fires mostly on truncated responses, and
    truncation frequency varies with context length, i.e. by arm.
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    p = re.sub(r"[^A-Z]", "", lines[-1].upper()) if lines else "ERROR"
    if p in labels:
        return p, False
    hits = [
        (m.start(), lab)
        for lab in labels
        for m in re.finditer(rf"\b{re.escape(lab)}\b", raw.upper())
    ]
    return (max(hits)[1] if hits else p), True


def clean_prediction(raw, labels):
    """Extract the predicted label from raw model output."""
    return parse_prediction(raw, labels)[0]
