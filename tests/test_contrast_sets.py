"""The contrast sets fed to the crystallization prompt must span the axis.

`_contrast_sets` picks failures at opposite extremes of the detected direction
and hands them to the LLM as "side A" and "side B" of a hidden axis. If both
sides carry the same label the prompt is degenerate -- it asserts a contrast
that is not in the examples, and the model has to confabulate one.

The bug this pins: the projection was computed on `r["residual"]` regardless of
which space the axis lives in. In `contrast_on="embedding_mean"` mode (the
Rule-Shift configuration) the axis comes from `_reduced_residuals` keyed on
"embedding", so residual-projection ordered the failures by a mostly-unrelated
quantity. Measured on the real seed windows: correlation 0.11-0.46 between the
two orderings, and on seeds 23 and 42 the residual ordering returned six
examples all sharing one label while the embedding ordering did not.
"""

import numpy as np

from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from conftest import NullConn


def _kernel(contrast_on):
    return EigenMemoryKernel(NullConn(), None, model="m", rng_seed=0,
                             contrast_on=contrast_on)


def test_contrast_sets_span_the_axis_in_embedding_mode():
    """Side A and side B must sit at opposite ends of the axis they are told to.

    Construct failures whose EMBEDDINGS separate cleanly along v while their
    RESIDUALS carry the opposite ordering. Projecting the wrong vector picks the
    ends backwards, which is what shipped.
    """
    k = _kernel("embedding_mean")
    d = 8
    v = np.zeros(d)
    v[0] = 1.0
    rng = np.random.default_rng(0)
    for i in range(10):
        pos = (i - 5) / 5.0                    # -1 .. +1 along v
        emb = pos * v + 0.01 * rng.standard_normal(d)
        resid = -pos * v + 0.01 * rng.standard_normal(d)   # deliberately inverted
        k.fail_records.append(
            {"residual": resid, "embedding": emb, "input": f"f{i}",
             "prediction": "X", "actual": "LOW" if pos < 0 else "HIGH", "t": i})

    side_a, side_b, _ = k._contrast_sets(v)
    a_lab = {r["actual"] for r in side_a}
    b_lab = {r["actual"] for r in side_b}
    assert a_lab != b_lab, (
        f"contrast sides do not span the axis: A={a_lab} B={b_lab}. The "
        f"projection is ordering failures by the wrong vector."
    )
    # And specifically: the axis is v in EMBEDDING space, so side A must be the
    # low-embedding end.
    assert a_lab == {"LOW"} and b_lab == {"HIGH"}, (
        f"sides are inverted: A={a_lab} B={b_lab}"
    )


def test_contrast_sets_use_residuals_in_residual_mode():
    """The static-task configuration must keep projecting residuals.

    contrast_on="residual" derives the axis from residuals, so the projection
    must follow it there. This is the arm every committed static-task result was
    produced with; the fix must not silently change it.
    """
    k = _kernel("residual")
    d = 8
    v = np.zeros(d)
    v[0] = 1.0
    rng = np.random.default_rng(1)
    for i in range(10):
        pos = (i - 5) / 5.0
        resid = pos * v + 0.01 * rng.standard_normal(d)
        emb = -pos * v + 0.01 * rng.standard_normal(d)     # inverted
        k.fail_records.append(
            {"residual": resid, "embedding": emb, "input": f"f{i}",
             "prediction": "X", "actual": "LOW" if pos < 0 else "HIGH", "t": i})

    side_a, side_b, _ = k._contrast_sets(v)
    assert {r["actual"] for r in side_a} == {"LOW"}
    assert {r["actual"] for r in side_b} == {"HIGH"}
