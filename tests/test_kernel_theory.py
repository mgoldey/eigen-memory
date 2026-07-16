"""Executable check of the corrected kernel theory (docs/THEORY.md).

A planted-attribute simulation: embeddings have a dominant topic structure and a
sub-dominant binary polarity attribute B; the hidden label flips on B within each
topic (the C1-and-C3 task of docs/C1_C3_TASK.md). Retrieval is nearest-neighbor
against a stored buffer, like the agent's episodic memory. No LLM, no DB — pure
numpy/sklearn.

It demonstrates, quantitatively:

1. The setting is real: B is linearly recoverable from embeddings (C1) while
   nearest-exemplar label-copying is mediocre (C3), and copying *improves* as the
   buffer densifies — RAG's accuracy is a coverage phenomenon.
2. The OLD mechanism claim is false, and backwards: PCA over failure *embeddings*
   does not surface B. Failure-conditioning mildly SUPPRESSES variance along B
   (errors select items where B is weakly expressed), and B stays buried at the
   same rank behind the topic axes.
3. The CORRECTED mechanism works: PCA over failure *residuals* (query minus
   retrieved neighbor) puts B at rank 1 — retrieval cancels the dominant topic
   axes, and failure-conditioning makes the residual's B-component deterministic.
4. Contrastive PCA (failure residuals vs success residuals as background) is the
   robust form: it cancels topic leakage from sparse-buffer retrieval and rescues
   regimes where plain residual-PCA degrades.
5. Detectability needs samples: at ~10 residuals (what the current code
   crystallizes on per batch) the estimated direction is noise; alignment climbs
   toward 1 only past ~100 failures — the BBP/Marchenko-Pastur phase transition,
   and the quantitative reason the observed axioms were garbage.
6. The XOR label is not linearly decodable from embeddings — the spectral stage
   only locates the B *direction*; inducing the flip *rule* is the LLM's job.

Run verbosely with:  uv run pytest tests/test_kernel_theory.py -s
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

SEED = 0
D = 256           # embedding dimension
N = 4000          # corpus size
TOPICS = 8        # dominant clustering directions
TOPIC_SCALE = 10.0
BETA = 1.0        # strength of the polarity attribute B along v_B
SIGMA = 9.0       # total noise norm per vector (per-dim std = SIGMA/sqrt(D))
BUF = 1000        # episodic-buffer size retrieval draws from


def _make_world(seed=SEED, buf=BUF, sigma=SIGMA):
    rng = np.random.default_rng(seed)
    # Orthonormal frame: TOPICS centroid directions + one polarity direction v_B.
    Q, _ = np.linalg.qr(rng.normal(size=(D, TOPICS + 1)))
    v_topics, v_B = Q[:, :TOPICS], Q[:, TOPICS]

    topic = rng.integers(0, TOPICS, size=N)
    b = rng.choice([-1.0, 1.0], size=N)
    X = (
        TOPIC_SCALE * v_topics[:, topic].T
        + BETA * np.outer(b, v_B)
        + rng.normal(scale=sigma / np.sqrt(D), size=(N, D))
    )
    # Hidden label: flips on B within each topic (XOR structure). Within any
    # topic the two polarities always map to different classes.
    label = np.where(b > 0, topic % 3, (topic + 1) % 3)

    # Retrieval: nearest neighbor among a stored subset (the episodic buffer).
    store = rng.choice(N, size=buf, replace=False)
    query = np.setdiff1d(np.arange(N), store)
    nn_local = NearestNeighbors(n_neighbors=1).fit(X[store])
    nn = store[nn_local.kneighbors(X[query], return_distance=False)[:, 0]]

    fail = label[nn] != label[query]  # items where copying the neighbor's label fails
    return dict(
        X=X, topic=topic, b=b, label=label, v_B=v_B,
        query=query, nn=nn, fail=fail,
        d_fail=X[query][fail] - X[nn[fail]],
        d_succ=X[query][~fail] - X[nn[~fail]],
    )


@pytest.fixture(scope="module")
def world():
    return _make_world()


def _vb_alignment(components, v_B):
    """|cos| of each component with v_B, and the rank (1-indexed) of the best."""
    cos = np.abs(np.asarray(components) @ v_B)
    return cos, int(np.argmax(cos)) + 1


def _cpca_top(d_fail, d_succ):
    """Top eigenvector of Cov(failure residuals) - Cov(success residuals)."""
    C = np.cov(d_fail, rowvar=False) - np.cov(d_succ, rowvar=False)
    w, V = np.linalg.eigh(C)
    return V[:, -1], w


def test_setting_is_in_the_eigen_window(world):
    """C1 holds (B linearly recoverable) while copying is mediocre (C3), and
    copy-accuracy is a COVERAGE phenomenon: it rises with buffer density."""
    X, b = world["X"], world["b"]
    q, nn, label = world["query"], world["nn"], world["label"]

    Xtr, Xte, btr, bte = train_test_split(X, b, test_size=0.3, random_state=SEED)
    probe_acc = LogisticRegression(max_iter=2000).fit(Xtr, btr).score(Xte, bte)

    m = float(np.mean(b[nn] == b[q]))               # neighbor polarity-match rate
    copy_acc = float(np.mean(label[nn] == label[q]))  # nearest-exemplar label-copy accuracy

    # Same world, sparser and denser buffers: m (== copy accuracy) must rise
    # with density. Compression pays exactly where coverage is sparse.
    sparse = _make_world(buf=300)
    dense = _make_world(buf=3000)
    m_sparse = float(np.mean(sparse["b"][sparse["nn"]] == sparse["b"][sparse["query"]]))
    m_dense = float(np.mean(dense["b"][dense["nn"]] == dense["b"][dense["query"]]))

    print(f"\nB-probe accuracy: {probe_acc:.3f}  |  m (nn polarity match): {m:.3f}  "
          f"|  copy accuracy: {copy_acc:.3f}")
    print(f"m by buffer size: 300 -> {m_sparse:.3f}, 1000 -> {m:.3f}, 3000 -> {m_dense:.3f}")

    assert probe_acc > 0.85, "C1 fails: B is not linearly recoverable"
    assert m < 0.75, f"C3 fails: copying already works (m={m:.2f})"
    assert abs(copy_acc - m) < 0.05  # same-topic nn + within-topic flip => copy tracks m
    assert m_sparse < m_dense, "copy accuracy must rise with buffer density"


def test_failure_conditioning_does_not_amplify_B(world):
    """The old claim — 'PCA over failure vectors surfaces B' — is false, and
    backwards: failures mildly SUPPRESS variance along v_B (errors concentrate
    on items whose B expression is weak), and v_B stays buried at the same rank
    behind the topic axes."""
    X, v_B = world["X"], world["v_B"]
    q, fail = world["query"], world["fail"]

    var_all = float(np.var(X @ v_B))
    var_fail = float(np.var(X[q][fail] @ v_B))
    ratio = var_fail / var_all

    k = TOPICS + 2
    _, rank_all = _vb_alignment(PCA(n_components=k).fit(X).components_, v_B)
    cos_fail, rank_fail = _vb_alignment(
        PCA(n_components=k).fit(X[q][fail]).components_, v_B
    )

    print(f"\nvar along v_B  fail/all: {ratio:.2f} (suppression, not amplification)")
    print(f"rank of v_B in global PCA: {rank_all}  |  in failure PCA: {rank_fail}")
    print(f"top-3 |cos| with v_B (failure PCA): {np.round(cos_fail[:3], 3)}")

    assert ratio < 1.1, "failure-conditioning must not amplify variance along v_B"
    assert max(cos_fail[:3]) < 0.3, "v_B must not appear in the top failure components"
    assert rank_fail >= rank_all - 1  # no rank improvement from failure-conditioning


def test_residual_pca_isolates_B(world):
    """The corrected mechanism: PCA over failure residuals (x - x_nn) puts B at
    rank 1. Retrieval cancels the dominant topic axes (neighbors are topic-close
    — that is why they were retrieved), and conditioning on failure makes the
    residual's B-component deterministic (+-2*beta)."""
    d_fail, v_B = world["d_fail"], world["v_B"]

    pca = PCA(n_components=5).fit(d_fail)
    cos, rank = _vb_alignment(pca.components_, v_B)
    gap = pca.explained_variance_[0] / pca.explained_variance_[1]

    print(f"\nn failure residuals: {len(d_fail)}")
    print(f"|cos(PC1, v_B)| = {cos[0]:.3f}  (rank of v_B: {rank})  eigengap: {gap:.2f}x")

    assert rank == 1
    assert cos[0] > 0.85, "residual-PCA must isolate the planted B direction"
    assert gap > 1.5


def test_contrastive_residual_pca_is_robust(world):
    """The robust form: cPCA of failure residuals against SUCCESS residuals as
    background. Both share retrieval's topic leakage and noise; only failures
    carry the +-2*beta spike, so the covariance difference isolates v_B — even
    in sparse-buffer regimes where plain residual-PCA degrades."""
    v_B = world["v_B"]

    v, _ = _cpca_top(world["d_fail"], world["d_succ"])
    cos_here = float(abs(v @ v_B))

    # The hard regime: sparse buffer + heavy noise. Plain residual-PCA breaks
    # (topic leakage from far/cross-topic neighbors); cPCA still finds v_B.
    hard = _make_world(buf=300, sigma=12.0)
    plain_hard, _ = _vb_alignment(
        PCA(n_components=3).fit(hard["d_fail"]).components_, hard["v_B"]
    )
    v_hard, _ = _cpca_top(hard["d_fail"], hard["d_succ"])
    cpca_hard = float(abs(v_hard @ hard["v_B"]))

    print(f"\ncPCA |cos| (reference regime): {cos_here:.3f}")
    print(f"hard regime (buf=300, sigma=12): plain residual-PCA {plain_hard[0]:.3f} "
          f"-> cPCA {cpca_hard:.3f}")

    assert cos_here > 0.85
    assert cpca_hard > 0.8, "cPCA must rescue the sparse-buffer regime"
    assert cpca_hard > plain_hard[0] + 0.2, "cPCA must beat plain residual-PCA when retrieval leaks"


def test_success_residuals_have_no_spike(world):
    """Control for the control: success residuals (neighbor matched on B) carry
    no B spike — they are a valid cPCA background set, available for free from
    the same interaction logs."""
    cos, _ = _vb_alignment(
        PCA(n_components=3).fit(world["d_succ"]).components_, world["v_B"]
    )
    print(f"\nsuccess residuals |cos(top PCs, v_B)|: {np.round(cos, 3)}")
    assert max(cos) < 0.3


def test_crystallization_needs_samples(world):
    """Detectability is a phase transition in sample count (BBP / spiked
    covariance): with ~10 residuals — the per-batch count the current code
    crystallizes on — the estimated direction is essentially noise. Alignment
    climbs toward 1 only past ~100 failures. This is the quantitative reason
    crystallizing every batch produced garbage axioms."""
    d_fail, d_succ, v_B = world["d_fail"], world["d_succ"], world["v_B"]
    rng = np.random.default_rng(SEED)

    aligns = {}
    print()
    for n in (10, 25, 100, 400):
        cs = []
        for _ in range(5):  # average over resamples: the small-n story must not hinge on one draw
            f = d_fail[rng.choice(len(d_fail), size=n, replace=False)]
            s = d_succ[rng.choice(len(d_succ), size=n, replace=False)]
            v, _ = _cpca_top(f, s)
            cs.append(abs(float(v @ v_B)))
        aligns[n] = float(np.mean(cs))
        print(f"  n={n:4d} residuals -> mean |cos(v, v_B)| = {aligns[n]:.3f}")

    assert aligns[10] < 0.5, "at n~10 the direction must be (near) noise"
    assert aligns[400] > 0.8, "with enough failures the direction must be found"
    assert aligns[400] > aligns[10] + 0.3


def test_label_is_not_linearly_decodable(world):
    """The XOR objection, quantified: no linear model reads the label off the
    embedding. The spectral stage cannot and need not solve the task — it only
    locates the B direction; the LLM induces the flip rule from examples."""
    X, label = world["X"], world["label"]

    Xtr, Xte, ytr, yte = train_test_split(X, label, test_size=0.3, random_state=SEED)
    label_acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)

    print(f"\nlinear label-probe accuracy: {label_acc:.3f} (vs ~0.95 for the B-probe)")
    assert label_acc < 0.75, "label should not be linearly decodable (XOR structure)"
