"""Residual-error spectral consolidation kernel (docs/THEORY.md).

This replaces the original kernel, which implemented a mechanism the theory review
disproved (PCA over raw failure embeddings, crystallize-every-batch, cosine-to-
eigenvector axiom retrieval, an arithmetic-hint prompt). The corrected mechanism:

- **Input = retrieval residuals** (query embedding minus retrieved-neighbor embedding),
  split by outcome into failure and success buffers. Retrieval cancels the dominant
  similarity axes; what survives in the residual is what retrieval failed to control.
- **Estimator = contrastive eigendecomposition**: top eigenvector of
  Cov(failure residuals) - Cov(success residuals). Successes share retrieval's leakage
  and noise but carry no failure-specific spike, so the shared structure cancels.
- **Trigger = detectability + stability**, not a schedule: crystallize only when the top
  eigenvalue clears a permutation-estimated noise edge (shuffle fail/success labels; the
  empirical analogue of the Marchenko-Pastur bulk edge) AND the direction is stable
  across successive checks AND it has not already been crystallized.
- **Contrast sets by projection**: the introspection prompt is seeded with failures at
  extreme opposite projections along the detected axis, plus matched successes — and is
  task-neutral (no arithmetic hints).

Verified end-to-end on a planted world in tests/test_kernel_theory.py; unit-tested with
fakes in tests/test_kernel_consolidation.py.
"""

import json

import numpy as np
from sklearn.decomposition import PCA

# Working dimension for the eigenanalysis. Residuals are projected onto the top
# R_COMPONENTS principal components of the pooled residual cloud first: the spike
# survives any projection that keeps its neighborhood, and reducing d moves the
# BBP detectability transition earlier (THEORY.md section 5).
R_COMPONENTS = 50
# Floors before the spectral machinery runs at all. Below these, even the
# permutation edge is too noisy to trust.
MIN_FAIL_RESIDUALS = 25
MIN_SUCC_RESIDUALS = 10
# Number of label-shuffles used to estimate the noise edge. The edge is the max
# top-eigenvalue over shuffles: anything a random fail/success split can produce
# is noise by construction. With max-over-N as the edge, a pure-noise lambda1
# clears it with probability ~1/(N+1) per check — 20 shuffles keeps that under
# 5% (the old value of 5 allowed ~17%, and the gate is re-checked every batch).
N_PERMUTATIONS = 20
# The direction must persist across consecutive checks (|cos| above this) before
# it is trusted — the failure stream is non-stationary, so a one-off axis is noise.
STABILITY_COS = 0.95
# A new direction too close to an already-crystallized one (|cos| above this) is
# the same axiom again; skip it. Deduplication falls out of geometry, not bookkeeping.
NOVELTY_COS = 0.8
# Contrast-set sizes for the introspection prompt.
N_CONTRAST_PER_SIDE = 3
N_MATCHED_SUCCESSES = 3


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _top_contrast_eig(fail_r, succ_r):
    """Top eigenpair of Cov(fail) - Cov(succ) in the given coordinates."""
    C = np.cov(fail_r, rowvar=False) - np.cov(succ_r, rowvar=False)
    w, V = np.linalg.eigh(C)
    return float(w[-1]), V[:, -1], w


def crystallization_prompt(side_a, side_b, successes):
    """Task-neutral contrastive prompt. side_a/side_b: (input, predicted, actual)
    tuples from opposite extremes of the detected axis; successes: (input, actual)."""
    def fmt_f(rows):
        return "\n".join(
            f"- Input: {i} | I predicted: {p} | Actual: {a}" for i, p, a in rows
        )

    fmt_s = "\n".join(f"- Input: {i} | Actual: {a}" for i, a in successes) or "- (none available)"
    return f"""I am a classification agent making systematic mistakes. My failures vary along a single hidden axis. Below are failures from the two opposite ends of that axis, plus similar cases I got right.

Failures (side A of the axis):
{fmt_f(side_a)}

Failures (side B of the axis):
{fmt_f(side_b)}

Similar cases I got RIGHT:
{fmt_s}

Task: Inside a <thought> block, work out what single property distinguishes side A from side B, and how that property determines the correct label.

Your final line must be exactly:
RULE: [one concise, testable rule mapping the property to the labels]"""


class EigenMemoryKernel:
    def __init__(
        self,
        db_conn,
        openai_client,
        model="gemma3:4b",
        min_fail_residuals=MIN_FAIL_RESIDUALS,
        min_succ_residuals=MIN_SUCC_RESIDUALS,
        n_permutations=N_PERMUTATIONS,
        stability_cos=STABILITY_COS,
        novelty_cos=NOVELTY_COS,
        rng_seed=0,
    ):
        self.conn = db_conn
        self.client = openai_client
        self.model = model
        self.min_fail_residuals = min_fail_residuals
        self.min_succ_residuals = min_succ_residuals
        self.n_permutations = n_permutations
        self.stability_cos = stability_cos
        self.novelty_cos = novelty_cos
        self.rng = np.random.default_rng(rng_seed)

        # Residual records, split by outcome. Every retrieval-bearing trial feeds
        # these (NOT gated on surprise: the covariance contrast needs unbiased
        # samples of both outcomes).
        self.fail_records = []  # {residual, embedding, input, prediction, actual}
        self.succ_records = []  # {residual, embedding, input, actual}

        # Running mean of query embeddings, used to center projections when
        # selecting axioms at inference time.
        self._embed_sum = None
        self._embed_count = 0

        # Trigger state: last observed top direction (full space) and the
        # directions already crystallized.
        self.prev_direction = None
        self.consumed_directions = []

        # Telemetry: (lambda1, permutation edge) per check, and top-3 eigenvalue
        # shares per check for the spectrum heatmap in plot_results.py.
        self.detectability_history = []
        self.spectrum_history = []

    # ------------------------------------------------------------------ ingest

    def observe(self, embedding, residual, was_correct, context_input, prediction, actual):
        """Record one trial's retrieval residual, keyed on outcome (correctness),
        not on surprise."""
        embedding = np.asarray(embedding, dtype=float)
        residual = np.asarray(residual, dtype=float)
        if self._embed_sum is None:
            self._embed_sum = np.zeros_like(embedding)
        self._embed_sum += embedding
        self._embed_count += 1

        rec = {
            "residual": residual,
            "embedding": embedding,
            "input": context_input,
            "prediction": prediction,
            "actual": actual,
        }
        (self.succ_records if was_correct else self.fail_records).append(rec)

    def embedding_mean(self):
        if not self._embed_count:
            return None
        return self._embed_sum / self._embed_count

    # ---------------------------------------------------------------- spectral

    def _reduced_residuals(self):
        """Project pooled residuals onto their top principal components.

        Returns (fail_reduced, succ_reduced, basis) with basis rows orthonormal in
        the full space, so directions can be mapped back via basis.T @ v.
        """
        F = np.array([r["residual"] for r in self.fail_records])
        S = np.array([r["residual"] for r in self.succ_records])
        pooled = np.vstack([F, S])
        r = min(R_COMPONENTS, pooled.shape[0] - 1, pooled.shape[1])
        pca = PCA(n_components=r).fit(pooled)
        basis = pca.components_
        return F @ basis.T, S @ basis.T, basis

    def _permutation_edge(self, F_red, S_red):
        """Noise edge: max top-eigenvalue over random fail/success relabelings."""
        pooled = np.vstack([F_red, S_red])
        n_fail = len(F_red)
        edge = 0.0
        for _ in range(self.n_permutations):
            idx = self.rng.permutation(len(pooled))
            lam, _, _ = _top_contrast_eig(pooled[idx[:n_fail]], pooled[idx[n_fail:]])
            edge = max(edge, lam)
        return edge

    def check_and_crystallize(self):
        """Run once per batch: eigenanalysis + the detectability/stability/novelty
        gates. Crystallizes at most one axiom per call."""
        if (
            len(self.fail_records) < self.min_fail_residuals
            or len(self.succ_records) < self.min_succ_residuals
        ):
            return

        F_red, S_red, basis = self._reduced_residuals()
        lam1, v_red, w = _top_contrast_eig(F_red, S_red)
        edge = self._permutation_edge(F_red, S_red)
        v_full = _unit(basis.T @ v_red)

        self.detectability_history.append((lam1, edge))
        top3 = np.sort(np.clip(w, 0, None))[::-1][:3]
        total = top3.sum()
        self.spectrum_history.append((top3 / total if total > 0 else top3).tolist())

        detectable = lam1 > edge
        stable = self.prev_direction is not None and (
            abs(float(v_full @ self.prev_direction)) > self.stability_cos
        )
        novel = all(
            abs(float(v_full @ c)) < self.novelty_cos for c in self.consumed_directions
        )
        self.prev_direction = v_full

        print(
            f"[EIGEN] lam1={lam1:.3f} edge={edge:.3f} "
            f"({'detectable' if detectable else 'below edge'}, "
            f"{'stable' if stable else 'unstable'}, {'novel' if novel else 'consumed'}) "
            f"fail={len(self.fail_records)} succ={len(self.succ_records)}"
        )

        if detectable and stable and novel:
            self._crystallize(v_full, strength=lam1 / edge if edge > 0 else 1.0)

    # ----------------------------------------------------------- crystallization

    def _contrast_sets(self, v_full):
        """Failures at extreme opposite projections along v_full, plus successes
        matched to them by embedding similarity."""
        proj = np.array([float(r["residual"] @ v_full) for r in self.fail_records])
        order = np.argsort(proj)
        side_a = [self.fail_records[i] for i in order[:N_CONTRAST_PER_SIDE]]
        side_b = [self.fail_records[i] for i in order[::-1][:N_CONTRAST_PER_SIDE]]

        successes = []
        if self.succ_records:
            succ_emb = np.array([_unit(r["embedding"]) for r in self.succ_records])
            used = set()
            for f in side_a + side_b:
                sims = succ_emb @ _unit(f["embedding"])
                for j in np.argsort(sims)[::-1]:
                    if j not in used:
                        used.add(int(j))
                        successes.append(self.succ_records[j])
                        break
                if len(successes) >= N_MATCHED_SUCCESSES:
                    break
        return side_a, side_b, successes

    def _crystallize(self, v_full, strength=1.0):
        """Translate the detected axis into a linguistic rule via contrastive
        introspection, and store it with its (full-space) direction."""
        side_a, side_b, successes = self._contrast_sets(v_full)
        prompt = crystallization_prompt(
            [(r["input"], r["prediction"], r["actual"]) for r in side_a],
            [(r["input"], r["prediction"], r["actual"]) for r in side_b],
            [(r["input"], r["actual"]) for r in successes],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=0,
                max_tokens=400,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[EIGEN] crystallization LLM call failed: {e}")
            return

        # Store ONLY the final RULE: line. The <thought> block is scaffolding for
        # the introspection call; storing the full reply used to inject ~1.2k
        # chars of CoT rambling into every context the axiom was selected for —
        # sabotaging the very arm under test.
        rule = raw.rpartition("RULE:")[2].strip()
        axiom = f"RULE: {rule}" if rule else raw.strip()

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_core (axiom_content, eigen_vector, strength_score)
                    VALUES (%s, %s, %s)
                    """,
                    (axiom, v_full.tolist(), float(strength)),
                )
            self.conn.commit()
        except Exception as e:
            # Roll back so the connection is not left in an aborted-transaction
            # state that would poison every later DB call on it. The axis stays
            # unconsumed, so crystallization retries at the next check.
            self.conn.rollback()
            print(f"[EIGEN] axiom store failed: {e}")
            return

        self.consumed_directions.append(v_full)
        print(f"[AXIOM+] {axiom[:80].strip()}...")

    # ------------------------------------------------------------- inference use

    def score_axioms(self, query_embedding, axiom_rows):
        """Relevance of stored axioms to a query: |projection| of the centered
        query embedding onto each axiom's axis. Sign-invariant, unlike the old
        cosine-to-eigenvector ranking (an eigenvector's sign is arbitrary).

        axiom_rows: iterable of (axiom_content, eigen_vector) where eigen_vector
        may be a pgvector text literal or a sequence. Returns rows sorted by
        descending relevance as (score, axiom_content).
        """
        mu = self.embedding_mean()
        q = np.asarray(query_embedding, dtype=float)
        centered = q - mu if mu is not None else q
        scored = []
        for content, vec in axiom_rows:
            if isinstance(vec, str):
                # pgvector's text literal ("[0.1,0.2,...]") happens to be valid
                # JSON — a format coincidence, named here so it reads as a choice.
                vec = json.loads(vec)
            v = np.asarray(vec, dtype=float)
            scored.append((abs(float(centered @ v)), content))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored
