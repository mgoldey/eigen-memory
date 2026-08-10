# Residual-Error Spectral Consolidation: a Corrected Underpinning

> The original mechanism story — "PCA over failure embeddings surfaces the hidden attribute" —
> is provably wrong, and a review of this repo showed it. This document replaces it with a
> corrected theory whose every claim is **executable**: `tests/test_kernel_theory.py` builds a
> planted-attribute world and verifies each section's numbers
> (`uv run pytest tests/test_kernel_theory.py -s`). The corrected mechanism also
> retro-explains, quantitatively, why the axioms observed in the real runs were garbage.

Throughout, the setting is the C1∧C3 task family ([C1_C3_TASK.md](C1_C3_TASK.md)): embeddings
`x = t_topic + b·v_B + ε` with a dominant topic structure `t`, a sub-dominant binary attribute
`b ∈ {−β, +β}` along a direction `v_B`, isotropic noise `ε`, and a hidden label that **flips on
B within each topic** (XOR structure). The agent retrieves nearest neighbors from an episodic
buffer and gets correct/incorrect feedback.

## 1. The old story is false — and backwards

The old claim: *"B has low global variance but high residual variance among failures, because
errors concentrate where B-copying went wrong; PCA over the failure vectors therefore surfaces
B."* Two things are wrong with this:

1. **Variance along a fixed direction is a property of spread, not of label-relevance.**
   Failures under neighbor-copying occur for both polarities symmetrically, so the failure set
   spans `v_B` the same way the corpus does. Conditioning on failure cannot amplify variance
   along `v_B`.
2. **In fact it *suppresses* it.** An item that strongly expresses B sits closer to same-B
   neighbors, so it rarely fails; failures select for items whose B expression is *weak*.
   Measured: variance along `v_B` among failures is **0.78×** the global value, and `v_B` sits
   at the **same rank (8th)** behind the topic axes in failure-PCA as in global PCA — top-3
   failure components have |cos| ≤ 0.06 with `v_B`.

A separate, structural objection: the XOR label is **not linearly decodable** from the
embedding at all (linear label-probe: **0.52** accuracy, vs **0.95** for a probe of B itself).
No linear-spectral stage can classify this task — so any theory in which PCA "finds the rule"
was dead on arrival. What PCA *can* find is a **direction**; see §4.

Finally, the old code crystallized on ~10 vectors per batch in 768 dimensions — below any
detectability threshold (§5). Three independent reasons the observed axioms were noise.

## 2. The corrected object: retrieval residuals

Don't analyze failure *embeddings*. Analyze failure **residuals** — the difference between the
query and what memory retrieved for it:

> **δ = x − x_nn = (b − b′)·v_B + (t − t′) + (ε − ε′)**

Retrieval selects the neighbor by minimizing distance, which is dominated by the
high-variance topic axes — so `t − t′ ≈ 0`. **Retrieval is a matched filter for the dominant
axes**: whatever survives in the residual is exactly what retrieval failed to control.
Conditioning on *failure* then makes the B-component deterministic: with the within-topic flip
rule, a wrong copied label means `b ≠ b′`, so

> **Cov(δ | fail) ≈ 4β²·v_B v_Bᵀ + (2σ²/d)·I**

— a rank-one spike on `v_B` sitting on an isotropic noise floor, *regardless of how deeply B
is buried in the global spectrum*. Measured: residual-PCA puts `v_B` at **rank 1 with
|cos| = 0.91**, up from rank 8 in the raw spectrum.

This is not a new trick — it is what Representation Engineering actually does. RepE's reading
method (LAT; Zou et al. 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) runs PCA
over **difference vectors of contrastive stimulus pairs**, never over raw activations,
precisely because raw PCA returns dominant nuisance variance. The transport here: the
contrastive pair is **(query, retrieved exemplar)** — generated *for free* by the agent's own
memory, with the pairing chosen by the very similarity structure the memory uses. The original
kernel skipped the differencing step, which is why it inherited none of RepE's power.

## 3. The robust form: contrastive residual PCA

Plain residual-PCA degrades when retrieval leaks: with a sparse buffer or heavy noise the
nearest neighbor is far (or cross-topic), so `t − t′` no longer cancels and topic directions
re-enter the residual spectrum. Measured in the hard regime (buffer 300, high noise): plain
residual-PCA alignment collapses to **0.39**.

The fix uses a background set the agent also gets for free — **success residuals**. Successes
share the same retrieval leakage and noise structure, but carry no `±2β` spike (the neighbor
matched on B). So take the top eigenvector of the **covariance difference**:

> **C = Cov(δ | fail) − Cov(δ | success)**

Shared structure cancels; only the failure-specific spike survives. Measured: the hard regime
recovers from 0.39 to **0.86**; success residuals alone show no spike (|cos| ≤ 0.09). This is
contrastive PCA (Abid, Zhang, Bagaria & Zou, *Nature Communications* 9:2134, 2018;
[arXiv:1709.06716](https://arxiv.org/abs/1709.06716)) with both datasets drawn from the
agent's own interaction log — the only supervision needed is the correct/incorrect feedback
the agent already receives.

*Novelty note:* a literature check found no established method applying PCA to
(query − retrieved-neighbor) residuals for error analysis or rule induction. Adjacent work
uses PCA for retrieval-vector *compression* (PCA-RAG) and residual quantization for
*indexing* — same words, different purpose. The defensible ancestors of the mechanism are
LAT and cPCA.

## 4. Division of labor: the spectrum finds *where to look*, the LLM finds *the rule*

The spectral stage cannot solve the task (§1: the label is linearly undecodable). It doesn't
have to. Its two outputs are:

1. **A direction** `v̂` — the axis along which failures differ from their retrieved neighbors.
2. **A maximally contrastive example set** — failure pairs with extreme opposite projections
   onto `v̂`, plus matched successes.

The **LLM** then induces the *function* (the flip rule) from examples organized along that
axis. Rule induction from a handful of well-chosen examples is what LLMs are good at; finding
which handful to show them, inside thousands of stored episodes, is what the spectral stage is
for. Under this division, XOR is easy: the spectrum locates the request/report axis, and the
LLM reads "same topic, opposite label" off four examples straddling it.

This also fixes axiom *retrieval*, which was broken in the old code: ranking axioms by
cosine between a query embedding and an eigenvector is meaningless (eigenvector sign is
arbitrary; components are directions, not locations). The corrected selection is by
**|projection| of the query onto v̂** — "this input sits far out on an axis my failures vary
along" — which is sign-invariant and actually means something.

## 5. When to crystallize: detectability, not schedule

Estimating the top eigenvector of a spiked covariance from `n` samples in `d` dimensions has a
**phase transition** (Baik, Ben Arous & Péché, *Ann. Probab.* 33(5), 2005 — stated there for
complex sample covariances; Baik & Silverstein 2006 for the real-valued analogue): below a
critical sample count the leading sample eigenvector is asymptotically *uncorrelated* with the
true spike, no matter how carefully you compute it. Pure-noise eigenvalues fill the
Marchenko–Pastur bulk with upper edge `σ_n²(1 + √(d/n))²`; only an eigenvalue clearing that
edge carries signal. In the implementation the operational gate is a **permutation-estimated
edge** (shuffle the fail/success labels, take the max top-eigenvalue over shuffles) — the BBP
result is the motivation, not the formula on the hot path.

Measured on the planted task (averaged over resamples):

| n failure residuals | alignment with v_B |
|---:|---:|
| 10 | **0.18** — noise |
| 25 | 0.36 |
| 100 | 0.72 |
| 400 | **0.89** |

The old kernel crystallized **every batch, on ~10 vectors, in d = 768**. The table says what
that produces: a random direction. And a random direction has a downstream failure mode
specific to LLM pipelines — the contrast sets handed to the introspection prompt are
effectively arbitrary episodes, and **an LLM will confabulate a confident rule from arbitrary
examples**. The garbage and "just guess randomly" axioms observed in the real runs are the
predicted signature of crystallizing below the detectability threshold, not a mystery about
prompt quality.

**Measured operating characteristic of the full gate (gate_roc.py, 2026-07-16).** Driving the
actual kernel over planted rank-1 contrasts at controlled multiples of its own permutation
edge (n_fail ∈ {50, 100, 200}, 20 replicates/cell): full-gate false-positive rate **0.00** at
pure noise (the edge alone leaks ~5% ≈ 1/21, exactly the max-over-20-permutations expectation;
the stability check removes the rest); detectability turns on at 1× the edge (100%); but the
**full** gate fires only from ~8×, because the cross-check direction cosine (0.84 / 0.89 /
0.94 / 0.96 at snr 1/2/4/8) crosses the 0.95 stability threshold between 4× and 8×. So the
stability gate — not the eigenvalue edge — is the binding constraint at realistic n: the
compound gate trades an ~8× detection margin for measured-zero false positives. Implication
for future designs: to detect weaker real axes, scale the stability threshold to the expected
estimator wobble at the current sample count (a sample-size-aware cosine), rather than
lowering the edge. Raw sweep: `results/calibration/gate_roc.json`.

The corrected trigger — which finally earns the "eigen" in eigen-memory:

> Crystallize a direction only when **(a)** its eigenvalue in the contrast matrix `C` exceeds
> the estimated noise edge, **and (b)** the direction is stable across successive updates
> (`cos(v̂_t, v̂_{t−1}) > 0.95`).

Practical corollaries: accumulate on the order of **≥100 failure residuals** before the first
crystallization; reduce the working dimension first (project residuals onto the top ~50
global PCs — this moves the `√(d/n)` transition earlier at no cost, since the spike survives
any projection that keeps `v_B`'s neighborhood); and expect to re-trigger after axiom
injection shifts the failure distribution.

## 6. Surprise must be conditioned on memory

The current probe measures the *bare model's* prediction error — the retrieved context is
stripped from the surprise call. That gates storage on item difficulty, not on what the
*agent* still gets wrong: items the memory already handles keep being stored as "failures,"
and the surprise signal can never decline as the agent learns.

The corrected definition follows the Prioritized-Experience-Replay analogy properly (Schaul
et al. 2015 prioritize by the *current* network's TD error, recomputed as the network learns):
**surprise = prediction error of the full agent, memory in context.** This buys the
self-limiting property the architecture was missing: once a crystallized axiom fixes a failure
mode, those items stop being surprising, the failure-residual buffer drains, and the same
direction is not re-crystallized — deduplication falls out of the theory instead of needing
bookkeeping. (It also makes the failure stream non-stationary, which is the second reason the
stability check in §5's trigger is load-bearing.)

## 7. The regime map: two measurable numbers decide everything

Both negative results in this repo, and the design of the next task, reduce to two statistics
that can be measured **before any agent run**, using only the embedding model and the
generator/dataset:

- **probe-AUC(B)** — train a linear probe for the latent attribute on embeddings. This is C1:
  *is the attribute in the representation at all?*
- **m** — the nearest-neighbor label-informativeness: retrieve each item's nearest stored
  neighbor and measure how often copying its label is right, **at the buffer size the agent
  will actually have**. This is C3, operationalized: *how far does exemplar-copying already
  get you?*

| Regime | Diagnosis | Seen in |
|---|---|---|
| probe-AUC ≈ chance | Substrate blind: nothing downstream can work — not RAG, not eigen | number-game |
| m high | Exemplar-copying suffices; axioms are redundant context | TREC |
| probe-AUC high **and** m ≈ chance | **The eigen window** — rules can carry what lookup can't | the C1∧C3 task, by construction |

Two measured warnings for anyone designing for the window:

1. **Sub-dominance does not buy low m.** In the simulation, B at 1/10th the topic scale —
   thoroughly sub-dominant — still produced **m = 0.87** against a dense buffer: even weak
   embedding leakage of B lets a dense neighbor pool match on it. `m` must be *measured* with
   the real generator, embedder, and buffer size, never inferred from a variance share.
2. **m is a coverage phenomenon: it rises with buffer density** (measured: 0.66 → 0.69 → 0.71
   for buffers of 300 → 1000 → 3000; the effect is stronger at lower noise). RAG's copy
   accuracy is a function of how densely memory covers the input manifold. The corollary is
   the honest general statement of where this architecture wins: **rule-compression is a hedge
   against sparse coverage** — early in an agent's life, in long-tail domains, wherever
   episodes are expensive — and lookup overtakes it as coverage densifies. C3 and the
   "episodes costly/redundant" condition of [USE_CASES.md](USE_CASES.md) are the same fact.

## 8. What this changes in code and protocol (items 1–7 implemented)

The agent loop keeps its shape; the kernel and probe change. Items 1–7 below are now
implemented in `src/eigen_memory_agent/` (unit tests: `tests/test_kernel_consolidation.py`);
item 8 is the C1∧C3 protocol, which has since been built and run — it became the label-flip
experiment ([C1_C3_TASK.md](C1_C3_TASK.md)), and running it honestly showed the C1∧C3 regime
cannot exist on a static task:

1. **Log the pairing**: store each episode's retrieved-neighbor ids and the correct/incorrect
   outcome (schema: two columns).
2. **Kernel input = residuals**: maintain failure-residual and success-residual buffers
   (`δ = query_embedding − retrieved_embedding`), not raw embeddings.
3. **Dimension-reduce** residuals onto the top ~50 global PCs before eigenanalysis.
4. **Trigger** = eigenvalue of `Cov_fail − Cov_succ` above the estimated noise edge **and**
   direction stability across updates — replacing crystallize-every-batch.
5. **Contrast sets by projection**: seed the introspection prompt with failures at extreme
   opposite projections on `v̂` plus matched successes — replacing cosine-to-eigenvector
   retrieval. Same for axiom injection at inference: select by |projection|.
6. **Task-neutral crystallization prompt** — the current prompt hardcodes "check for
   arithmetic properties," which contaminated the TREC arm.
7. **Memory-conditional surprise**: include the retrieved context in the surprise probe (§6).
8. **Protocol** (details in [C1_C3_TASK.md](C1_C3_TASK.md)): the win condition is
   `Eigen > max(RAG, Baseline)` — anti-correlated or uninformative retrieval can drag RAG
   *below* Baseline, and beating a sabotaged arm proves nothing — with an Oracle arm (rule
   given in context) defining the headroom ceiling.

## References

- Zou et al. 2023, *Representation Engineering: A Top-Down Approach to AI Transparency*,
  [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) — LAT: PCA over contrastive difference
  vectors.
- Abid, Zhang, Bagaria & Zou 2018, *Exploring patterns enriched in a dataset with contrastive
  principal component analysis*, Nature Communications 9:2134
  ([arXiv:1709.06716](https://arxiv.org/abs/1709.06716)) — cPCA.
- Baik, Ben Arous & Péché 2005, *Phase transition of the largest eigenvalue for nonnull
  complex sample covariance matrices*, Annals of Probability 33(5):1643–1697 — the BBP
  detectability transition (complex case; real-valued analogue: Baik & Silverstein 2006,
  *J. Multivariate Anal.* 97(6); bulk edge: Marchenko & Pastur 1967).
- Schaul et al. 2015, *Prioritized Experience Replay*,
  [arXiv:1511.05952](https://arxiv.org/abs/1511.05952) — priority from the *current* policy's
  prediction error.

Verification: `uv run pytest tests/test_kernel_theory.py -s` (7 tests; every number quoted
above is printed and asserted there).
