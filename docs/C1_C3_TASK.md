# The C1∧C3 Task: Where Eigen-Memory Could Actually Win

> A concrete task design — generator, protocol, and falsifiable hypotheses — for the one regime
> the two prior experiments never reached. Number-game failed **C1** (rule invisible to
> embeddings). TREC satisfied C1 but failed **C3** (a single retrieved exemplar already gives the
> answer away, so rule-compression adds nothing). This task is built to satisfy **C1 and C3 at
> once** — the only setting where eigen-axioms can beat plain RAG.
>
> The design below was red-teamed; the central correction (why a naive *conjunction* fails, and
> why an *XOR-style label-flip* is needed instead) is documented in
> ["The trap we avoided"](#the-trap-we-avoided-why-not-a-conjunction).
>
> **Theory correction (second red-team pass):** the mechanism story this design originally
> leaned on — "PCA over failure vectors surfaces B" — is provably false; the corrected
> mechanism (contrastive PCA over **retrieval residuals**, with a detectability-gated trigger)
> is derived and numerically verified in [THEORY.md](THEORY.md). The design below has been
> updated to match: the mechanism section, Guardrail 1's numeric gates, the Oracle ceiling
> arm, and the `Eigen > max(RAG, Baseline)` win condition all follow from it.

See [USE_CASES.md](USE_CASES.md) for the C1–C4 definitions and [VALID_EXPERIMENT.md](VALID_EXPERIMENT.md)
for the protocol scaffold this extends.

## The one thing both prior tasks missed

| | Number-game | TREC | **This task** |
|---|---|---|---|
| C1 — rule visible in embeddings | ✗ | ✓ | **✓** |
| C2 — needs generalization (held-out) | ✗ | ✓ (if split) | **✓** |
| **C3 — one exemplar is NOT enough** | ✗ | **✗** | **✓ (by construction)** |
| C4 — few stable rules | ✓ | ✓ | **✓** |

The crux is **C3**. On TREC, retrieving the single nearest question (`"Where is the Eiffel
Tower?"` → `LOC`) hands you the label — the nearest exemplar *is* the answer. So plain RAG wins and
there is nothing for an abstract rule to add. To make rule-compression pay, the task must be one
where **copying the nearest labeled example is unreliable** — its accuracy pinned near chance —
while the governing rule stays a semantic property an embedding model can represent. The
operational statistic is **m**, the nearest-neighbor label-match rate ([THEORY.md](THEORY.md) §7):
TREC has high m (one exemplar settles a question); this task is built so m sits near chance *and*
every neighbor mismatch produces a confidently wrong label rather than mere blur.

## The trap we avoided: why not a conjunction?

The obvious first design is a **conjunction**: label = `APPROVE` if (about money **AND** urgent),
`REVIEW` if (money, calm), `IGNORE` if (not money). It looks like it needs a rule — but it
**re-creates TREC** and hands the win back to RAG. The reason is a genuine tension between C1 and
C3:

- For PCA to crystallize an "urgency" axiom, urgency (attribute B) must be **visible in embedding
  space** — that's C1.
- But if B is visible, then top-k retrieval *already* separates the neighbors by urgency: a calm
  money-message retrieves calm money-neighbors → `REVIEW`, an urgent one retrieves urgent
  neighbors → `APPROVE`. **The conjunction gets solved by retrieval precisely because both
  attributes are embedding-visible.** k=5 majority vote wins with no rule needed.

So for a conjunction, *the more C1 holds, the more C3 fails.* The nearest exemplar is at worst
**ambiguous** (a mixed bag of APPROVE/REVIEW money-messages), and majority vote resolves ambiguity.
A naive conjunction is the weakest possible choice for isolating C3.

## The fix: a polarity-flip (XOR-style) rule

Make every neighbor mismatch **cost the full label**, not just add noise. One honesty note up
front: natural retrieval can never be made *anti*-correlated — if B is embedding-recoverable at
all, it weakly *attracts* same-B neighbors, so the polarity-match rate m sits at or above 0.5
(the planted-attribute simulation in [THEORY.md](THEORY.md) §7 measured m = 0.87 for a
thoroughly sub-dominant B against a dense buffer). What the XOR structure buys is different and
sufficient: **copy accuracy equals m exactly** — every polarity mismatch flips the label — and
majority vote cannot exceed m, because the k neighbors' polarities are draws from the same
coin. A conjunction leaves majority vote room to resolve ambiguity; the flip rule leaves
copying nothing to average over. True anti-correlation exists only where we force it: the
`Adversarial_RAG` arm (Guardrail 3).

**Domain:** 1–2 sentence workplace messages (a request-routing assistant).

**Two hidden attributes** (never named to the agent):
- **A — topic** (the *dominant* similarity axis: which team/subject the message is about). This is
  what embeddings cluster on most strongly, so it drives nearest-neighbor retrieval.
- **B — a binary "polarity" attribute that is semantic and embedding-recoverable, but *not* the
  dominant variance axis** (e.g. whether the message is a *request* vs. a *report* — same topic,
  different speech act).

**Hidden label rule — the label flips on B within each topic:**

> label = `TOPIC_LABEL(A)` **XOR** `B`
>
> i.e. for a given topic, requests route one way and reports route the *opposite* way — and the
> mapping is such that **two same-topic messages get opposite labels.**

Concretely, with topics {billing, infra} and B ∈ {request, report}, three route-classes
{ESCALATE, FILE, DEFER}:

| Topic (A) | request (B=1) | report (B=0) |
|-----------|---------------|--------------|
| billing | ESCALATE | DEFER |
| infra   | FILE | ESCALATE |

The nearest neighbor of a billing-**request** (→ESCALATE) is, by topic dominance, a
billing-message of *either* polarity — and with probability 1−m it is a billing-**report**
(→DEFER): **same topic, opposite label.** With m pinned near chance (Guardrail 1), copying is a
coin flip that always pays out a confidently wrong label. Yet a single stable axiom ("within
billing, requests ESCALATE and reports DEFER; within infra it's reversed") generalizes
perfectly.

Why each condition holds:
- **C1 ✓** — both topic (A) and speech-act/polarity (B) are semantic and embedding-recoverable,
  so the spectral stage can find the B *direction* (mechanism below).
- **C3 ✓ (by construction + measurement)** — copy accuracy equals m, the generator is tuned
  until m ≈ chance (Guardrail 1), and majority vote cannot exceed m under the flip rule.
- **C2 ✓** — held-out templates/vocabulary at test time (below).
- **C4 ✓** — the rule is a tiny, fixed lookup table per run.

**The mechanism, stated correctly (see [THEORY.md](THEORY.md) for derivation and numerical
verification):** the original claim here — that failure-conditioning inflates variance along B
so PCA over failure *embeddings* surfaces it — is false (failures mildly *suppress* B
expression, and B stays buried behind the topic axes at the same rank). What actually isolates
B is the **retrieval residual** δ = query − retrieved-neighbor: retrieval cancels the dominant
topic axes (that is what made it the neighbor), and conditioning on failure makes the residual's
B-component deterministic (±2β), giving a rank-one spike on v_B. The robust estimator is
**contrastive PCA of failure residuals against success residuals** (both available from the
agent's own logs), triggered only past the BBP detectability threshold (~100 failure residuals,
never per-batch). PCA never needs to solve the XOR — it locates the B *axis*; the LLM induces
the flip *rule* from contrast examples selected along that axis. That division of labor is the
entire theoretical reason eigen-memory should beat RAG here, and it is the thing the experiment
tests.

## Three non-negotiable guardrails (run before generating a single message)

The red-team flagged that this task dies on arrival if any of three things go wrong. Each is a
cheap pre-check, not a full run.

### Guardrail 1 — Measure the two regime statistics: probe-AUC(B) and m (~30 min)

Two numbers decide whether the task is in the eigen window ([THEORY.md](THEORY.md) §7), and both
are measurable with just the generator and the embedding model — no LLM, no agent:

- **probe-AUC(B) ≥ 0.8** — train a linear probe on sentence embeddings to predict B alone. If
  the probe can't recover B, **C1 is false** — no spectral method has a chance either. Stop,
  redesign.
- **m ∈ [0.45, 0.60]** — store the seed's actual train set, retrieve each **held-out** item's
  nearest neighbor, and measure the polarity-match rate m plus the 3-class label-copy accuracy
  (`copy_acc`, the honest C5 ceiling). If m is high, RAG wins by lookup and there is nothing
  for a rule to add (the conjunction trap, TREC's regime).

  > **Post-mortem (2026-07-16):** the original guardrail measured m with *train-split* queries
  > against a 150-item uniform store — the wrong queries at the wrong buffer size. Measured
  > under protocol conditions (held-out queries, disjoint vocabulary, ≤100-item train store),
  > cross-split neighbors match on **polarity itself**: m = 0.73–0.89 on every seed, with topic
  > match collapsing to 0.38–0.47. **C3 actually fails for this task on every seed** — the
  > within-split measurement was an artifact. The general lesson (C1 ⇒ ¬C3 on static tasks:
  > whatever attribute generalizes across the split dominates cross-split similarity, and
  > that's the rule attribute) is formalized in [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md).
  > `guardrail_flip.py` now measures under protocol conditions and writes
  > `results/flip/guardrail.flip.<seed>.json` for the aggregator.

Do **not** substitute a variance-share check for the m measurement: the simulation showed a B
at 1/10th the topic scale — thoroughly sub-dominant — still yielding **m = 0.87** against a
dense buffer. Sub-dominance does not buy retrieval-invisibility; m also *rises* with buffer
density, so it must be measured at the protocol's buffer size.

If either gate fails, the task is dead on arrival. This 30-minute check gates everything else.

**What tuning m actually took (measured with `guardrail_flip.py`, embeddinggemma, seed 42).**
Four generator designs failed this gate before one passed — the record is the design lesson:

| Generator design | probe(B) | m | verdict |
|---|---:|---:|---|
| Whole-sentence polarity frames ("Please handle X" / "FYI, X is done") | 1.000 | 0.800 | FAIL |
| + compositional topic phrases (object × context, ~48/topic) | 1.000 | 0.860 | FAIL |
| Minimal-pair frames (shared skeleton, few words differ) | 0.997 | 0.707 | FAIL |
| Tighter pairs (1–2 word diffs, short frames) | 0.997 | 0.740 | FAIL |
| **Marker-slot design**: polarity = 2–3 shared-head-word tokens ("not yet handled" / "fully handled") inside a long compositional neutral shell (opener × phrase × 2 tails) | **0.947** | **0.567** | **PASS** |

Lessons: (1) any whole-sentence realization of B lets the embedder pre-split retrieval on it —
speech-act is a *strong* sentence-embedding feature, and softening the wording does not help;
(2) the working trick is the probe-vs-variance asymmetry: shrink B to a few tokens sharing
head words across polarity (embedders underweight negation/aspect), so B's share of pairwise
distance collapses while a *supervised* probe still recovers it; (3) m varies by seed
(0.567 on seed 42, 0.633 on seed 7 with the passing design) — gate **each** rule-instance.

### Guardrail 2 — Paraphrased, disjoint-vocabulary held-out test (kills the substrate/template confound)

Templating from fragment banks risks the embedding clustering by **surface vocabulary**, not by the
latent attribute — the number-game's substrate bug in a new costume. PCA-1 would then be "uses
request-bank words," a lexical axis masquerading as semantic, yielding a trivial keyword-spotting
"win" that won't survive paraphrase. Mitigations:
- **Crossed, not nested, templates** — every surface frame appears across *all* label classes, so
  template identity can't shortcut the label.
- **Multiple unrelated lexical realizations per attribute** — express B (e.g. "request") via
  several disjoint framings so B ≠ any single keyword.
- **Disjoint train/test vocabulary banks** — hold out whole word-banks, not just instances.
- **LLM-paraphrased test set** — express the same A/B attributes in surface form unlike the train
  bank. **If eigen's win evaporates on paraphrase, it was lexical, not semantic — and you caught
  yourself.**
- **Adversarial lexical confounds** — inject messages with B-keywords but the opposite semantic B
  ("no rush, but…"). If PCA tracks the keyword and mislabels these, the axis is lexical, not C1.

### Guardrail 3 — Pre-register the ceiling check and per-cell analysis (kills self-fooling)

The silent killer: a capable LLM induces the rule **zero-shot** from the examples, making *both*
RAG and eigen irrelevant and collapsing all arms into noise.
- **Oracle ceiling gate:** run an `Oracle_Rule` arm — no memory, but the true flip-table pasted
  into context. It defines the headroom ceiling (it also catches a model too weak to *apply*
  the rule even when told it). Pre-register that `Baseline` (no memory, no rule) must sit
  **well below Oracle** (≥ 20 points), or **abort** — if the model solves the task unaided, the
  comparison is moot. Do *not* gate Baseline against RAG: this task is built to make RAG
  unreliable, so RAG may legitimately land at or below Baseline, and a Baseline-vs-RAG gate
  would abort a healthy run.
- **Executor gate (C5) — discovered by the first live run, replicated on all four corrected
  seeds:** require **Oracle > copy_acc**, the guardrail-1 3-class label-copy ceiling (NOT the
  2-way polarity m — different scales). If the model executes the *true* rule worse than blind
  neighbor-copying scores, then rule-based memory cannot beat RAG **regardless of axiom
  quality**, and the H1 comparison is decided before training starts. Measured for gemma3:4b,
  corrected 4-seed run: Oracle 0.411 ± 0.103 vs ceilings 0.578–0.600 — **gate fails on every
  seed** (paired Oracle − ceiling = −0.178 ± 0.093). This gate costs one test-phase pass (~45
  calls) and one number you already have. The fix is a qualified executor (see the RFμ
  microbenchmark in [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md)), not more seeds.
- **Per-cell error breakdown:** the eigen win must concentrate on the **anti-correlated cells** (the
  same-topic-opposite-label confusions). A *uniform* gain across cells is a generic prompt-quality
  effect, not a C3 win.
- **Adversarial-RAG control:** also run a RAG arm that deliberately retrieves same-topic-opposite-label
  neighbors. Eigen beating *adversarial* RAG (not just random RAG) isolates the C3 mechanism.
- **Don't read a 1–2 point aggregate gain as success:** bootstrap CIs over items; require the
  eigen−RAG gap to exceed the noise these tasks always carry.

## Protocol (extends VALID_EXPERIMENT.md)

- **Train phase:** N_train ≈ 150 trials, feedback on, memory built.
- **Test phase:** N_test ≈ 50 **held-out** items (held-out templates *and* vocabulary, paraphrased),
  feedback **off**, memory **frozen**. Held-out accuracy is the headline metric.
- **Arms:**
  - `Baseline` — no memory.
  - `Oracle_Rule` — no memory, true flip-table in context: the headroom ceiling (Guardrail 3).
  - `Control_RAG` — top-k episodic retrieval.
  - `Adversarial_RAG` — top-k forced to same-topic-opposite-label neighbors (Guardrail 3).
  - `Treatment_Eigen` — RAG + crystallized axioms (the corrected kernel of
    [THEORY.md](THEORY.md) §8: contrastive residual PCA, detectability-gated trigger,
    projection-selected contrast sets).
  - `RAG_large` — top-k with k sized to match Eigen's *token budget* (controls "Eigen just adds
    more context"; see H3).
- **Seeds:** ≥ 10, each a fresh rule-instance (re-permute the flip table) + data split.

## Hypotheses (pre-registered, falsifiable)

- **H1 (primary):** held-out `Treatment_Eigen` > **`max(Control_RAG, Baseline)`**, paired across
  seeds, `p < 0.05`. Beating RAG alone is not enough: a task built to make retrieval unreliable
  can drag RAG *below* Baseline, and outperforming a sabotaged arm proves only that bad
  neighbors are bad. The memory has to beat having no memory at all.
- **H2 (the C3 signature):** Eigen's gain is **concentrated on the polarity-mismatch cells**
  (items whose nearest neighbor has opposite B — the confusions only the flip-rule resolves),
  and Eigen beats **`Adversarial_RAG`**. A uniform gain across cells is a generic prompt-quality
  effect, not a C3 win.
- **H3 (mechanism, not context size):** `Treatment_Eigen` > `RAG_large` at equal token budget. If
  this fails, the "win" is just more context.

## Decision rule (honest readout)

- **H1 ✓, H2 ✓, H3 ✓** → eigen-memory wins *and* it's the rule-compression mechanism, isolated to
  the flip boundary only an abstract rule can resolve. The strongest possible positive.
- **H1 ✓, H3 ✗** → just more context; report as no real eigen win.
- **H1 ✗ with `Treatment_Eigen` > `Control_RAG` but ≤ `Baseline`** → memory of both kinds hurts
  here; the axioms only diluted poisoned retrieval. Report as a negative for the architecture,
  not a win over RAG.
- **H1 ✗** → even on a C1∧C3 task with held-out generalization and copy accuracy pinned at
  chance, compressing failures into rules still doesn't beat retrieval — *with the corrected
  kernel, past its detectability threshold* (verify the mechanism-health telemetry before
  reading the null: alignment-stability at crystallization time, per §5 of THEORY.md). This is
  the **decisive** negative the prior two tasks couldn't deliver — because here RAG has **no
  structural excuse** to win and the kernel has none either.

## How it slots into the current code

Two kinds of change, and it is no longer honest to say "only the dataset changes":

1. **Task wiring (same as TREC):** add a `flip` task to `src/dataset.py` emitting
   `{input: message, label: ...}` from the crossed-template, polarity-flip generator above,
   register it in `LABELS`/`get_labels`, run with `TASK=flip`. Plus: a held-out frozen-memory
   test phase in `simulate.py`, and the `Oracle_Rule` / `Adversarial_RAG` arms.
2. **Kernel rework (done — [THEORY.md](THEORY.md) §8):** the kernel now implements the
   corrected mechanism — failure/success *residual* buffers keyed on outcome, contrastive PCA
   with the detectability-gated (permutation-edge + stability + novelty) trigger,
   projection-selected contrast sets, a task-neutral crystallization prompt, |projection|-based
   axiom injection, and a memory-conditional surprise probe. Unit-tested against a planted
   axis in `tests/test_kernel_consolidation.py`.

**Build order (fail fast):**
1. **Guardrail 0 is already discharged** — `tests/test_kernel_theory.py` verifies the mechanism
   end-to-end on a planted world (`uv run pytest tests/test_kernel_theory.py -s`). If those
   tests didn't pass, no generator tuning could save the design.
2. Guardrail 1's probe-AUC + m measurement is a standalone ~30-minute script with *no*
   LLM-in-the-loop: generator + embedder only. Write and run it before any agent code. If m
   can't be tuned into [0.45, 0.60] at buffer size ~150, the task is dead and you've spent half
   an hour, not a week.
3. Only then the kernel rework and arms.

## Live runs

**First run (2026-07-14, seed 42) — later found compromised.** Held-out: Baseline 0.222,
Oracle 0.467, RAG 0.533, Eigen 0.356; the kernel appeared to crystallize one correct-axis
axiom. A review pass then found two silent bugs (full-CoT stored as axioms and injected into
the Treatment arm; multi-token labels flattening the surprise probe to a constant for 2 of 3
classes) plus the guardrail measurement error documented above. Archived in
`results_prefix_bug/` as the record.

**Corrected run (2026-07-16, 4 seeds {42, 2, 18, 23}, temperature 0, health telemetry).**
Held-out means ± std: Baseline 0.289 ± 0.048, Oracle 0.411 ± 0.103, **RAG 0.600 ± 0.101**,
Eigen 0.617 ± 0.106. **H1 not supported** — 0–1 axioms per seed; on 3/4 seeds zero axioms and
Eigen ≡ RAG prediction-for-prediction; the one fired axiom (seed 18) named the polarity axis
with an inverted mapping. C5 fails 4/4 (see the executor gate above). Aggregate:
`results/flip/comparison_results.flip.aggregate.json`; per-seed `results/flip/comparison_results.flip.<seed>.json` +
`results/flip/guardrail.flip.<seed>.json`. Write-up: [BLOG_POST.md](BLOG_POST.md).

## What this task turned out to prove

The design goal — remove every structural excuse — succeeded, just not in the intended
direction. The substrate can see the rule (C1 held, probe 0.95–0.97); the test demands
generalization (C2 held, disjoint banks); but the corrected measurement shows **C3 cannot be
made to hold at the same time as C1 on a static task** (the post-mortem above), and C5 shows
the executor couldn't have cashed in a perfect axiom anyway. A loss with all excuses removed
would have falsified the hypothesis; instead the run falsified the *task family* — static
single-session rule tasks — and produced the two gates (corrected-C3, C5) plus the gate-ROC
calibration that the successor design inherits. That successor (Rule-Shift: break copying with
time instead of geometry) is pre-registered in [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md).
