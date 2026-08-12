# Eigen-Memory: compressing an agent's surprises into rules instead of weights

Titans ([Behrouz et al. 2024](https://arxiv.org/abs/2501.00663)) keeps what surprises it and
compresses it into neural weights — but Titans is an architecture you pretrain from scratch.
This project rebuilds those economics as a pure inference-time wrapper around a frozen model:
surprising failures get compressed into short, legible English rules instead of weights.

**How it works, in three sentences.** The agent answers a stream of classification items and
scores its own *surprise* on each one — how wrong and how unsure it was, read off the model's
logits. Surprising failures get stored; when enough of them line up along a single direction
in embedding space, a **trigger** fires and the agent writes itself one short English rule
(an **axiom**) summarizing that whole cluster of failures, which is then injected into later
prompts. The comparison throughout is against plain **RAG** — retrieving the most similar past
examples — because that is what you would actually build instead.

Four controlled experiments ask where the rules beat the retrieval.

**They never do — and the instrumented reasons why are the deliverable.**

| # | Experiment | Verdict | Why |
|---|------------|---------|-----|
| 1 | **Number-game** — classify integers by a hidden arithmetic rule | tie | The embeddings can't see the rule: text embeddings don't encode primality, so *no* memory can work. |
| 2 | **TREC** — question-type classification | RAG wins (0.80 vs 0.75) | The rig detects real memory benefits — but one retrieved example settles each question, so rules have nothing to add. |
| 3 | **Label-flip** — purpose-built, 4 seeds, held-out | tie (0.617 vs 0.600) | The trigger stayed shut on 3 of 4 seeds, where predictions were *item-identical* to RAG's; the whole gap is one seed's single wrong-mapping axiom. Two structural reasons, both below. |
| 4 | **Rule-Shift** — the rule flips mid-run, 12B model, 5 seeds | **miss** — +0.078 against a +0.10 bar set in advance | The trigger fired on only 1 of 5 seeds. But on that seed, the rule the agent wrote beat copying **0.922 vs 0.522**. |

The first two experiments fail for reasons that have nothing to do with rule-compression: one
picked a rule the embeddings can't represent, the other a task a single example already solves.
Experiments **3 and 4 were built to remove those excuses, and they are the real results**:

- **Flip (3)** produced two replicated discoveries, both about *why* a rule couldn't win:
  - **The model applies the rule worse than it copies.** Paste the *true* rule into the 4B
    model's context and it scores 0.411 — worse than the 0.58–0.60 you get by blindly copying
    the nearest stored example's label, on **every** seed. If a model applies a correct rule
    worse than it copies, no amount of rule quality can help. (Called **C5** in the docs.)
  - **You can't have it both ways.** For rules to beat copying, the rule must be visible to the
    embeddings (so the agent can find the pattern) but copying must *fail* (so there's something
    to win). Measuring retrieval the way the protocol actually retrieves showed these can't
    co-occur: making a rule visible to the embeddings makes it visible to retrieval too, so
    copying never fell to chance in the first place. (Written **C1 ⇒ ¬C3** in the docs.)
- **Rule-Shift (4)** broke copying with *time* instead of geometry, on a 12B model that *can*
  apply a rule. It missed its pre-registered bar, and the autopsy is the finding: forced to
  crystallize with no trigger, all four shut seeds wrote *correct* rules — so the signal was
  there. Replaying the trigger on real per-trial correctness then showed why it still doesn't
  fire, and it is not a tidy single cause: the trigger runs with **no margin**, and its own
  noise threshold varies enough between estimates (up to 1.31× on identical data) to decide a
  seed's outcome by itself.
  ([Details below](#act-three--rule-shift-breaking-copying-with-time-ran-verdict-miss-with-one-loud-exception).)

Along the way: **five** bugs that silently corrupted the signal (two made "surprise" a
constant, one flattened it for 2 of 3 classes, and two injected the model's raw
chain-of-thought into the arm under test — the last found in an audit *after* the headline
run, and disclosed in place rather than quietly re-run; each caught by reading raw values and
stored artifacts, not accuracy curves); a theory review that
disproved the project's own original mechanism and replaced it with one where **every claim is
an executable test** ([tests/test_kernel_theory.py](tests/test_kernel_theory.py)); a measured
sensitivity curve for the trigger (the original gate never fires on pure noise; the later
amended estimators fire on 5–10% of noise draws at the smaller sample sizes); and a one-line
conclusion — *compress into weights when your model is small; compress into sentences when
your model can read.*

Everything runs **locally**: `gemma3:4b` + `embeddinggemma` via Ollama, Postgres + `pgvector`.
No API keys, no cloud. Full narrative: **[docs/BLOG_POST.md](docs/BLOG_POST.md)**.

---

## The idea: lossy compression of surprise — into rules, not weights

The three-sentence version is above; this is the mechanism.

Titans' principle: memory should be *lossy, and surprise should decide what survives*. It
compresses surprising experience into the weights of a small MLP at test time — opaque,
fixed-capacity, and only available if you pretrain the whole architecture. This agent keeps the
same economics as a pure inference-time wrapper around a frozen model:

- **Surprise is read from the model's own logits** — entropy of the next-token distribution
  (*perceptual* surprise: the model is unsure) and NLL of the true label (*predictive*
  surprise: the model was wrong) — a black-box analogue of Titans' gradient signal.
- Surprise gates what enters the episodic store, and it is measured **with memory in context**,
  so already-solved items stop registering (the analogue of Titans' forgetting gate).
- When enough failures share one geometric axis, the agent distills the whole cluster into a
  single self-written **axiom** — auditable, portable across models, injectable into any context.

The three-tier memory that implements it:

| Tier | Table | What it holds |
|------|-------|---------------|
| **Episodic buffer** | `episodic_buffer` | Raw experiences, written when *surprising* (high prediction error) or *salient* (high entropy). |
| **Semantic core** | `semantic_core` | Compressed "axioms" — natural-language rules the agent writes about its own failure patterns. |
| **Eigen kernel** | (in-memory) | The crystallization trigger: contrastive PCA over *retrieval residuals* (query − retrieved neighbor), failures vs successes, gated by a random-matrix detectability threshold. See [docs/THEORY.md](docs/THEORY.md). |

The headline question: does compressing failures into eigen-axioms (**Treatment**) help the
agent learn faster than plain retrieval of similar past episodes (**Control / RAG**)?

---

## The experiments

Every task runs the same agent with arms toggled:

| Arm | Retrieval | Eigen-memory | What it tests |
|-----|-----------|--------------|---------------|
| **Baseline** | off | off | Can the model do this with no memory at all? |
| **Control_RAG** | on | off | Plain retrieval of top-k similar past episodes. |
| **Treatment_Eigen** | on | on | RAG **plus** crystallized eigen-axioms. |
| **Oracle_Rule** (flip task only) | off | off | The *true* rule pasted into context — the headroom ceiling. |

### Results — the static three-task negative

| | Number-game | TREC | Label-flip (held-out, frozen memory) |
|---|---|---|---|
| Rule visible to text embeddings? | ✗ | ✓ | ✓ (a classifier fit on the embeddings recovers the rule attribute: AUC 0.95–0.97) |
| One exemplar enough? | — | ✓ (copy ceiling high) | ✓ under honest measurement — copy ceiling 0.58–0.60 |
| **RAG vs no-memory** | tie — 0.60 vs 0.70¹ | **RAG wins — 0.80 vs 0.75**¹ | **RAG wins — 0.60 vs 0.29**² |
| **Eigen vs RAG** | tie¹ | loses — 0.75 vs 0.80¹ | exact tie — 0.62 vs 0.60 (0–1 axioms; identical predictions on 3/4 seeds)² |
| Why rules can't win here | embeddings can't see the rule | one exemplar already settles it | **model applies the rule worse than it copies** (0.41 with rule given < 0.59 copy ceiling, 4/4 seeds) **+ visible-to-embeddings ⇒ visible-to-retrieval** |

¹ final-batch accuracy, mean of 2 seeds (42, 7) — bands overlap heavily; see [FINDINGS.md](FINDINGS.md).
² held-out accuracy, mean of 4 seeds (42, 2, 18, 23), n=45 each, temperature 0; raw data
`results/flip/comparison_results.flip.<seed>.json`, aggregate `results/flip/comparison_results.flip.aggregate.json`.
(An earlier, compromised version of this run is archived in `results_prefix_bug/` — see below.)

The **flip task is the key result**, though not the way it was designed to be. It was built to
sit in the one regime where rule-compression should win (rule embedding-visible, one exemplar
*not* enough). Three replicated findings came out of running it honestly:

1. **The model can't apply the rule** (**C5**, the executor gate). An Oracle arm with the
   *true* rule pasted into context scores 0.411 ± 0.103, below the 0.578–0.600 label-copy
   ceiling — the accuracy you'd get by ignoring the rule and copying the nearest stored
   example's label — on **all four seeds** (paired difference −0.178 ± 0.093). A 4B model
   applies a rule worse than blind copying scores, so rule-memory cannot win here regardless
   of axiom quality.
2. **Visible to the embeddings ⇒ visible to retrieval** (**C1 ⇒ ¬C3**, the static-task
   paradox). The guardrail originally measured near-chance copying with the *wrong queries*.
   Measured the way the protocol actually retrieves (held-out queries, disjoint vocabulary,
   against the train store), nearest neighbors match on the rule attribute itself (polarity:
   0.73–0.89) — because whatever generalizes across the split dominates cross-split
   similarity, and that's exactly the attribute the rule depends on. The window rules need in
   order to win closes as you open it. ([docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md)
   formalizes this and the escape routes.)
3. **The gate is calibrated, and silence was the correct output.** A synthetic ROC of the
   actual crystallization gate ([gate_roc.py](gate_roc.py)) measures a **0.00 false-positive
   rate** at pure noise, detection at 1× the *noise edge* (the largest eigenvalue that noise
   alone would produce, estimated by permutation), and full-gate firing only past ~8×. So
   the stability check (direction reproducible at |cos| > 0.95 across checks) is the binding
   constraint, and residual failure structure on the flip task never came close. The zero
   axioms weren't a malfunction; they're what a calibrated compressor does with no rank-1
   signal.

**And when a real axis exists, the compressor still writes a true rule** — re-verified on the
fixed code: 120 TREC trials → exactly one axiom (strength 1.13):

> *"Questions requiring numerical answers should be labeled NUM, and questions requiring
> location or descriptive answers should be labeled LOC."*

So the honest scope: the *compression* side works, is calibrated, and refuses to compress
noise; the *decompression* side — a model that can actually read and apply a rule — is the
frontier, and a 4B model isn't it. This repo measures that boundary instead of claiming a win.

![Learning curve](figures/learning_curve.png)

*Learning curve from the **number-game** run (experiment 1, old kernel, 2 seeds) — the arms'
error bands overlap almost completely. Rendered from `results/static/comparison_results.json`;
the flip and Rule-Shift experiments report held-out accuracy rather than curves.*

**The most revealing early artifact** — before the theory correction, the old kernel crystallized
this axiom on the number-game:

> **RULE:** The model must always output one of the specified colors (RED, BLUE, or GREEN)
> without any interpretation or analysis of the input number. It should simply pick one at random.

That is the substrate problem narrated from inside the agent: since text embeddings hide the
arithmetic rule, reasoning genuinely didn't pay, and the agent correctly induced that there was
no learnable signal it could act on — then rationalized surrender. (Full story in
[FINDINGS.md](FINDINGS.md).)

<details>
<summary>Memory cost + eigen-spectrum (supplementary)</summary>

![Memory cost](figures/memory_cost.png)

![Eigen spectrum](figures/eigen_spectrum.png)

</details>

### Act three — Rule-Shift: breaking copying with time (ran; verdict: miss, with one loud exception)

Static tasks being structurally unwinnable, the follow-up broke copying with *time*: the
label rule flips at trial 100, so stored examples keep retrieving perfectly and answering
*wrongly*, while a re-crystallized rule stays current. The executor was upgraded to a 12B
model that passes a cheap rule-following pre-test (**RFμ**) that gemma3:4b failed — with the
rule in context it scores 0.90–0.99, so the executor boundary found on the flip task is real
and crossable. The kill arm, registered in advance: RAG that weights recent
examples more heavily — the obvious cheap fix a reviewer would reach for instead of rules.

**Pilot seed**: the whole mechanism fired in sequence — surprise spiked at the shift (the
model's error on the true label jumped 0.30 → 9.78 — from the run log, not a committed
artifact), the trigger detected on exactly 3
consecutive checks, and the crystallizer wrote one legible prose rule (including a stale
exception clause a human reviewer would strike — the auditability pitch, self-demonstrating).
Held-out: **0.922 vs 0.522** for the kill arm (p = 5.6e-9, exact McNemar), beating even a copy policy given
*perfect* knowledge of which examples were stale (0.778). An audit found that the extractor
behind the original 0.911 run kept ~250 chars of trailing chain-of-thought after the rule, so
that run injected CoT residue and was not a clean test. **Rerun on the fixed extractor
(2026-08-11): the result holds.** The stored axiom is one clean rule line with no CoT, and
held-out accuracy came in at **0.922** (original: 0.911). All four control arms reproduced
within ±0.022, so the rig was stable across the comparison. Details and the audit trail:
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §6. **Five-seed verdict, scored against
the bar set in advance: miss** — a **+0.078** gain over the kill arm **against the +0.10 bar**
(the direction is real, p = 0.002, but it didn't clear the line). The trigger fired on **1 of
5 seeds**, and on 3 of the 4 seeds where it stayed shut, Treatment's predictions are
item-identical to plain RAG — no axiom, no effect, no hidden channel. (The fourth, seed 23,
differs on exactly 1 of 90 held-out items with zero axioms stored at temperature 0 — a
residual nondeterminism in the serving stack, not a memory effect, but the honest statement
is 3 of 4 rather than all four.) The autopsy
([gate_roc_v2.py](gate_roc_v2.py)): a more sensitive trigger design fires no more often than
the crude one, because on the shut seeds the statistic it watches sits at the level you'd see
from noise alone — **no signal in what the trigger measures, rather than a bar set too high**.
Whether signal existed at all was settled by the ungated ablation (run 2026-08-11): forced to
crystallize from those same windows, all four shut seeds wrote rules that map both polarities
to the correct post-shift labels. **The signal was there; the live estimator did not surface
it.** That ablation rebuilt which trials failed with a proxy cleaner than live reality, so the
gate was then replayed on real per-trial correctness across all four shut seeds (2026-08-12).
The result is less tidy than either earlier story: **λ₁ is identical between replay and live run
on every seed**, so every ratio difference is the permutation *noise edge*, which varies up to
1.31× on identical data and flips seed 18's verdict by itself. λ₁/edge on real labels spans 0.78–1.28 across all seeds, and seed 7 — whose original run never crossed the edge — reached two
of the three required detections on rerun. Conditional on detection, compression beats copying
by a wide, auditable margin. Detection is the bottleneck, but **the gate is not clearly
mis-featurized or mis-thresholded — it runs with no margin**, so seed outcomes turn on
estimator variance. The fix therefore starts with stabilizing the edge, then replacing the
threshold-and-streak pair with a sequential test that has a real error guarantee. Full ledger:
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §5–9a.

---

## Run it yourself

**Prerequisites:** Docker, [Ollama](https://ollama.com) (running locally — everything talks to
`localhost:11434`), and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Models (local, ~3-4 GB)
ollama pull gemma3:4b
ollama pull embeddinggemma

# Only needed for the Rule-Shift experiment (act three): a ~8 GB executor that
# can actually apply a rule in context, which gemma3:4b can't. Skip for 1-3.
ollama pull gemma4:12b

# 2. Config
cp .env.example .env            # defaults work out of the box

# 3. Database (Postgres + pgvector)
docker compose up -d
docker exec -i memory-db psql -U postgres -d memory_agent < schema.sql

# 4. Dependencies
uv sync

# 5. Run the experiments
uv run python simulate.py                    # number-game (default)
TASK=trec uv run python simulate.py          # TREC question classification
uv run python guardrail_flip.py 42           # flip-task pre-run gates (rule visibility, copy ceiling)
uv run python run_flip_experiment.py 42      # the 4-arm flip experiment (incl. Oracle)

# Act three — Rule-Shift (needs gemma4:12b; budget most of a day per seed)
# Measured 2026-08-10 on a single consumer GPU: ~9 s per executor call and
# ~2,050 calls per seed across the five arms, so 6+ h wall-clock. The two
# memory arms dominate (retrieval adds embedding calls per trial).
uv run python guardrail_shift.py 42          # pre-run gates for the shift task
uv run python run_shift_experiment.py 42     # the 5-arm shift experiment (incl. recency kill arm)

# 6. Generate the plots
uv run python plot_results.py
```

Tests: `uv run pytest` — includes the executable theory
([tests/test_kernel_theory.py](tests/test_kernel_theory.py)) and kernel unit tests with a fake
DB/LLM ([tests/test_kernel_consolidation.py](tests/test_kernel_consolidation.py)).

---

## What I'd do next

The Rule-Shift experiment above consumed the old version of this list (the executor pre-test,
the sample-size-aware stability threshold, and the recency kill arm all ran as pre-registered).
What its miss opens up, in order — full ledger in
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §7–8:

- ~~**The ungated-trigger ablation**~~ — **run 2026-08-11: correct on 4 of 4 shut seeds**
  (pre-registered bar was ≥2). Forced to crystallize with no gate, every shut seed wrote a
  rule mapping both polarities correctly. The signal was in the episodes and the estimator
  missed it. Artifacts: `results/shift/ungated_ablation.<seed>.json`; details in
  [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §9.
- ~~**Persist per-trial correctness**~~ — *done, and all four shut seeds replayed 2026-08-12.*
  The harness writes `trial_correct`; [gate_replay.py](gate_replay.py) recomputes the gate
  statistic on the real split versus the ablation's proxy over identical featurization. The
  one-seed answer (seed 23: real labels don't rescue the gate ⇒ featurization is the
  bottleneck) **did not generalize.** Across four seeds λ₁ is *identical* between replay and
  live run, so every ratio difference is the permutation **edge** moving, not the signal — and
  on seed 18 the same λ₁ lands on opposite sides of an edge that differs 31% on identical data.
  **The gate isn't clearly mis-featurized or mis-thresholded; it runs with no margin** (λ₁/edge
  on real labels spans 0.78–1.28 across all four seeds), so outcomes turn on estimator variance.
  Seed 7 reached streak 2-of-3 on rerun against a committed run that never crossed the edge.
  Details: [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §9a.
- **Stabilize the noise edge** — now the immediate item, ahead of featurization work. More
  permutation draws, or a pooled/smoothed edge across checks instead of an independent estimate
  each time. At these effect sizes a featurization improvement and a lucky edge draw are
  currently indistinguishable, so this gates the next experiment rather than competing with it.
- **Residual featurization** — the estimator contrasts embedding means (a deliberate
  amendment: a covariance contrast is provably blind to this shift's *location* structure — see
  `contrast_on` in [memory_kernel.py](src/eigen_memory_agent/memory_kernel.py)). The open
  question is whether a *whitened* location statistic — a shrinkage-regularized Fisher
  direction, which keeps the location sensitivity the amendment requires — beats the unwhitened
  mean difference at n≈60, d≈768. Blocked on the edge work above: not interpretable until a
  featurization change can be told apart from estimator variance.
- ~~**An anytime-valid sequential test**~~ — *built 2026-08-12 (`sequential_gate=True`, off by
  default), and the result is a warning rather than a win.* Type-I control is genuine and
  measured, and it fires where the streak rule cannot (seed 7: 0→1 axiom; seed 42: 1→2). But
  **the rules it writes are worse**: three axioms across two seeds, every one with a wrong
  branch, against the streak rule's one clean axiom. Cause: it fires *before the shift*. On
  seed 42 the shift lands at batch 11 and the sequential gate fired at batch 7, writing an
  accurate statement of the **pre-shift** rule that was false four batches later. The
  3-consecutive requirement was acting as a delay that let the post-shift signal dominate —
  a real function, not the pure conservatism it looked like. Details:
  [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §9b.
- **Validate axioms before injection** — the gap §9b exposes: the crystallizer has no notion of
  *when* a rule stopped being true, so on a non-stationary target firing faster produces
  confidently stale rules. Scoring a candidate axiom against recent trials before injecting it
  would make early firing safe instead of preventing it, and the §9 ablation (4/4 correct rules
  when forced) suggests the accept rate would be usable.
- **The label-noise wedge** — the third way to break copying (after geometry, which failed,
  and time, which split): exemplar-copying inherits annotator noise one-for-one at
  retrieval; a crystallized rule is a pooled majority-policy estimator, noise-free at
  inference. Pre-registerable as a noise-rate sweep on the *static* task — reopening the
  regime that the visible-to-embeddings ⇒ visible-to-retrieval result closed.

Crystallization precision so far, scored blind against the planted rules — three axioms have
ever fired across all experiments, so this is a count, not a rate: **TREC** 1 axiom, correct.
**Flip** 1 axiom, half credit (it found the right axis but inverted the mapping). **Rule-Shift**
1 axiom, legible and correct, with one stale clause a reviewer would veto — which is the
auditability argument working, and, per the audit note above, ~250 chars of trailing
chain-of-thought a reviewer would also strike.

## Going deeper

- **[docs/BLOG_POST.md](docs/BLOG_POST.md)** — the full narrative: Titans lineage, the corrected
  compressor, the flip showdown, and the executor gate.
- **[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md)** — the visible-to-embeddings result, the success
  ladder, and the pre-registered Rule-Shift design.
- **[FINDINGS.md](FINDINGS.md)** — the number-game and TREC results, and the constant-surprise
  bug archaeology.
- **[docs/THEORY.md](docs/THEORY.md)** — the corrected mechanism: contrastive PCA over
  *retrieval residuals* with a detectability-gated trigger. Every claim is executable, and it
  retro-explains the old garbage axioms quantitatively.
- **[docs/C1_C3_TASK.md](docs/C1_C3_TASK.md)** — the flip task: designed, guardrail-gated, and
  run (first live run 2026-07-14; the hypothesis was not supported, and the executor gate
  was discovered).
- **[docs/USE_CASES.md](docs/USE_CASES.md)** — where eigen-memory *would* win: the winning
  conditions and a scorecard.
- **[docs/PRIOR_ART.md](docs/PRIOR_ART.md)** — how this sits in the literature (Titans, RepE,
  cPCA, BBP, the verbal-consolidation line).
- **[docs/VALID_EXPERIMENT.md](docs/VALID_EXPERIMENT.md)** — the pre-registered experimental
  design the flip task implements.

## Repository layout

```
# Experiment entrypoints
simulate.py                     # number-game / TREC experiment: 3 arms x N seeds
run_flip_experiment.py          # the 4-arm flip experiment (Baseline/Oracle/RAG/Eigen)
run_shift_experiment.py         # act three: the Rule-Shift experiment (incl. recency kill arm)

# Pre-run gates and qualification (cheap, run before spending LLM budget)
guardrail_flip.py               # flip-task gates measured under protocol conditions
guardrail_shift.py              # Rule-Shift pre-run gates — no LLM, no Postgres
run_rfmu.py                     # cheap pre-test: can this model apply a rule at all?

# Analysis of committed artifacts
aggregate_flip.py               # multi-seed aggregation + executor gate from guardrails
derisk_pilot.py                 # complaint-driven pilot checks (no new LLM arms)
run_trec_verify.py              # re-verifies the TREC one-true-axiom claim
plot_results.py                 # learning curve, memory cost, eigen-spectrum

# Gate calibration (synthetic ROC of the real crystallization gate)
gate_roc.py                     # v1 trigger: fire on a run of checks beating the noise edge
gate_roc_mean.py                # the 2026-07-17 mean-contrast amendment
gate_roc_v2.py                  # v2: evidence accumulation — the act-three autopsy
ungated_ablation.py             # bypasses the gate entirely: was the signal there at all?
gate_replay.py                  # replays the gate on real vs proxy correctness

schema.sql                      # episodic_buffer + semantic_core (pgvector)
src/config.py                   # env-driven DB / Ollama / embedding-model config
src/dataset.py                  # number-game + TREC loader + flip-task generator
src/paths.py                    # canonical artifact locations (results/, figures/)
src/eigen_memory_agent/
  agent.py                      # the agent loop: predict, measure surprise, learn
  memory_kernel.py              # contrastive residual PCA -> gated axiom crystallization
docs/                           # theory, task design, prior art, blog post, next experiment
tests/
  test_kernel_theory.py         # the theory as executable tests (see docs/THEORY.md)
  test_kernel_consolidation.py  # kernel unit tests (fake DB/LLM)

# Committed artifacts — every number in the docs traces to one of these
results/static/                 # number-game + TREC runs
results/flip/                   # label-flip: per-seed runs, guardrails, aggregate
results/shift/                  # Rule-Shift: per-seed runs, guardrails, derisk
results/calibration/            # trigger sensitivity sweeps + executor pre-test results
figures/                        # learning curve, memory cost, eigen-spectrum
results_prefix_bug/             # the compromised first multi-seed run, kept as the record
                                #   (see its README for which two bugs corrupted it)
```

*Built with heavy LLM pair-work; every theoretical claim is enforced by tests, and every number
in the docs traces to a committed artifact or is labeled as a live smoke run.*

## License

[MIT](LICENSE). Datasets are downloaded at runtime, not redistributed here, and keep their own
terms — see [docs/DATASETS.md](docs/DATASETS.md).
