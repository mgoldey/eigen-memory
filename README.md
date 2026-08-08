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
| 3 | **Label-flip** — purpose-built, 4 seeds, held-out | exact tie | The trigger never fired, so the agent wrote no rules and its predictions were *identical* to RAG's. Two structural reasons, both below. |
| 4 | **Rule-Shift** — the rule flips mid-run, 12B model, 5 seeds | **miss** — +0.078 against a +0.10 bar set in advance | The trigger fired on only 1 of 5 seeds. But on that seed, the rule the agent wrote beat copying **0.911 vs 0.522**. |

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
  apply a rule. It missed its pre-registered bar, and the autopsy is the finding: on the seeds
  where the trigger never fired, the statistic it watches sits at noise level — nothing to
  detect *in what it measures*, rather than a threshold set too high. The fix belongs in what
  the trigger measures, not in where the bar sits.
  ([Details below](#act-three--rule-shift-breaking-copying-with-time-ran-verdict-miss-with-one-loud-exception).)

Along the way: **four** bugs that silently corrupted the signal (three made "surprise" a
constant; the fourth injected the model's raw chain-of-thought into the arm under test — each
caught by reading raw values and stored artifacts, not accuracy curves); a theory review that
disproved the project's own original mechanism and replaced it with one where **every claim is
an executable test** ([tests/test_kernel_theory.py](tests/test_kernel_theory.py)); a measured
sensitivity curve for the trigger, showing it never fires on pure noise; and a one-line
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
   rate** at pure noise, detection at 1× the *noise edge* — the largest eigenvalue that noise
   alone would produce, estimated by permutation —, and full-gate firing only past ~8× —
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
model's error on the true label jumped 0.30 → 9.78), the trigger detected on exactly 3
consecutive checks, and the crystallizer wrote one legible prose rule (including a stale
exception clause a human reviewer would strike — the auditability pitch, self-demonstrating).
Held-out: **0.911 vs 0.522** for the kill arm (p = 1.6e-8), beating even a copy policy given
*perfect* knowledge of which examples were stale (0.778). **Five-seed verdict, scored against
the bar set in advance: miss** — a **+0.078** gain over the kill arm **against the +0.10 bar**
(the direction is real, p = 0.002, but it didn't clear the line). The trigger fired on **1 of
5 seeds**, and on every seed where it stayed shut, Treatment's predictions are item-identical
to plain RAG — no axiom, no effect, no hidden channel. The autopsy
([gate_roc_v2.py](gate_roc_v2.py)): a more sensitive trigger design fires no more often than
the crude one, because on the shut seeds the statistic it watches sits at the level you'd see
from noise alone — **no signal in what the trigger measures, rather than a bar set too high**.
(Whether signal exists at all in some *other* featurization is exactly what the ungated
ablation below — written, not yet run — is meant to settle.) Conditional on
detection, compression beats copying by a wide, auditable margin. Detecting that a rule has
changed, at the signal levels a real workload offers, is the bottleneck — and the fix belongs
in *what the trigger measures*, not in where the bar sits. Full ledger:
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §5–8.

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

# Act three — Rule-Shift (needs gemma4:12b; ~1-2 h per seed)
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

- **The ungated-trigger ablation** — feed the gate-shut seeds' post-shift windows to a
  crystallizer on a fixed schedule, no gate, and score what it writes against the planted
  rule. This is the arbiter: a correct rule means the signal was there and the *estimator* missed it; garbage
  means the calibration called it right. It also finally isolates the spectral gate as a
  variable — the one ablation this repo has owed since the flip experiment.
- **Residual featurization** (conditional on the ablation saying "signal was there") — the
  estimator currently contrasts raw embedding means; the shut seeds show that stream carries
  noise-level contrast on most vocabularies.
- **The label-noise wedge** — the third way to break copying (after geometry, which failed,
  and time, which split): exemplar-copying inherits annotator noise one-for-one at
  retrieval; a crystallized rule is a pooled majority-policy estimator, noise-free at
  inference. Pre-registerable as a noise-rate sweep on the *static* task — reopening the
  regime that the visible-to-embeddings ⇒ visible-to-retrieval result closed.

Crystallization precision so far, scored blind against the planted rules: TREC 1/1, flip 1/2
(right axis, inverted mapping), Rule-Shift 1/1 legible-and-correct (with one stale clause a
reviewer would veto — which is the auditability argument working).

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
