# Eigen-Memory: compressing an agent's surprises into rules instead of weights

Titans ([Behrouz et al. 2024](https://arxiv.org/abs/2501.00663)) keeps what surprises it and
compresses it into neural weights — but Titans is an architecture you pretrain from scratch.
This project rebuilds those economics as a pure inference-time wrapper around a frozen model:
surprising failures get compressed into short, legible English rules instead of weights.

Four controlled experiments ask where that beats plain retrieval.

**It never does — and the instrumented reasons why are the deliverable.**

| # | Experiment | Verdict | Why |
|---|------------|---------|-----|
| 1 | **Number-game** — classify integers by a hidden arithmetic rule | tie | Substrate is blind to the rule: text embeddings can't see primality, so *no* memory can work. |
| 2 | **TREC** — question-type classification | RAG wins (0.80 vs 0.75) | The rig detects real memory benefits — but one retrieved exemplar settles each question, so rules have nothing to add. |
| 3 | **Label-flip** — purpose-built, 4 seeds, held-out | exact tie | The gate stayed shut; eigen-memory degenerated to *exactly* RAG's predictions. **C5** + **C1 ⇒ ¬C3** explain why (below). |
| 4 | **Rule-Shift** — the rule flips mid-run, 12B executor, 5 seeds | **miss** — Δ+0.078 vs a +0.10 pre-registered bar | Gate fired on 1 of 5 seeds. But *conditional on detection*, the crystallized rule beat copying **0.911 vs 0.522**. |

Experiment 1 is a substrate failure and 2 is a task failure — neither says anything about
rule-compression itself. Experiments **3 and 4 are the real results**:

- **Flip (3)** produced two replicated discoveries. **C5** — with the *true* rule pasted into
  context, the 4B executor scores 0.411, below the 0.58–0.60 nearest-neighbor copy ceiling, on
  **every** seed, making rule-memory unwinnable regardless of axiom quality. And **C1 ⇒ ¬C3** —
  measuring retrieval the way the protocol actually retrieves showed that making a rule
  embedding-visible makes it retrieval-visible too, so copying never fell to chance in the
  first place.
- **Rule-Shift (4)** broke copying with *time* instead of geometry, on a 12B executor that
  clears the C5 boundary. It missed its pre-registered bar, and the autopsy is the finding:
  the gate-shut seeds sit at **noise-level contrast** — signal-starved, not threshold-starved.
  The bottleneck is the featurization, not the threshold. ([Details below](#act-three--rule-shift-breaking-copying-with-time-ran-verdict-miss-with-one-loud-exception).)

Along the way: **four** bugs that silently corrupted the signal (three made "surprise" a
constant; the fourth injected the model's raw chain-of-thought into the arm under test — each
caught by reading raw values and stored artifacts, not accuracy curves); a theory review that
disproved the project's own original mechanism and replaced it with one where **every claim is
an executable test** ([tests/test_kernel_theory.py](tests/test_kernel_theory.py)); a
**measured ROC for the crystallization gate** (false-positive rate 0.00; the stability check,
not the noise edge, is the binding constraint); and a one-line conclusion — *compress into
weights when your model is small; compress into sentences when your model can read.*

Everything runs **locally**: `gemma3:4b` + `embeddinggemma` via Ollama, Postgres + `pgvector`.
No API keys, no cloud. Full narrative: **[docs/BLOG_POST.md](docs/BLOG_POST.md)**.

---

## The idea: lossy compression of surprise — into rules, not weights

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
| Rule visible to text embeddings? | ✗ | ✓ | ✓ (probe AUC 0.95–0.97) |
| One exemplar enough? | — | ✓ (copy ceiling high) | ✓ under honest measurement — copy ceiling 0.58–0.60 |
| **RAG vs no-memory** | tie — 0.60 vs 0.70¹ | **RAG wins — 0.80 vs 0.75**¹ | **RAG wins — 0.60 vs 0.29**² |
| **Eigen vs RAG** | tie¹ | loses — 0.75 vs 0.80¹ | exact tie — 0.62 vs 0.60 (0–1 axioms; identical predictions on 3/4 seeds)² |
| Why rules can't win here | substrate blind to the rule | one exemplar already settles it | **C5 (0.41 Oracle < 0.59 ceiling, 4/4 seeds) + C1 ⇒ ¬C3** |

¹ final-batch accuracy, mean of 2 seeds (42, 7) — bands overlap heavily; see [FINDINGS.md](FINDINGS.md).
² held-out accuracy, mean of 4 seeds (42, 2, 18, 23), n=45 each, temperature 0; raw data
`results/flip/comparison_results.flip.<seed>.json`, aggregate `results/flip/comparison_results.flip.aggregate.json`.
(An earlier, compromised version of this run is archived in `results_prefix_bug/` — see below.)

The **flip task is the key result**, though not the way it was designed to be. It was built to
sit in the one regime where rule-compression should win (rule embedding-visible, one exemplar
*not* enough). Three replicated findings came out of running it honestly:

1. **C5 — the executor gate.** An Oracle arm with the *true* rule pasted into context scores
   0.411 ± 0.103, below the 0.578–0.600 nearest-neighbor label-copy ceiling, on **all four
   seeds** (paired Oracle − ceiling = −0.178 ± 0.093). A 4B model applies a rule worse than
   blind copying scores — so rule-memory cannot win here regardless of axiom quality.
2. **C1 ⇒ ¬C3 — the static-task paradox.** The guardrail originally measured near-chance
   copying with the *wrong queries*. Measured the way the protocol actually retrieves
   (held-out queries, disjoint vocabulary, against the train store), nearest neighbors match
   on the rule attribute itself (polarity: 0.73–0.89) — because whatever generalizes across
   the split dominates cross-split similarity, and that's exactly the attribute the rule
   depends on. Making a rule embedding-visible makes it retrieval-visible: the "eigen window"
   closes as you open it. ([docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) formalizes this
   and the escape routes.)
3. **The gate is calibrated, and silence was the correct output.** A synthetic ROC of the
   actual crystallization gate ([gate_roc.py](gate_roc.py)) measures a **0.00 false-positive
   rate** at pure noise, detection at 1× the noise edge, and full-gate firing only past ~8× —
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

Static tasks being structurally unwinnable (C1 ⇒ ¬C3), the follow-up broke copying with
*time*: the label rule flips at trial 100, stale exemplars keep retrieving perfectly and
answering wrongly, and a re-crystallized rule stays current. Executor upgraded to a 12B
that passes the RFμ qualification gemma3:4b failed (Oracle arms score 0.90–0.99 — the C5
executor boundary is real and crossable). Pre-registered kill arm: recency-weighted RAG.

**Pilot seed**: the whole banner mechanism fired in sequence — surprise spiked 0.30 → 9.78
NLL at the shift, the gate detected on exactly 3 consecutive checks, and the crystallizer
wrote one legible prose rule (with a stale exception clause a human reviewer would
strike — the auditability pitch, self-demonstrating). Held-out: **0.911 vs 0.522** for the
kill arm (McNemar p = 1.6e-8), above even a copy policy with *perfect* staleness filtering
(0.778). **Five-seed pre-registered verdict: miss** — pooled Δ vs the kill arm **+0.078
against a +0.10 bar** (direction real, p = 0.002); the gate fired on **1 of 5 seeds**, and
on every gate-shut seed Treatment's predictions are item-identical to plain RAG — no
axiom, no effect, no hidden channel. The autopsy ([gate_roc_v2.py](gate_roc_v2.py)): a
budget-calibrated evidence-accumulating gate fires no more often than the crude streak
rule, because the shut seeds sit at noise-level contrast (λ/edge 0.81–0.85, noise
averages 0.87) — **signal-starved, not threshold-starved**. Conditional on detection,
compression beats copying by a wide, auditable margin; detection at real-world SNR is the
bottleneck, and the bottleneck is the featurization, not the threshold. Full ledger:
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
# clears the C5 boundary gemma3:4b fails. Skip if you're only running 1-3.
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
uv run python guardrail_flip.py 42           # flip-task pre-run gates (probe AUC, copy ceiling m)
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

The Rule-Shift experiment above consumed the old version of this list (RFμ, the
sample-size-aware stability threshold, and the recency kill arm all ran as pre-registered).
What its miss opens up, in order — full ledger in
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §7–8:

- **The ungated-trigger ablation** — feed the gate-shut seeds' post-shift windows to a
  crystallizer on a fixed schedule, no gate, and G4-score what it writes. This is the
  arbiter: a correct rule means the signal was there and the *estimator* missed it; garbage
  means the calibration called it right. It also finally isolates the spectral gate as a
  variable — the one ablation this repo has owed since the flip experiment.
- **Residual featurization** (conditional on the ablation saying "signal was there") — the
  estimator currently contrasts raw embedding means; the shut seeds show that stream carries
  noise-level contrast on most vocabularies.
- **The label-noise wedge** — the third way to break copying (after geometry, which failed,
  and time, which split): exemplar-copying inherits annotator noise one-for-one at
  retrieval; a crystallized rule is a pooled majority-policy estimator, noise-free at
  inference. Pre-registerable as a noise-rate sweep on the *static* task — reopening the
  regime C1 ⇒ ¬C3 closed.

Crystallization precision so far, scored against planted rules (G4): TREC 1/1, flip 1/2
(right axis, inverted mapping), Rule-Shift 1/1 legible-and-correct (with one stale clause a
reviewer would veto — which is the auditability argument working).

## Going deeper

- **[docs/BLOG_POST.md](docs/BLOG_POST.md)** — the full narrative: Titans lineage, the corrected
  compressor, the flip showdown, and C5.
- **[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md)** — the C1 ⇒ ¬C3 theorem, the success
  ladder, and the pre-registered Rule-Shift design.
- **[FINDINGS.md](FINDINGS.md)** — the number-game and TREC results, and the constant-surprise
  bug archaeology.
- **[docs/THEORY.md](docs/THEORY.md)** — the corrected mechanism: contrastive PCA over
  *retrieval residuals* with a detectability-gated trigger. Every claim is executable, and it
  retro-explains the old garbage axioms quantitatively.
- **[docs/C1_C3_TASK.md](docs/C1_C3_TASK.md)** — the flip task: designed, guardrail-gated, and
  run (first live run 2026-07-14; H1 not supported, C5 discovered).
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
guardrail_shift.py              # Rule-Shift gates G1 + G2 — no LLM, no Postgres
run_rfmu.py                     # RFmu: qualifies an executor against the C5 boundary

# Analysis of committed artifacts
aggregate_flip.py               # multi-seed aggregation + C5 gate from guardrail artifacts
derisk_pilot.py                 # complaint-driven pilot checks (no new LLM arms)
run_trec_verify.py              # re-verifies the TREC one-true-axiom claim
plot_results.py                 # learning curve, memory cost, eigen-spectrum

# Gate calibration (synthetic ROC of the real crystallization gate)
gate_roc.py                     # v1: streak-over-max-edge, the flip-era estimator
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
results/calibration/            # gate ROC sweeps (v1/mean/v2) + RFmu qualification
figures/                        # learning curve, memory cost, eigen-spectrum
results_prefix_bug/             # the compromised first multi-seed run, kept as the record
```

*Built with heavy LLM pair-work; every theoretical claim is enforced by tests, and every number
in the docs traces to a committed artifact or is labeled as a live smoke run.*

## License

[MIT](LICENSE). Datasets are downloaded at runtime, not redistributed here, and keep their own
terms — see [docs/DATASETS.md](docs/DATASETS.md).
