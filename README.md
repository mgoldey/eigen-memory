# Eigen-Memory: compressing an agent's surprises into rules instead of weights

**Titans ([Behrouz et al. 2024](https://arxiv.org/abs/2501.00663)) keeps what surprises it and
compresses it into neural weights — but Titans is an architecture you pretrain from scratch.
This project rebuilds those economics on a frozen 4B model with no training loop, compressing
surprising failures into short, legible English rules — and runs three controlled experiments to
find where that beats plain retrieval. It never does, and the instrumented reasons why are the
deliverable:**

1. **Number-game** (classify integers by a hidden arithmetic rule) — text embeddings can't see
   primality, so *no* memory can work; the agent even crystallized an axiom telling itself to
   guess randomly, which was locally correct.
2. **TREC** (question-type classification) — retrieval alone wins (RAG 0.80 vs 0.75 no-memory):
   the rig detects real memory benefits. But one retrieved exemplar settles each question, so
   rules have nothing to add.
3. **A purpose-built label-flip task** — the corrected kernel found the planted rule axis live
   and crystallized exactly one, correct-axis axiom… and still lost to RAG, because an Oracle
   arm revealed the 4B model applies a *true* rule worse than nearest-neighbor label-copying
   scores (0.47 vs a 0.57 copy ceiling; single seed, n=45 — suggestive, not significant).
   Rule-memory was unwinnable regardless of axiom quality — a new pre-registerable gate
   (**C5**) for anyone putting verbal memory on a small model.

Along the way: three bugs that had silently turned "surprise" into a **constant** (caught by
reading the raw signal values, not the accuracy curve); a theory review that disproved the
project's own original mechanism and replaced it with one where **every claim is an executable
test** ([tests/test_kernel_theory.py](tests/test_kernel_theory.py)); and a one-line conclusion —
*compress into weights when your model is small; compress into sentences when your model can
read.*

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

### Results — a rigorous three-task negative

| | Number-game | TREC | Label-flip (held-out, frozen memory) |
|---|---|---|---|
| Rule visible to text embeddings? | ✗ | ✓ | ✓ (probe AUC 0.947) |
| One exemplar enough? | — | ✓ (copy ceiling m high) | ✗ (m = 0.567, near chance) |
| **RAG vs no-memory** | tie — 0.60 vs 0.70¹ | **RAG wins — 0.80 vs 0.75**¹ | RAG wins — 0.53 vs 0.22² |
| **Eigen vs RAG** | tie¹ | loses — 0.75 vs 0.80¹ | loses — 0.36 vs 0.53² |
| Why rules can't win here | substrate blind to the rule | one exemplar already settles it | **C5: executor below the copy ceiling** |

¹ final-batch accuracy, mean of 2 seeds (42, 7) — bands overlap heavily; see [FINDINGS.md](FINDINGS.md).
² held-out accuracy, single seed, n=45 — demonstration scale; see [docs/C1_C3_TASK.md](docs/C1_C3_TASK.md).

The **flip task is the key result**. It was built to sit in the one regime where rule-compression
should win (rule embedding-visible, one exemplar *not* enough — four generator designs failed the
guardrails before one passed). The kernel did its job: the detectability gate stayed shut on the
number-game (**zero** axioms, where the old ungated kernel emitted 15–20 confabulations) and
crystallized exactly **one** axiom on the flip task — on the correct axis:

> *"If the input describes a completed action or a status update without a clear indication of a
> problem needing immediate attention, predict DEFER; otherwise, predict ESCALATE."*

Completed-action-vs-needs-attention **is** the planted request/report polarity. The 4B model then
fumbled the conditional — it collapsed the per-topic flip into one global mapping — and the
Oracle arm showed why no axiom could have saved it: with the *true* rule in context, the model
scores 0.467, **below** the 0.567 you'd get by mindlessly copying your nearest neighbor's label.
When the executor's rule-following sits below the copy ceiling, rule-memory cannot win regardless
of memory quality. That's **C5**, the fifth pre-registerable gate this project contributes.

So the honest scope: the *compression* side works and refuses to compress noise; the
*decompression* side — a model that can actually read and apply a rule — is the frontier, and a
4B model isn't it. This repo measures that boundary instead of claiming a win.

![Learning curve](learning_curve.png)

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

![Memory cost](memory_cost.png)

![Eigen spectrum](eigen_spectrum.png)

</details>

---

## Run it yourself

**Prerequisites:** Docker, [Ollama](https://ollama.com), and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Models (local, ~3-4 GB)
ollama pull gemma3:4b
ollama pull embeddinggemma

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

# 6. Generate the plots
uv run python plot_results.py
```

Tests: `uv run pytest` — includes the executable theory
([tests/test_kernel_theory.py](tests/test_kernel_theory.py)) and kernel unit tests with a fake
DB/LLM ([tests/test_kernel_consolidation.py](tests/test_kernel_consolidation.py)).

---

## What I'd do next

- **Swap in a stronger executor.** C5 says the boundary is the model, not the memory: rerun the
  flip task with a model that can apply a rule better than it copies (cheap — the detectability
  gate makes crystallization rare by design).
- **Run the flip task at ≥5 seeds.** The C5 result is one seed at demonstration scale; the
  protocol itself pre-registers more.
- **Ablate the surprise gate itself** (store-everything vs gated) — the banner mechanism has
  never been isolated as a variable. (Note: crystallization already deliberately consumes
  *ungated* residuals; surprise gates only the episodic store.)
- **Validate axiom quality before injection** — a wrong self-written rule poisons the context
  (the flip run's report-polarity cells show exactly this).

## Going deeper

- **[docs/BLOG_POST.md](docs/BLOG_POST.md)** — the full narrative: Titans lineage, the corrected
  compressor, the flip showdown, and C5.
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
simulate.py                     # number-game / TREC experiment: 3 arms x N seeds
run_flip_experiment.py          # the 4-arm flip experiment (Baseline/Oracle/RAG/Eigen)
guardrail_flip.py               # pre-run gates: probe AUC + copy ceiling m
plot_results.py                 # learning curve, memory cost, eigen-spectrum
schema.sql                      # episodic_buffer + semantic_core (pgvector)
src/config.py                   # env-driven DB / Ollama config
src/dataset.py                  # number-game + TREC loader + flip-task generator
src/eigen_memory_agent/
  agent.py                      # the agent loop: predict, measure surprise, learn
  memory_kernel.py              # contrastive residual PCA -> gated axiom crystallization
docs/                           # theory, task design, prior art, blog post
tests/
  test_kernel_theory.py         # the theory as executable tests (see docs/THEORY.md)
  test_kernel_consolidation.py  # kernel unit tests (fake DB/LLM)
```

*Built with heavy LLM pair-work; every theoretical claim is enforced by tests, and every number
in the docs traces to a committed artifact or is labeled as a live smoke run.*
