# Eigen-Memory Agent

**An LLM agent that learns hidden rules from its own mistakes — and a controlled experiment testing whether a novel "eigen-memory" beats plain retrieval.**

The agent plays a game: classify a number as `RED`, `BLUE`, or `GREEN`. It is never told the
rule (`prime → RED`, `divisible by 5 → BLUE`, `else → GREEN`). It only gets *correct / incorrect*
feedback, and must induce the rule by remembering and reflecting on what surprised it.

It runs entirely **locally** — a 4B-parameter model (`gemma3:4b`) via Ollama, with a
Postgres + `pgvector` memory store. No API keys, no cloud.

---

## The idea: surprise-gated, self-consolidating memory

The agent has a three-tier memory inspired by how brains consolidate experience:

| Tier | Table | What it holds |
|------|-------|---------------|
| **Episodic buffer** | `episodic_buffer` | Raw experiences, written only when *surprising* (high prediction error) or *salient* (high token entropy). |
| **Semantic core** | `semantic_core` | Compressed "axioms" — natural-language rules the agent writes about its own failure patterns. |
| **Eigen kernel** | (in-memory PCA) | Incremental PCA over surprise vectors. When a principal direction stabilizes, the agent introspects on the failures along it and crystallizes an axiom. |

Two signals gate what gets remembered, both read from the model's own logits:

- **Perceptual surprise** — entropy of the next-token distribution (the model is *unsure*).
- **Predictive surprise** — negative log-likelihood of the *true* label (the model was *wrong*).

The headline question: does compressing failures into eigen-axioms (**Treatment**) help the
agent learn faster than plain retrieval of similar past episodes (**Control / RAG**)?

---

## The experiment

Three arms, each run over multiple seeds and averaged:

| Arm | Retrieval | Eigen-memory | What it tests |
|-----|-----------|--------------|---------------|
| **Baseline** | off | off | Can a 4B model do this with no memory at all? |
| **Control_RAG** | on | off | Plain retrieval of top-k similar past episodes. |
| **Treatment_Eigen** | on | on | RAG **plus** crystallized eigen-axioms. |

### Result — a rigorous two-substrate negative result

The honest headline: **eigen-memory never beats plain RAG** — tested on *two* substrates — and the
more interesting finding is *why*, which differs between them. Getting the mechanism to genuinely
work first meant fixing bugs that had quietly turned the "surprise" signal into a **constant**, so
the original experiment was measuring nothing. Once it worked, I ran two tasks:

| | Number-game (arithmetic rule) | TREC (question-type rule) |
|---|---|---|
| Rule visible to text embeddings? | ✗ | ✓ |
| **Does RAG beat no-memory?** | No — 0.46 vs 0.47 | **Yes — 0.80 vs 0.75** |
| **Does Eigen beat RAG?** | No | No — 0.75 vs 0.80 |
| Why the effect can't show | substrate blind to the rule | ceiling effect; one exemplar already settles each question |

The **TREC arm is the key result**: when the rule *is* visible in embedding space, plain RAG
clearly beats no-memory — which proves the experimental rig can detect a real memory benefit.
Eigen-memory still doesn't beat it, because TREC is solvable from a single retrieved exemplar, so
there is nothing for rule-compression to add. The number-game fails earlier still: text embeddings
of bare integers don't cluster by primality, so retrieval is blind to the rule and even RAG can't
help.

This repo is therefore a **post-mortem of an idea**, not a victory lap — which is the point. It
isolates the precise condition eigen-memory would need to win (a rule that's embedding-visible
*and* not solvable from one exemplar) and shows neither task meets it. Full analysis in
**[FINDINGS.md](FINDINGS.md)**.

The number-game learning curve below shows how little RAG improves on no-memory there — retrieval
barely helps because the neighbors it finds aren't informative for an *arithmetic* rule embedded in
*text* space:

![Learning curve](learning_curve.png)

![Memory cost](memory_cost.png)

**A real axiom the agent crystallized about its own failures** — it concluded that the winning
move is to *stop reasoning and guess*, which is locally correct precisely because the substrate
hides the rule:

> **RULE:** The model must always output one of the specified colors (RED, BLUE, or GREEN)
> without any interpretation or analysis of the input number. It should simply pick one at random.

<details>
<summary>Eigen-spectrum evolution (supplementary)</summary>

![Eigen spectrum](eigen_spectrum.png)

How the explained variance of the top principal components of the surprise-vector space
evolves as the agent accumulates failures.
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

# 5. Run the experiment (multi-seed; takes a while on CPU)
uv run python simulate.py              # number-game (default)
TASK=trec uv run python simulate.py    # TREC question classification (rule visible in embeddings)

# 6. Generate the plots
uv run python plot_results.py
```

`TASK=trec` runs the *same agent* on a substrate where the hidden rule is embedding-visible — the
re-test that shows RAG beating no-memory (see [FINDINGS.md](FINDINGS.md)). It auto-downloads and
caches the TREC dataset on first run.

Tests: `uv run pytest`.

---

## What I'd do next

- More seeds + a significance test (paired, per-batch) for a statistically powered claim.
- Harder environments (compositional rules, distribution shift) where memory should matter more.
- Validate axiom *quality* before injection — a wrong self-written rule can poison the context.

## Going deeper

- **[FINDINGS.md](FINDINGS.md)** — the honest result and why the task can't show a difference.
- **[docs/USE_CASES.md](docs/USE_CASES.md)** — where eigen-memory *would* win: the four winning
  conditions, a scorecard, and deep dives on coding self-correction and user-preference learning.
- **[docs/VALID_EXPERIMENT.md](docs/VALID_EXPERIMENT.md)** — the experiment that could decisively test it.
- **[docs/PRIOR_ART.md](docs/PRIOR_ART.md)** — how this sits in the literature.

## Repository layout

```
simulate.py                     # the experiment: 3 arms x N seeds
plot_results.py                 # learning curve, memory cost, eigen-spectrum
schema.sql                      # episodic_buffer + semantic_core (pgvector)
src/config.py                   # env-driven DB / Ollama config
src/dataset.py                  # the hidden-rules game (number) + TREC loader (TASK=trec)
src/eigen_memory_agent/
  agent.py                      # the agent loop: predict, measure surprise, learn
  memory_kernel.py              # incremental PCA -> axiom crystallization
docs/superpowers/               # design spec + implementation plan
tests/                          # unit tests (surprise extraction)
```
