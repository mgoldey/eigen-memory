# Findings

> Honest write-up of what the experiment actually showed. Results are the mean of
> **2 seeds** (42, 7), 100 trials each, on a local `gemma3:4b` model.

## TL;DR

<!-- TLDR_PLACEHOLDER: one-paragraph honest summary once numbers are in -->

## Final accuracy (mean of 2 seeds, last batch)

| Arm | Final accuracy |
|-----|----------------|
| Baseline (no memory) | <!-- ACC_BASELINE --> |
| Control_RAG (retrieval) | <!-- ACC_RAG --> |
| Treatment_Eigen (retrieval + axioms) | <!-- ACC_EIGEN --> |

![Learning curve](learning_curve.png)

## What I had to fix before the experiment was even valid

The most important part of this project is *not* the architecture — it's that the original
experiment was measuring nothing. Three separate bugs each made the "surprise" signal a
**constant**, so memory was being gated on noise:

1. **Flat 2.0** — an `UnboundLocalError` when the true-label token was missing from the
   logprobs was silently caught and replaced with a hardcoded `2.0`. Every recorded surprise
   score was exactly `2.00`.
2. **Flat 7.0** — after fixing (1), a completion-style probe prompt (`"...the label:"`) was
   fed to a *chat* model, which replies with prose. The label was never the first token, so
   surprise collapsed to the missing-token fallback (`7.0`) for every item.
3. Fixed by constraining the probe to emit exactly one label word, so its logprob is a real,
   varied prediction-error signal (`1.55, 4.62, 5.45, ...`).

**Lesson:** a sophisticated-looking pipeline can produce authoritative numbers while
measuring a constant. The only way I caught it was instrumenting and *looking at the actual
surprise values*, not just the final accuracy. This is the engineering-judgment story.

## The result, honestly

<!-- RESULT_NARRATIVE_PLACEHOLDER:
  Did Eigen beat RAG? By how much? Is it within seed-to-seed noise?
  State plainly. If it's a tie or a loss, say so. -->

### Memory cost

![Memory cost](memory_cost.png)

<!-- MEMORY_NARRATIVE_PLACEHOLDER: Eigen stores more (episodes + axioms). Quantify the overhead. -->

## Why this is a hard task for *any* memory scheme: the embedding-substrate problem

The deepest finding is about the **representation**, not the memory architecture.

The hidden rule is *arithmetic*: `prime → RED`, `÷5 → BLUE`, `else → GREEN`. But the agent
embeds a bare integer (e.g. `47`) with a **text** embedding model, then retrieves by cosine
similarity and runs PCA over those vectors. Text embeddings of number-tokens do **not** cluster
by primality or divisibility — `47` and `53` (both prime) are not necessarily close; `47` and
`48` (RED vs GREEN) may be closer.

So the retrieval substrate **cannot express the rule**. Neither RAG nor Eigen-memory can exploit
similarity that doesn't encode the relevant property. This is very likely the dominant reason
sophisticated memory does not pull away from the baseline here — the bottleneck is upstream of
the memory architecture entirely.

This reframes the whole result: it's less "PCA-based memory doesn't help" and more "you cannot
retrieve your way to a rule your representation can't see." A fairer test of eigen-memory would
use a task whose similarity structure lives in embedding space (e.g. semantic/textual rules).

## How this sits in the literature

This architecture recombines three well-established ideas (see [docs/PRIOR_ART.md](docs/PRIOR_ART.md)):

- **Surprise-gated storage** ← Prioritized Experience Replay (Schaul 2015), curiosity as
  prediction error (Pathak 2017).
- **Consolidating experience into natural-language rules** ← Generative Agents reflection
  (Park 2023), Reflexion (Shinn 2023), ExpeL insights (Zhao 2023).
- **PCA over representations to extract directions** ← Representation Engineering (Zou 2023).

"Eigen-memory" is this project's own coinage, not an established method. And the field-wide
pattern is that elaborate agent-memory schemes often only modestly beat a well-tuned RAG
baseline — consistent with what I see here.

## What I'd do next

- **Fix the substrate**: a task where embedding similarity actually encodes the rule, or embed
  engineered numeric features instead of raw text.
- **More seeds + a paired significance test** for a statistically powered claim.
- **Validate axiom quality before injection** — a wrong self-written rule poisons the context;
  measure how often crystallized axioms are actually correct.
- **Ablate the two surprise signals** (entropy vs NLL) to see which, if either, is load-bearing.
