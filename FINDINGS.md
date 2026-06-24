# Findings

> Honest write-up of what the experiment actually showed. Results are the mean of
> **2 seeds** (42, 7), 100 trials each, on a local `gemma3:4b` model.

## TL;DR

I built a surprise-gated "eigen-memory" agent, made the mechanism genuinely work (fixing
bugs that had silently turned the surprise signal into a constant), ran a controlled
multi-seed experiment — and found that the **experiment as designed cannot demonstrate the
hypothesis**. Eigen-memory does <!-- not / not clearly --> beat plain RAG here, but the more
important finding is *why the task can't show a difference even in principle*: the embedding
substrate is blind to the (arithmetic) rule, the task rewards memorization over
generalization, and the effect is below the noise floor. The deliverable is a rigorous
post-mortem, not a win.

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

## The real finding: the experiment cannot answer the question it asks

The most valuable result of this project is a **critique of its own experimental design**.
After making the mechanism actually work, I asked the harder question — *can this task even
demonstrate the hypothesis?* — and the answer is no. Three design flaws each independently
undermine the test:

### 1. The embedding substrate cannot represent the rule

The hidden rule is *arithmetic*: `prime → RED`, `÷5 → BLUE`, `else → GREEN`. But the agent
embeds a bare integer (e.g. `47`) with a **text** embedding model, then retrieves by cosine
similarity and runs PCA over those vectors. Text embeddings of number-tokens do **not** cluster
by primality or divisibility — `47` and `53` (both prime) are not necessarily close; `47` and
`48` (RED vs GREEN) may be closer.

So the retrieval substrate **cannot express the rule**. Neither RAG nor Eigen-memory can exploit
similarity that doesn't encode the relevant property. A null result is therefore uninterpretable:
it can't distinguish "eigen-memory is bad" from "the representation is blind to the rule."

### 2. The task rewards memorization, not generalization — which structurally favors RAG

Inputs are integers 1–100 with a fixed rule, run for 100 trials. By the end the agent has seen
**most of the input space**. Plain episodic RAG only needs to have stored the *exact* number
before — it never needs a rule. But the entire point of crystallizing an axiom ("primes are RED")
is to **generalize to unseen inputs**, and this task has almost none. The design gives the
rule-compression mechanism nothing to win with, and hands the win to lookup.

A valid test requires a **train/test split**: induce the rule on some inputs, then evaluate on
**held-out** inputs the agent has never stored. That is the only setting where "compress into a
generalizable rule" can beat "look up the nearest past episode."

### 3. The effect size is below the noise floor

A single 100-trial run is very noisy — batch accuracy swings from 20% to 90%, and two separate
"before" runs disagreed (Eigen 0.5 vs 0.9 final). Averaging seeds (done here) helps, but the
effect being hunted is plausibly smaller than seed-to-seed variance with one small 4B model.

### What a valid experiment would require

- A task whose similarity structure lives in **embedding space** (e.g. a hidden *semantic* rule
  over short texts), so retrieval can actually see the signal.
- A **train/test split** so generalization — not memorization — is what's measured.
- Enough seeds (and ideally a paired significance test) to clear the noise floor.

I deliberately did **not** rebuild the task — the critique itself is the result. Knowing *why*
an experiment can't answer its question, and what the valid version looks like, is the point.
The full design of the experiment that *could* validate this approach — a hidden semantic rule
over short texts, a held-out test phase with frozen memory, an equal-token-budget control arm,
and a pre-registered decision rule — is written up in
[docs/VALID_EXPERIMENT.md](docs/VALID_EXPERIMENT.md).

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
