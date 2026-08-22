# The Experiment That *Could* Validate Eigen-Memory

The experiment in this repo cannot demonstrate the hypothesis (see
[FINDINGS.md](FINDINGS.md)). This document specifies one that can. It is a design, not a
run — included to show that the negative result is understood, not hand-waved.

## What the original task gets wrong (recap)

| Flaw | Consequence | Requirement for a valid test |
|------|-------------|------------------------------|
| Arithmetic rule, text embeddings | Retrieval substrate is blind to the rule | Rule must live in **embedding space** |
| Inputs reused; ~no unseen items | Episodic lookup suffices; rules add nothing | **Held-out test set** the agent never stored |
| 100 trials, 1 small model | Effect < seed-to-seed noise | Enough seeds + a **significance test** |

The mechanism (surprise-gated writes, PCA over failures, axiom crystallization) is unchanged.
Only the *task and protocol* change.

> **Superseded on this point (see [THEORY.md](THEORY.md) §1–3).** The mechanism did *not* stay
> unchanged: a later theory review disproved the "PCA over failure embeddings" story and
> replaced it with contrastive PCA over retrieval *residuals* under a detectability gate. The
> task/protocol design below still stands; the mechanism sentence above does not.

## Design principle

Eigen-memory can only beat RAG when **a compressed general rule outperforms a bag of specific
episodes**. That happens precisely when:

1. The shared structure of failures is **visible in the embedding space** (so PCA finds a real
   direction, and retrieval finds genuinely related cases), and
2. Evaluation is on **held-out inputs** (so "store the exact past case" is not enough — you must
   have *generalized*).

If either is false, plain RAG ties or wins, as it did here.

## Proposed task: a hidden *semantic* rule over short texts

Replace integers with short natural-language items whose hidden label depends on a **semantic**
property that text embeddings represent well.

**Example rule family (one hidden rule per run, never revealed to the agent):**

> "Label `URGENT` if the message is about a time-sensitive problem (outage, deadline, safety);
> label `ROUTINE` if it is informational; label `SOCIAL` if it is interpersonal/non-work."

Items are 1–2 sentence messages, e.g.:
- `"The database has been down for 20 minutes and customers can't check out."` → URGENT
- `"Reminder: the Q3 report template moved to the shared drive."` → ROUTINE
- `"Thanks for covering my shift yesterday, I owe you a coffee."` → SOCIAL

Why this fixes the substrate: messages with the same hidden label **are** close in text-embedding
space (that is what sentence embeddings are good at). So cosine retrieval surfaces genuinely
similar past cases, and PCA over failure embeddings can find a real "urgency" direction to
crystallize into a rule.

**Generating the data:** programmatically template messages from labeled fragment banks (verbs,
objects, framings) so you control the rule and can produce disjoint train/test pools. Hold out
*templates and vocabulary*, not just instances, so the test set isn't paraphrase-memorizable.

## Protocol

- **Train phase:** N_train trials with feedback (the agent reads, predicts, learns). Memory is
  built here. (e.g. N_train = 150)
- **Test phase:** N_test **held-out** items, feedback **off**, memory **frozen** (no new writes).
  Accuracy on this phase is the headline metric. (e.g. N_test = 50)
- **Arms (unchanged):** Baseline / Control_RAG / Treatment_Eigen.
- **Seeds:** ≥ 10 independent seeds (new rule instance + data split per seed).
- **Stronger ablation — add a 4th arm, `RAG_large`:** plain RAG with a retrieval budget (top-k)
  large enough to roughly match the *token cost* of RAG+axioms. This controls for the confound
  "Eigen just adds more context"; a real win must beat RAG given equal context budget.

## Hypotheses (pre-registered, falsifiable)

- **H1 (primary):** On the held-out test phase, `Treatment_Eigen` mean accuracy >
  `Control_RAG`, paired across seeds, `p < 0.05` (paired t-test or Wilcoxon).
- **H2 (efficiency):** `Treatment_Eigen` reaches a target test accuracy with **fewer stored
  episodes** than `Control_RAG` (rules compress many episodes into one).
- **H3 (mechanism, not context size):** `Treatment_Eigen` > `RAG_large` at equal token budget.
  If this fails, any H1 win is just "more context," not the eigen-memory idea.

## Metrics

| Metric | Definition | Why |
|--------|------------|-----|
| **Held-out accuracy** | accuracy on frozen-memory test phase | the real generalization signal |
| **Tokens at equal-accuracy** | context tokens to hit a target | efficiency of compression |
| **Axiom precision** | fraction of crystallized axioms a judge rates as correct | guards against poisoning |
| **AUSC** | area under the surprise curve over training | learning *speed*, from the original EDD |

## Decision rule (how to read the outcome honestly)

- **H1 ✓ and H3 ✓** → the approach works *and* it's the mechanism, not the context size. Real win.
- **H1 ✓ but H3 ✗** → the "win" is just more context; eigen-memory adds no value. Report as such.
- **H1 ✗** → on a fair substrate with held-out generalization, compressing failures into rules
  still doesn't beat retrieval. A genuinely informative negative result (unlike the current one,
  which is uninterpretable).

## Cost / feasibility

~10 seeds × 4 arms × 200 trials × 2 LLM calls ≈ 16k calls. Infeasible on local CPU at the
current ~1 call/sec; feasible on a hosted endpoint (batchable) or a GPU. This is the main reason
it's specified rather than run here.

## Why this is the right next step

The current repo proves the mechanism *fires* and that the original task *can't* test it. This
design closes the loop: it isolates the one condition under which the hypothesis is decidable
(generalization on a substrate that can see the rule) and pre-commits to a decision rule that
can't be massaged into a win. Running it is the natural follow-up.
