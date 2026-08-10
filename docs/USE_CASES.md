# Where Eigen-Memory Would Actually Win

> **Status note.** Written after the number-game result, before the flip and Rule-Shift
> experiments. Its four winning conditions still frame the analysis, but the repo now reports
> four experiments and a fifth condition (the executor must be able to *apply* a rule — C5).
> Current summary: [../README.md](../README.md).

The experiment this document was written against is a negative result: on the
number-classification task, compressing failures into "axioms" did not beat plain retrieval
(see [../FINDINGS.md](../FINDINGS.md)). This document is the constructive flip side — *given why it
lost, where would it win?* The answer follows directly from the failure analysis.

## The principle

Eigen-memory beats RAG only when **a few compressed rules beat many stored episodes.**

RAG stores and retrieves individual past cases. Eigen-memory additionally distills the *shared
structure* of failures into a handful of reusable rules. That extra step only pays off when the
problem genuinely has a small set of rules that explain a large set of cases — and when the
agent must apply them to situations it has never stored. Where every case is effectively its own
isolated fact, there is nothing to compress, and RAG wins on simplicity.

## The four winning conditions

For eigen-memory to beat RAG, all four must hold at once:

- **C1 — The rule is visible in embedding space.** Cases governed by the same rule must land
  near each other under the embedding model, so PCA can find a real direction and retrieval can
  surface genuinely related cases.
- **C2 — Success requires generalizing to unseen inputs.** Evaluation must hit inputs the agent
  never stored, so "look up the nearest past episode" is insufficient and an *abstracted* rule
  is what carries.
- **C3 — Exemplars can't carry the task.** Retrieving stored cases must be insufficient: one
  exemplar must not settle the answer (operationally: nearest-neighbor label-copy accuracy `m`
  near chance — see [THEORY.md](THEORY.md) §7), or coverage must be sparse or costly to build
  (long traces, long-tail inputs, noisy duplicates), so compressing many episodes into one rule
  is a real gain rather than a flourish. These are the same fact at two scales: copy accuracy
  is a coverage phenomenon — it rises as the episode buffer densifies — so rule-compression is
  a hedge against sparse coverage.
- **C4 — The governing rules are few, stable, and reusable.** A handful of axioms must cover a
  long tail of cases, and they must not churn so fast that a crystallized rule is stale before
  it is used.

## Scorecard

The number-game is included as the worked **✗** example: it satisfies only C4, and failing any
single condition is enough to sink the approach.

| Domain | C1 visible in embed space | C2 needs generalization | C3 episodes costly/redundant | C4 few stable rules | Verdict |
|--------|:---:|:---:|:---:|:---:|---------|
| **Number-game (this repo)** | ✗ | ✗ | ✗ | ✓ | **Loses** — arithmetic rule invisible to text embeddings; inputs reused; trivial episodes |
| **Coding / agent self-correction** | ✓ | ✓ | ✓ | ✓ | **Strong win** |
| **Personalized assistant preferences** | ✓ | ✓ | partial | ✓✓ | **Strong win** |
| Support / ticket triage | ✓ | ✓ | ✓ | partial | Likely win |
| Anomaly / policy induction | partial | ✓ | ✓ | partial | Mixed (rules drift) |
| Game agent w/ hidden mechanics | depends | ✓ | partial | ✓ | Depends entirely on C1 |

### Why the number-game scores ✗✗✗✓

It is worth being concrete, because it makes every condition tangible:

- **C1 ✗** — the rule is arithmetic (prime / ÷5), but inputs are bare integers embedded by a
  *text* model. `47` and `53` (both prime) need not be neighbors; `47` and `48` (different
  labels) may be closer. The substrate cannot see the rule, so neither PCA nor retrieval can
  exploit it.
- **C2 ✗** — inputs are integers 1–100 reused across 100 trials; by the end the agent has seen
  almost the whole input space. Episodic lookup suffices; no generalization is demanded.
- **C3 ✗** — each episode is a single integer. Storing them all is trivially cheap, so there is
  no compression dividend.
- **C4 ✓** — there *are* only three stable rules. This is the one thing the task got right, and
  it is why the idea looked plausible — but one ✓ out of four loses.

This is the value of the worked example: it shows that "the task has clean underlying rules"
(C4) is necessary but nowhere near sufficient.

## Deep dive 1 — Coding / agent self-correction

**The setup.** An agentic coding assistant repeatedly fails in a codebase or toolchain in
*patterned* ways: "this repo wants `pathlib`, not `os.path`", "tests here mock the HTTP client,
not `httpx` directly", "this API returns snake_case, the prior one camelCase." Each failure is a
long trace; the underlying mistakes are a small, recurring set.

**Why all four conditions hold:**

- **C1 ✓** — failures are described in natural language / code, which sentence and code
  embeddings represent well. Similar mistakes genuinely cluster.
- **C2 ✓** — the agent must apply the lesson to *new* files and tasks it has never seen, not
  re-run an identical one. An abstracted rule ("mock the client, not httpx") transfers; a stored
  episode of one specific file does not.
- **C3 ✓** — episodes are long, expensive traces (full reasoning + diffs + tool calls). Keeping
  and re-retrieving all of them is costly and noisy; one crystallized rule replaces dozens.
- **C4 ✓** — a codebase's conventions are a small, stable set that recurs across hundreds of
  edits.

**Honest caveat (straight from this repo's data).** In our run, crystallized axioms were *uneven*
— some named the real rule, others were filler or, memorably, advised the agent to *stop
reasoning and guess*. A wrong axiom injected into context actively poisons it. So in a coding
setting this approach **requires axiom validation before injection** (e.g. only keep a rule that
demonstrably fixes a held-out failure). This is exactly the "validate axioms" item in FINDINGS'
next steps — the failure here tells us it is load-bearing, not optional.

**Prior art.** This is the territory of Reflexion (verbal self-reflection stored as memory) and
ExpeL (extracting natural-language insights across tasks) — see [PRIOR_ART.md](PRIOR_ART.md). The
eigen-memory twist is *when* to consolidate: trigger on a stabilizing failure direction rather
than on every trial or a fixed schedule.

## Deep dive 2 — Personalized assistant preferences

**The setup.** A long-lived assistant learns a user's stable preferences from their corrections:
"don't be verbose," "always use metric," "I write Python, not JavaScript," "prefer bullet points
over prose," "never suggest YAML." (The original schema's own placeholder for `axiom_content` was
literally *"User hates YAML"* — preferences are the use case this architecture was quietly built
for.)

**Why this is the textbook fit — C4 is unusually strong (✓✓):**

- **C1 ✓** — preferences and the requests they apply to are natural-language; embeddings cluster
  "tone" requests, "format" requests, "stack" requests.
- **C2 ✓** — the payoff is applying a learned preference to a *brand-new* request the user never
  made before. That is generalization by definition; no stored episode matches a novel ask.
- **C3 partial** — individual preference signals are small (cheaper episodes than coding traces),
  so the compression saving is more about *signal-to-noise* than storage cost: one clean axiom
  beats twenty scattered, partially-contradictory correction episodes cluttering the context.
- **C4 ✓✓** — preferences are the *most* few-stable-reusable kind of rule there is. A user has a
  handful of them; they rarely change; they apply everywhere. This is the strongest possible C4.

**Why compression beats episode-lookup here specifically.** With RAG, the assistant retrieves a
few past corrections and hopes the current request is similar enough. With a crystallized axiom
("user prefers terse, metric, Python"), the rule applies even when the new request is *topically*
unrelated to any past correction — a regime where nearest-episode retrieval misses entirely.

**Caveat.** Same as above: a wrongly-induced preference is worse than none ("user hates emojis"
crystallized from two coincidental cases). Preference axioms need a confidence threshold and easy
user override.

## Briefly: the rest

- **Support / ticket triage** — hidden routing/resolution rules over *text* tickets. C1–C3 hold;
  C4 is *partial* because resolution playbooks drift as products change. Likely a win with
  periodic re-consolidation.
- **Anomaly / policy induction (fraud, moderation)** — compress many flagged cases into a few
  policy directions. C2/C3 strong, but C1 is shaky (adversaries deliberately evade the embedding
  structure) and C4 drifts. Mixed.
- **Game agent with hidden mechanics** — induce rules from play. Fit hinges entirely on C1: does
  the game state embed such that same-rule states are neighbors? If state is symbolic/arithmetic
  (like our number-game), it fails for the same reason. If state is rich/perceptual, it can work.

## The unifying insight

Eigen-memory is a bet that **the world has few rules and many instances** — and that those rules
are visible in the representation you embed with. It wins where that bet is true: coding
conventions, user preferences, support playbooks. It loses, as it did here, exactly where the bet
is false — where every instance is effectively its own isolated fact (the number-game), or where
the rule is real but invisible to the embedding (also the number-game). Knowing *which* world
you're in, before reaching for the mechanism, is the whole game.

See also: [FINDINGS.md](../FINDINGS.md) (why it lost here), [THEORY.md](THEORY.md) (the corrected
mechanism and the two-statistic regime map), [VALID_EXPERIMENT.md](VALID_EXPERIMENT.md)
(how to test it fairly), [PRIOR_ART.md](PRIOR_ART.md) (the lineage).
