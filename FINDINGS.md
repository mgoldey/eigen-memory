# Findings

> Honest write-up of what the experiments actually showed. The number-game and TREC sections
> are the mean of **2 seeds** (42, 7), 100 trials each, on a local `gemma3:4b` model; the
> later flip section below is 4 seeds, and the Rule-Shift work (5 seeds, `gemma4:12b`) lives
> in [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md).

## TL;DR

I built a surprise-gated "eigen-memory" agent, made the mechanism genuinely work (fixing
bugs that had silently turned the surprise signal into a constant), and ran a controlled
multi-seed experiment on **two different substrates**:

1. **Number-game** (arithmetic rule, *invisible* to text embeddings): the three arms are
   **statistically indistinguishable**. RAG barely beats no-memory at all, because the retrieval
   substrate is blind to an arithmetic rule.
2. **TREC** (question-type rule, *visible* to text embeddings): now **RAG clearly beats
   Baseline (80% vs 75%)** — proving the rig can detect a real memory benefit — but **eigen-memory
   does not beat plain RAG (75% vs 80%)**; the crystallized axioms add nothing over raw exemplars.

A third experiment — a purpose-built label-flip task run later with the corrected kernel — is
summarized in its own section below (spoiler: RAG won again, and the reason is a new gate:
the executor itself can't apply a rule well enough to beat copying — called C5 here).

Across both substrates, **eigen-memory never beats plain RAG.** The two failures fail *differently*,
and that's the finding: on the number-game the substrate hides the rule; on TREC the baseline is
already high (a ceiling effect) and a single retrieved example settles each question, so there is
nothing for rule-compression to win with. The deliverable is a rigorous two-substrate post-mortem
that isolates *why* the idea doesn't win — not a victory lap.

## Results (mean of 2 seeds, 100 trials each)

| Arm | Final-batch accuracy | Cumulative accuracy (all 100 trials) |
|-----|----------------------|--------------------------------------|
| Baseline (no memory) | 0.70 | 0.47 |
| Control_RAG (retrieval) | 0.60 | 0.46 |
| Treatment_Eigen (retrieval + axioms) | 0.60 | **0.55** |

![Learning curve](figures/learning_curve.png)

**Read the error bands, not the lines.** Treatment_Eigen's cumulative accuracy (0.55) edges out
RAG (0.46) and Baseline (0.47) by ~9 points — but the ±1 std bands of all three arms **overlap
almost completely**, and on the noisier final-batch metric Baseline is *highest*. With only 2
seeds and per-batch swings of 0.30–0.75, **the three arms are statistically indistinguishable.**
There is a *hint* that Eigen accumulated more correct answers, but it is not a defensible win —
and, as the next section argues, even a clean win here would be uninterpretable.

## What I had to fix before the experiment was even valid

The most important part of this project is *not* the architecture — it's that the original
experiment was measuring nothing. Two separate bugs each made the "surprise" signal a
**constant**, so memory was being gated on noise:

1. **Flat 2.0** — an `UnboundLocalError` when the true-label token was missing from the
   logprobs was silently caught and replaced with a hardcoded `2.0`. Every recorded surprise
   score was exactly `2.00`.
2. **Flat 7.0** — after fixing (1), a completion-style probe prompt (`"...the label:"`) was
   fed to a *chat* model, which replies with prose. The label was never the first token, so
   surprise collapsed to the missing-token fallback (`7.0`) for every item.

Fixed by constraining the probe to emit exactly one label word, so its logprob is a real,
varied prediction-error signal (`1.55, 4.62, 5.45, ...`).

**Running count of the constant/corrupted-signal bug class, used consistently across these
docs:** (1) flat 2.0 and (2) flat 7.0 here; (3) the prefix-match bug that flattened NLL for
2 of 3 classes, and (4) the chain-of-thought-as-axiom bug — both found in the flip-era review
and archived in [`results_prefix_bug/`](results_prefix_bug/). A fifth, the incomplete fix to
(4) that left trailing CoT in the stored axiom, was found in a 2026-08-10 audit — see
[docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) §6.

**Lesson:** a sophisticated-looking pipeline can produce authoritative numbers while
measuring a constant. The only way I caught it was instrumenting and *looking at the actual
surprise values*, not just the final accuracy. This is the engineering-judgment story.

## The result, honestly

No arm clearly learns the rule. Cumulative accuracy lands at ~0.47 (Baseline), ~0.46 (RAG), and
~0.55 (Eigen) — all hovering not far above the ~0.33 three-class chance line, none climbing
toward mastery. RAG barely improved on no-memory at all, which is itself a signal: retrieval
added almost nothing because the retrieved neighbors are not actually informative for an
arithmetic rule. Eigen's slight edge is within noise and rests on 2 seeds; I would not claim it
as real without the more powered experiment in [docs/VALID_EXPERIMENT.md](docs/VALID_EXPERIMENT.md).

### The agent talked itself into giving up — and that *is* the finding

The single most revealing artifact of the whole project is an axiom the agent crystallized about
its own failures:

> **RULE:** The model must always output one of the specified colors (RED, BLUE, or GREEN)
> without any interpretation or analysis of the input number. **It should simply pick one at
> random.**

Its reasoning (paraphrased from the `<thought>` block): the failure cases are the ones where the
model *tried to reason* about the number (primes, mappings, codes); the success cases are where
it just picked a color. So it concluded the optimal policy is to **stop reasoning and guess**.

On this task that conclusion is *locally correct* — because the embedding substrate hides the
rule, reasoning genuinely doesn't pay off, so the "overthinking" runs do worse by chance. The
agent correctly induced that **there is no learnable signal it can act on**, and rationalized
surrender. That is the embedding-substrate problem, narrated from inside the agent.

### Axiom quality is real but uneven

Across the run the kernel crystallized ~15–20 axioms per Treatment phase. Inspecting them:
some genuinely name the rule concepts (`prime`, `divisible`, `5`); others are mostly preamble
("Here's my analysis:") or, as above, advise guessing. So crystallization *can* surface the
right structure, but injection is noisy and unvalidated — a wrong or defeatist axiom poisons the
context. Validating axioms before injection is listed in next steps.

A later theory pass ([docs/THEORY.md](docs/THEORY.md)) explains the unevenness quantitatively:
the kernel crystallized every batch on ~10 vectors in 768 dimensions — far below the spectral
detectability threshold — so the "direction" behind each axiom was close to a random vector,
and the contrast examples handed to the introspection prompt were effectively arbitrary. An LLM
confabulates a confident rule from arbitrary examples; garbage axioms are the predicted
signature of crystallizing below that threshold, not a prompt-quality accident.

(The kernel has since been reworked to the corrected mechanism — contrastive PCA over
retrieval residuals with a detectability-gated trigger, THEORY.md §8. All results in this
document are from the **old** kernel and are kept as the record of what it produced.)

### Memory cost

![Memory cost](figures/memory_cost.png)

Eigen stores strictly more than RAG — every episode RAG keeps, **plus** the crystallized axioms
(~15–20 per run) and their eigenvectors. So Eigen pays a higher memory and token cost for, here,
no reliable accuracy gain — the worst quadrant of the cost/benefit plane on this task.

## The follow-up: TREC, a substrate that *can* see the rule

The number-game's central flaw (below) is that the embedding substrate is blind to an arithmetic
rule. The obvious test of that diagnosis: run the **same agent, unchanged**, on a task where the
hidden rule *is* a semantic property that text embeddings represent — and see whether retrieval
suddenly starts helping. If RAG beats Baseline on such a task, the rig is sound and the
number-game's null was a substrate problem, not a bug.

I wired in **TREC question classification** (behind `TASK=trec`; see
[docs/DATASETS.md](docs/DATASETS.md)): short questions, a genuine hidden rule (the *type* of answer
a question asks for), classified into a 3-class subset `{HUM, LOC, NUM}`. Same agent loop, same
surprise gating, same PCA kernel — only `src/dataset.py` changed.

### TREC results (mean of 2 seeds, 100 trials each)

| Arm | Final-batch accuracy |
|-----|----------------------|
| Baseline (no memory) | 0.75 |
| Control_RAG (retrieval) | **0.80** |
| Treatment_Eigen (retrieval + axioms) | 0.75 |

(Raw TREC numbers: `results/static/comparison_results.trec.json`. The number-game's are in
`results/static/comparison_results.json`, which the committed plots are rendered from.)

**Mechanism health:** clean run — **0 embedding failures**, surprise NLL genuinely varied
(batch averages 0.89 → 2.20), and the eigen layer fired for real (**19 axioms crystallized,
170 axiom-injections** into context). The eigen layer was not a silent no-op; it was working and
*still* didn't help.

Two things this shows, both more informative than the number-game null:

1. **The rig is sound — RAG beats Baseline when the rule is visible.** Unlike the number-game
   (where RAG 0.46 ≈ Baseline 0.47), on TREC **RAG clearly beats no-memory** at nearly every batch
   (final 0.80 vs 0.75; at batch 40, 0.95 vs 0.65). This directly confirms the *rule is visible
   in embedding space* condition (**C1**) from
   [docs/USE_CASES.md](docs/USE_CASES.md): when the rule lives in embedding space, retrieving similar
   past episodes genuinely helps. The experimental apparatus *can* detect a memory benefit.

2. **Eigen still doesn't beat RAG — and here we can see why.** Despite 170 axiom-injections, the
   Treatment arm tracks Baseline (0.75), *below* plain RAG (0.80). Two compounding reasons:
   - **Ceiling effect.** Baseline is already 0.75 zero-shot — a 4B model can largely classify TREC
     question-types unaided. That leaves only ~25 points of headroom, and the seed variance (±0.15)
     is wide enough to swallow the RAG→Eigen gap.
   - **Axioms are noisier than exemplars** — the *exemplars can't carry the task* condition
     (**C3**) fails. TREC is *single-exemplar-solvable*: one retrieved "Where is X? → LOC"
     example already settles a question. An abstract self-written
     rule is weaker context for a 4B model than a concrete neighbor, so crystallization adds cost
     without adding signal. The axiom-over-exemplar bet only pays off when one example is *not*
     enough — compositional rules, many-shot tasks — which TREC is not.

### Two substrates, one conclusion

| | Number-game | TREC |
|---|---|---|
| Rule visible in embeddings (**C1**)? | ✗ | ✓ |
| Does RAG beat Baseline? | No (0.46 vs 0.47) | **Yes (0.80 vs 0.75)** |
| Does Eigen beat RAG? | No | No (0.75 vs 0.80) |
| Why the effect can't show | substrate blind to rule | ceiling + one exemplar already settles it (**C3** fails) |

Eigen-memory beat plain RAG on **neither** substrate — but the two tasks fail different winning
conditions (the four in `USE_CASES.md`; C1 and C3 are the ones at issue here), and neither
satisfies all four. The TREC arm is the cleaner
result: it rules out "the rig can't detect any memory effect" (RAG *does* win there), leaving the
specific conclusion that **crystallizing failures into axioms doesn't beat retrieving raw exemplars
unless a single exemplar is insufficient.** That is a precise, defensible negative — and it points
exactly at where the idea *would* have to be tested to win (see next section).

## The third act: the purpose-built flip task, the executor gate (C5), and bugs three and four (2026-07-14/16)

The task the next-steps list below calls for **was subsequently built and run** — a label-flip
task engineered to sit in the one regime where rule-compression should win. The story arrived
in two acts, and the second is this document's thesis eating its own tail:

**The first multi-seed run was compromised.** A review pass found two more silent bugs: the
crystallizer stored the model's full chain-of-thought as the "axiom" and injected ~1.2k chars
of it into every Treatment context (sabotaging the arm under test), and the surprise probe's
token matching failed on multi-token labels, flattening NLL to a constant for 2 of 3 classes —
**the third instance of the constant-surprise bug class** this document exists to warn about.
Both were verified live before fixing (NLLs 7.0/0.01/7.0 → 4.57/0.01/11.55 after a one-line
prefix match). The compromised run is archived in `results_prefix_bug/`.

**The corrected run (4 seeds, temperature 0, health counters in every artifact):** held-out
Baseline 0.289 ± 0.048, Oracle 0.411 ± 0.103, RAG 0.600 ± 0.101, Eigen 0.617 ± 0.106 — but
the Eigen–RAG "difference" is one seed's single wrong-mapping axiom; on the other three seeds
**zero axioms crystallized and the arms produced identical predictions**. The hypothesis was not
supported.
Three replicated findings:

1. **The model can't apply the rule** (**C5**, 4/4 seeds): the Oracle arm — true rule pasted
   in context — scores below the
   nearest-neighbor label-copy ceiling on every seed (paired −0.178 ± 0.093). A 4B executor
   applies a rule worse than blind copying scores; rule-memory could not win regardless of
   axiom quality.
2. **Visible to the embeddings ⇒ visible to retrieval** (**C1 ⇒ ¬C3**): the guardrail originally
   measured copying with the wrong queries. Measured
   under protocol conditions, cross-split neighbors match on the rule attribute itself
   (polarity 0.73–0.89) — making a rule embedding-visible makes it retrieval-visible, so the
   "eigen window" was a measurement artifact. See [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md).
3. **The zero axioms are calibrated behavior:** a synthetic ROC of the actual gate
   (`gate_roc.py`) shows a 0.00 false-positive rate and full-gate firing only past ~8× the
   *noise edge* — the largest eigenvalue noise alone would produce, estimated by permutation.
   What binds is the check that the direction reproduces, not that edge. The flip task's
   residual failures are high-rank per-topic confusions — nothing rank-1 to find.

Meanwhile the positive mechanism claim was re-verified on the final code: 120 TREC trials →
exactly one axiom, and it is true (`results/static/trec_verify.42.json`). Full design and guardrail history:
[docs/C1_C3_TASK.md](docs/C1_C3_TASK.md) and [docs/BLOG_POST.md](docs/BLOG_POST.md); raw data
`results/flip/comparison_results.flip.<seed>.json` + aggregate.

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

- **Surprise-gated storage** ← **Titans** (Behrouz et al. 2024) — the direct inspiration:
  surprise-gated, lossy, test-time memory, compressed here into legible rules instead of MLP
  weights. Ancestors: Prioritized Experience Replay (Schaul 2015), curiosity as prediction
  error (Pathak 2017).
- **Consolidating experience into natural-language rules** ← Generative Agents reflection
  (Park 2023), Reflexion (Shinn 2023), ExpeL insights (Zhao 2023).
- **PCA over representations to extract directions** ← Representation Engineering (Zou 2023).

"Eigen-memory" is this project's own coinage, not an established method. And the field-wide
pattern is that elaborate agent-memory schemes often only modestly beat a well-tuned RAG
baseline — consistent with what I see here.

## What I'd do next

The first item on the old version of this list — *build a task where the rule is
embedding-visible (C1) but copying still fails (C3)* —
**was done**, and running it honestly proved such a static task cannot exist (making the rule
visible to the embeddings makes it visible to retrieval too — above).
The unified roadmap now lives in [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md); the short
version:

- **The Rule-Shift experiment** — break copying with *time* instead of geometry: the rule
  changes mid-run, stale exemplars keep retrieving perfectly and answering wrongly, the
  re-crystallized rule stays current. Pre-registered gates include a recency-weighted-RAG
  control arm (the baseline that could kill it) and a rank-1 failure axis by construction.
  *(Ran 2026-07-22/26: pilot seed fired end-to-end and scored 0.911 vs 0.522 for the kill
  arm — rerun 2026-08-11 on a fixed rule extractor, 0.922, axiom clean of chain-of-thought;
  the five-seed pre-registered endpoint missed — pooled Δ +0.078 vs a +0.10 bar, gate
  fired 1/5 seeds. The calibration read the shut seeds as signal-starved; the ungated
  ablation — run 2026-08-11 — overturned that on the signal-existence question: forced to
  crystallize with no gate, all 4 shut seeds wrote rules correct on both polarities. The
  signal was there and the live estimator missed it, with the caveat that the ablation's
  reconstructed failure signal is cleaner than live. See README "Act three" and
  docs/NEXT_EXPERIMENT.md §6–9.)*
- **A 60-item executor pre-test** (**RFμ**: rule vs copy vs conflicting contexts) to check a
  model can apply a rule at all before burning a run — gemma3:4b already failed this, so
  candidates start at 12B.
- **A sample-size-aware stability threshold** — the gate-ROC shows the fixed 0.95 cosine
  demands ~8× the noise edge; principled earlier detection means scaling the threshold to the
  expected estimator wobble at the current n, not lowering the edge.
- **Ablate the surprise gate itself** (gated vs store-everything, and entropy vs NLL) — the
  banner mechanism has never been isolated as a variable; note the corrected kernel already
  feeds crystallization *ungated* residuals by design.
