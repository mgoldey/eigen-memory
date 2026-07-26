# Surprise, Compressed: A Lossy Agent Memory That Writes Its Own Rules

*Titans showed that a memory which decides what to keep by measuring its own surprise can beat
attention at scale — by compressing experience into neural weights. But Titans is an
architecture you have to pretrain. This project asks whether a frozen, black-box model can get
the same economics as a pure inference-time wrapper — with the compression target changed from
weights to sentences.*

---

## The idea: change what surprise compresses into

Google's **Titans** (Behrouz, Zhong & Mirrokni, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663))
is the cleanest recent statement of a very old principle: memory should be *lossy, and surprise
should decide what survives*. Titans bolts a neural long-term memory onto attention and trains
it **at test time**: each incoming token produces a "momentary surprise" (the gradient of an
associative-memory loss with respect to the memory's parameters), blended with a momentum-like
"past surprise" and an adaptive forgetting gate. High-surprise inputs reshape the memory;
unsurprising ones fade. The store is a small MLP — fixed capacity, continuously overwritten,
fundamentally **opaque**. It works: Titans outscales Transformers past 2M-token contexts.

One nuance matters for everything that follows: only the memory *writes* happen at test time.
The ability to write is trained in — the key/value projections and all three gates (surprise
scale, past-surprise decay, forgetting) are meta-learned end-to-end during pretraining. Titans
is an architecture you train from scratch, not a bolt-on; a frozen GPT- or Gemma-class model
can never have it.

This project — a local rig with a 4B model, pgvector, and no training loop — asks whether a
**frozen, black-box model** can get the same economics as a pure inference-time retrofit, and
swaps the substrate while it's at it. The bet: an agent's surprising experiences don't have to
be compressed into weights nobody can read. They can be compressed into **rules** — short
natural-language axioms the agent writes about its own failures, auditable by a human,
portable across models, injectable into any context window.

The pipeline is a two-stage lossy compressor:

1. **The gate (what to keep).** Surprise is read directly from the model's logits — the NLL of
   the true label, and next-token entropy — a training-free, black-box analogue of Titans'
   gradient signal. Unsurprising experience is discarded at the door; only genuine prediction
   errors enter the episodic buffer. Crucially, surprise is measured *with memory in context*,
   so an experience the memory already handles stops being surprising — the analogue of Titans'
   forgetting gate, and the property that makes the whole thing self-limiting. (In the live
   runs, mean surprise fell 8.6 → 4.1 → 1.3 across the first three batches as memory filled.)
2. **The crystallizer (what to abstract).** When enough failures share one geometric axis, the
   agent compresses the whole cluster into a single sentence: an eigendecomposition over its
   failures finds the axis, and the LLM introspects on contrast examples straddling it to write
   the rule. Hundreds of episodes → one direction → ~25 words.

One design note, for honesty about what gates what: surprise gates only the *episodic* store.
The crystallizer deliberately consumes **ungated** residuals from every trial, keyed on
correct/incorrect — because the covariance contrast needs unbiased samples of both outcomes.
Surprise decides what the agent can *retrieve*; the eigenvalue decides what it *abstracts*. (An
ablation isolating the surprise gate itself — gated vs store-everything — is on the to-do list;
it has not yet been tested as a variable.)

The endgame compression ratio is the pitch: in the live TREC run below, **120 trials of
experience distilled into exactly one rule** — the agent's entire semantic memory fit in a
tweet, and the rule was *true*.

| | Titans | This project |
|---|---|---|
| Deployment | new architecture, pretrained from scratch (writes at test time, capability trained in) | wrapper around any frozen model — no architecture change, no training |
| Surprise signal | gradient of associative loss w.r.t. memory params | NLL + entropy read from logits (black-box) |
| Memory substrate | MLP weights, updated at test time | episodic vectors + natural-language axioms |
| Compression | continuous, opaque, fixed-capacity | discrete, **legible**, rate-gated |
| Forgetting | data-dependent weight decay | memory-conditional surprise (solved items stop registering) |
| Applies memory by | forward pass (free) | the model *reading a rule* (a capability tax — see the ending) |

The interesting engineering question — and the one this post is really about — is the one lossy
compression always poses: **how do you keep the compressor from compressing noise?** A lossy
codec that hallucinates detail produces artifacts. A lossy memory that hallucinates structure
produces *confident false rules*, injected into every future decision. The answer turned out to
come from random matrix theory, and getting there required admitting the first version of the
mechanism was provably wrong.

## What the first version got wrong (it was backwards)

The original crystallizer ran PCA over the raw embeddings of surprising experiences, on the
story that *failures concentrate where the hidden attribute matters, so the attribute has high
variance among failures*. A review took that sentence apart, and a planted-attribute simulation
(embeddings with dominant topic structure, a sub-dominant binary attribute B, labels flipping
on B within each topic) confirmed the demolition:

- Failure-conditioning **suppresses** the attribute (variance ratio 0.78×): items strongly
  expressing B sit near same-B neighbors and rarely fail. The mechanism selected exactly the
  items where the signal was weakest.
- B stayed buried at the same rank (8th) behind the topic axes in failure-PCA as in global PCA.
- The flip-label itself is linearly undecodable (0.52 probe vs 0.95 for B) — an XOR. Any theory
  where "PCA finds the rule" was dead on arrival.

Worse, the old compressor had no rate control at all: it crystallized *every batch*, on ~10
vectors in 768 dimensions, and ranked axioms for injection by cosine to an eigenvector — whose
sign is arbitrary. The result was the known failure mode of ungated lossy compression: 15–20
confabulated axioms per run, including one that advised the agent to *stop reasoning and guess
at random*.

## The corrected compressor: residuals, contrast, and a noise floor

Three changes, each with the numbers that justify it (every claim is an executable test in the
repo, `tests/test_kernel_theory.py`):

**Compress retrieval residuals, not embeddings.** Every trial, the agent retrieves its nearest
stored neighbor — and that pair is a contrastive pair, exactly what Representation Engineering
(Zou et al. 2023) runs PCA over. The residual δ = query − neighbor cancels the dominant topic
axes (that's what made it the neighbor): *retrieval is a matched filter, and what survives in
the residual is precisely what retrieval failed to control*. Conditioned on failure, the hidden
attribute becomes a rank-one spike: B jumped from rank 8 to rank 1 at cos 0.91 in simulation.
For robustness, use *success* residuals as a contrastive-PCA background (Abid & Zou 2018) —
they share retrieval's leakage but carry no failure spike; in the hardest regime this rescued
alignment from 0.39 to 0.86. Both datasets come free from the agent's own logs.

**Gate crystallization on detectability, not a schedule.** Estimating a spiked covariance's top
eigenvector has a phase transition (Baik–Ben Arous–Péché 2005): below a critical sample count
the estimate is asymptotically uncorrelated with the truth. Measured: alignment 0.18 at n=10
residuals, 0.72 at n=100, 0.89 at n=400. This is the rate-distortion knob lossy memory needs:
**crystallize only when the top eigenvalue clears a permutation-estimated noise edge, the
direction is stable across checks, and it hasn't been crystallized before.** Below the edge,
the "structure" is noise, the contrast examples handed to the LLM are arbitrary — and an LLM
will confidently compress arbitrary examples into a rule. The old garbage axioms weren't a
prompt problem; they were the predicted artifact of compressing below the noise floor.

**Divide the labor.** The spectrum can't solve an XOR and doesn't need to: it finds *where to
look* (the axis, and the maximally-contrastive examples along it); the LLM finds *the function*
(the rule). Axiom injection at inference is ranked by |projection| of the query onto each
axiom's axis — sign-invariant, unlike the old cosine ranking.

## Live: the compressor refuses to compress noise

First outing of the rebuilt kernel, on the two old tasks, with a falsifiable prediction each:

- **Number-game, 80 trials** (arithmetic rule, invisible to text embeddings): the gate should
  stay closed. It did — **zero axioms**, where the old kernel emitted 15–20.
- **TREC, 120 trials** (question-type rule, embedding-visible): a real axis should clear the
  edge. It did — exactly one axiom. Re-verified end-to-end on the final, fully-debugged code
  (`run_trec_verify.py`, archived in `trec_verify.42.json`): one axiom, strength 1.13, zero
  missing-token probes, and the rule is true:

> *"Questions requiring numerical answers should be labeled NUM, and questions requiring
> location or descriptive answers should be labeled LOC."*

One hundred twenty trials, lossily compressed to twenty words — and the words are a **correct
rule about the task's hidden structure**, written by a 4B model. That is the artifact this
architecture exists to produce.

And the refusal side is now measured, not asserted. A synthetic ROC drives the *actual*
kernel over planted rank-1 contrasts at controlled multiples of its own noise edge
(`gate_roc.py`): false-positive rate **0.00** at pure noise (the permutation edge alone leaks
the predicted ~5%; the stability check mops it to zero), detection at 1× the edge, full-gate
firing from ~8×. The binding constraint isn't the eigenvalue edge at all — it's **stability**:
under realistic noise, the estimated direction only reproduces across checks at |cos| > 0.95
once the spike is far above the floor. The gate buys its zero false positives with a large
detection margin. Remember that number; it's about to matter.

## Where rule-compression should beat retrieval — building the test

Compression only pays where a few rules explain many cases *and lookup can't*. Two pre-run
statistics locate that regime: **probe-AUC(B)** (is the hidden attribute in the embedding at
all?) and **m** (copy your nearest stored neighbor's label — how often are you right, at your
actual buffer size?). Probe at chance → nothing works (the number-game). m high → copying
suffices, rules are redundant (TREC: one exemplar settles a question). Probe high *and* m near
chance → **the window where compressed rules are the only thing that generalizes.**

No standard dataset sits in that window, so I built one: workplace messages where the routing
label **flips on request-vs-report polarity within each topic** — retrieval finds same-topic
neighbors, which carry the opposite label with probability 1−m; copying pays exactly m,
majority-voting can't exceed it, and held-out items share zero surface vocabulary with
training. Getting the generator *into* the window took four failed designs (m = 0.80, 0.86,
0.71, 0.74) before one passed (probe 0.947, m 0.567): polarity as a 2–3-word shared-head-word
marker ("not yet handled" / "fully handled") inside a long compositional neutral shell.
Speech-act is a strong sentence-embedding feature; the trick is the probe-vs-variance
asymmetry — shrink the attribute's share of pairwise *distance* while a supervised probe still
recovers it.

## The showdown, in two acts

**Act one: the run I almost published.** 100 training messages, 45 held-out with memory
frozen, four seeds, four arms — including an **Oracle arm** (no memory, the true flip-table
pasted into context) as the headroom ceiling. The first multi-seed run produced a seductive
story: RAG won, but the compressor had "found the planted polarity axis live" and written a
correct-axis rule, and the Oracle sat below the copy ceiling. I had the narrative half-drafted.

Then a review pass over the experiment code found two more silent bugs — in a project whose
founding lesson was already "three bugs made surprise a constant." **Bug four:** the
crystallizer stored the model's entire chain-of-thought (~1.2k characters of `<thought>`
rambling) as the "axiom" and injected it verbatim into every Treatment context — sabotaging
the very arm under test. **Bug five:** the surprise probe required the whole label inside one
token, and the tokenizer splits ESCALATE into `ES`-something — so for two of three classes,
"surprise" had quietly become a constant *again* (instance number three of the same bug
class). Verified live before fixing: NLLs of 7.0/0.01/7.0 became 4.57/0.01/11.55 after a
one-line prefix match. The compromised run is archived in `results_prefix_bug/`, because the
archaeology *is* the project.

**Act two: the corrected run** (4 seeds, temperature 0, health counters in every artifact —
missing-token rate 0–2%, parse fallbacks ~1%, buffers 47–58/100 showing the write gate
actually gating):

| Arm | Held-out (mean ± std over 4 seeds) |
|---|---:|
| Baseline | 0.289 ± 0.048 |
| Oracle_Rule (true rule in context) | 0.411 ± 0.103 |
| **Control_RAG** | **0.600 ± 0.101** |
| Treatment_Eigen | 0.617 ± 0.106 |

Eigen "edges out" RAG by 0.017 — but read the instrument panel, not the scoreboard: on three
of four seeds **zero axioms crystallized and the two arms produced literally identical
predictions**; the entire difference is one seed whose single axiom named the right axis with
an *inverted* mapping (+0.067, well inside noise). H1 not supported, and the seductive Act-one
axiom story is gone — it was an artifact of the corrupted regime.

Three findings survived the cleanup, and they're better than the story they replaced:

- **C5, now replicated 4/4.** With the *true* rule in context, the executor scores below the
  honest nearest-neighbor copy ceiling on every seed (paired Oracle − ceiling =
  −0.178 ± 0.093). A 4B model applies a rule worse than it copies — so rule-memory could not
  have won here **regardless of axiom quality**. That's the fifth pre-registerable gate:
  *the model must apply a rule better than it can copy an exemplar.*
- **C1 ⇒ ¬C3: the static-task paradox.** The corrected guardrail (measuring retrieval the way
  the protocol actually retrieves — held-out queries, disjoint vocabulary) revealed the "eigen
  window" was a measurement artifact: cross-split neighbors match on *polarity* (0.73–0.89),
  not topic (0.38–0.47), because whatever attribute generalizes across the split dominates
  cross-split similarity — and that's exactly the attribute the rule depends on. Making the
  rule embedding-visible made it retrieval-visible. On a static task, the window closes as you
  open it.
- **The silence was calibrated.** With retrieval already controlling polarity, the residual
  failures scatter across per-topic confusions — high-rank structure, no rank-1 spike. The
  gate-ROC says the compound gate needs ~8× the noise floor to fire; the flip residuals never
  came close. Zero axioms is what a well-calibrated lossy compressor *does* with no compressible
  signal.

## What Titans gets for free, and what legibility costs

Here is the deepest thing the failed showdown taught me, and it's a direct consequence of the
substrate swap. **Titans never pays C5.** When surprise is compressed into weights, applying
the memory is just the forward pass — rule-following is free, baked in. When surprise is
compressed into *language*, application becomes a capability tax collected at inference: the
model has to read the rule, bind it to the current input, and execute the conditional. A 4B
model can't — it copies better than it follows, so RAG is the structurally *correct* memory for
it, not just a strong baseline.

So the honest scope of "lossy compressive surprise-memory → rules" is now sharp:

- **The compression side works, and is calibrated, not just asserted**: surprise-gated
  capture, spectral detection with a measured ROC (false-positive rate 0.00; fires past ~8×
  the noise floor), one true rule per real axis (TREC, re-verified on the final code), silence
  otherwise. That's the novel seam — Titans' surprise economics with a rate-gated, *legible*
  codec, where prior verbal-consolidation systems (Reflexion, Generative Agents, ExpeL)
  reflect on schedules and keep everything.
- **The decompression side is the frontier, and static tasks can't reach it.** C5 says the
  executor must apply a rule better than it copies (a 4B model doesn't, 4/4 seeds); the
  C1 ⇒ ¬C3 paradox says that on a fixed-rule task, making the rule visible to the memory makes
  it visible to retrieval too. The next experiment therefore breaks copying with *time*
  instead of geometry: a **Rule-Shift** design where the rule changes mid-run, stale exemplars
  keep retrieving perfectly and answering wrongly, and the re-crystallized rule stays current —
  gated by an executor microbenchmark (rule-following vs copying vs conflicting contexts), a
  recency-weighted-RAG control arm pre-registered as the baseline that could kill it, and a
  sample-size-aware stability threshold. Full pre-registration in the repo
  (`docs/NEXT_EXPERIMENT.md`). Legibility still buys what weights never can: audit,
  portability across models, human override — and now the experiment that could prove it pays.

## Act three: breaking copying with time (pilot, one seed)

That experiment has now run its pilot, and it did the thing.

The setup, briefly. First the executor tax got paid: a 60-item microbenchmark (RFμ) sweeping
candidate models found that gemma4:12b follows a pasted prose rule at 0.983 **even with five
contradicting stale exemplars in context** (zero seduction — smaller models lost 10–23 points
to the same trap), while a copy arm served stale exemplars realizes 0.450. That 0.53 gap,
measured before the experiment ran, is the headroom the design plays in. One sharp detail with
a moral: the same model scores 0.517 when the identical rule is formatted as a *table* —
rules must be injected as prose, which is conveniently what a crystallizer writes. Then the
task: one **global** polarity rule (rank-1 failure axis by construction), the request row
flips at trial 100, feedback labels are always era-correct — so the buffer ends up holding
100 confidently-labeled stale episodes plus 60 current ones, which is exactly what "policy
changed, memory didn't" looks like in production.

Two pre-registration amendments got forced *before* the pilot spend, both by arithmetic
rather than by peeking: the aggregate copy-ceiling gate was unachievable by construction
(only the request row shifts, so report queries stay copyable forever — the gate now binds on
the shifted row), and the pre-registered cPCA detector is provably blind to this failure
structure (post-shift failures concentrate on one polarity, successes on the other — a
*location* difference, which covariance contrasts cancel). The amended detector — a
two-sample mean contrast under the identical permutation-edge/stability/novelty gates — got
the same treatment as the original: a synthetic ROC at the pilot's exact configuration.
False-positive rate at pure noise 0.00–0.05; fires at 1.00 from *half* the noise edge upward.

And then the pipeline earned its keep — twice, because the first pilot surfaced **bug six**:
the crystallizer's token budget ran out inside its `<thought>` block and 1.4k characters of
truncated chain-of-thought were stored and injected as the "axiom." The G4 gate (score every
fired axiom against the planted rule *before* unblinding accuracy) caught it, in a project
whose founding lesson is that this bug class recurs. The fixed crystallizer — retry for a bare
`RULE:` line, never store scaffolding — produced a genuinely legible rule and scored *higher*:

| Arm (held-out, post-shift labels, disjoint vocab) | Accuracy |
|---|---:|
| Baseline (no memory) | 0.033 |
| Recency_RAG (newest-first + staleness hint — the kill arm) | 0.522 |
| Control_RAG | 0.556 |
| **Treatment_Eigen (one crystallized rule, exemplars retired)** | **0.911** |
| Oracle (true post-shift rule pasted) | 0.967 |

The detector stayed silent through 100 pre-shift trials (one marginal flicker, correctly
rejected by the streak rule), watched batch surprise spike 0.30 → 9.78 NLL at the shift, fired
on three consecutive checks, and crystallized exactly one axiom:

> *"If the status is 'resolved' or 'already through review', label as ESCALATE; if the status
> is 'pending' or 'awaiting review', label as DEFER (unless it is a specific 'still awaiting
> review' case which is FILE)."*

Both polarity clusters map to the correct post-shift labels. And the parenthetical is a
**stale fragment of the pre-shift rule** that a human auditor can see and strike — the
auditability pitch writing its own demo. (It names a training-vocabulary marker, so it never
fired on the disjoint held-out set; requests scored 1.00.)

The paranoia pass, since a result this clean invites it: Eigen beats the kill arm by +0.389
with 39-vs-4 discordant pairs (McNemar p = 1.6e-8); the axiom was the *entire* context on all
90 test items, and an empty context scores 0.033, so the content carries the effect; and a
copy policy with **perfect** staleness filtering — retrieval restricted to only the 60
post-shift episodes, stronger than any realizable recency weighting — ceilings at 0.778,
well below 0.911. No exemplar policy over this buffer reaches the treatment number.

The honest scope: this is **one seed and one crystallization event**, on a task built to be
winnable (rank-1, embedding-visible — the C1 ⇒ ¬C3 lesson applied in reverse). The
crystallized rule is extensional — it enumerates markers rather than naming the
request-vs-report concept — and the ablation that would isolate the spectral gate's value
over a dumb "summarize recent failures every N batches" trigger hasn't run. The five-seed
pre-registered decision is next. But the mechanism's full loop — mistakes → gated detection →
one legible sentence → exemplars retired → near-oracle accuracy — has now happened outside a
thought experiment.

### The replication: the bar not cleared

Four more seeds ran, and the pre-registered endpoint **missed**. Pooled over 450 paired
held-out items: Eigen 0.658 vs the recency kill-arm 0.580 — Δ = +0.078 against the +0.10
bar. The direction is real (89-vs-54 discordant pairs, p = 0.002), but I set an effect-size
bar precisely so a significant-but-small pooled number couldn't be dressed up as a win, and
it did its job. Verdict: miss.

The decomposition is unusually clean, because the architecture leaves no partial credit.
The gate fired on **one seed in five**. On that seed, treatment beat the kill arm by +0.389
with an auditable prose rule. On the four gate-shut seeds, zero axioms crystallized — and
treatment's predictions were *item-for-item identical* to plain RAG's (verified, all 90 per
seed). No axiom, no effect, no hidden channel. Everything the mechanism earned, it earned
on the seed where it fired.

Why didn't it fire? The telemetry says the gate wasn't refusing noise — it was refusing
*evidence*. Across the shut seeds, the failure-contrast direction reproduced check after
check (the stability flag lit on 6 of 7 checks) while its magnitude hovered at 0.6–1.1×
a detection edge defined as the **max of 20 permutation draws**, crossed **3 consecutive
times** — two thresholds I never derived from any stated error budget. A persistent
near-miss scores exactly zero under that rule. The fix isn't to loosen anything by feel:
it's a gate whose one threshold is calibrated against a measured noise null at a stated
false-fire budget — accumulate per-check permutation p-values into decayed log-evidence,
fire when the total clears the level that noise reaches only 5% of the time.

So I built exactly that gate and calibrated it, threshold tuned on synthetic noise only —
and **the calibration cancelled its own re-run**. Two findings. First, the honest
threshold is much higher than independence would suggest (the noise null's evidence
quantile runs ~2× the independent-checks prediction), because consecutive detection
windows share five-sixths of their residuals — even noise looks "direction-stable" under
that much overlap, which quietly demotes the strongest-looking evidence the shut seeds
had. Second, at that budget the evidence-accumulating gate fires no more often than the
crude streak rule anywhere on the grid. And the sweep's ratio column locates the shut
seeds precisely: detection under either rule needs the statistic sustained at ~1.05× the
noise edge; pure noise averages 0.87×; the shut seeds lived at 0.81–0.85×. They weren't
threshold-starved. They were **signal-starved** — no gate that honors a false-fire budget
fires there, and a gate that would have is a gate that compresses noise. The mechanism's
one-in-five fire rate wasn't a bug in the gate; it was the gate telling the truth about
the embedding stream it was watching.

So act three ends the way act two did: the headline number lost to the bar, and the
autopsy is worth more than the number. Conditional on detection, compression beats
copying by a wide, legible margin; detection at real-world SNR is the bottleneck — and
the bottleneck turns out to be the *featurization*, not the threshold. What the residual
stream carries on most seeds simply doesn't separate failure from success strongly enough
to compress. The next levers are an ungated-trigger ablation (if a scheduled crystallizer
extracts a correct rule from those same windows, the signal was there and the estimator
missed it; if it writes garbage, the calibration called it right) and a residual
representation that actually carries the contrast.

## What this models in the wild

The structure being simulated — *repeated decisions with after-the-fact feedback, a policy
that changes without announcement, and a memory of past cases that silently flips from asset
to liability* — is not exotic:

- **Ticket triage and support routing.** An org changes its escalation policy; every
  historical ticket in the retrieval index encodes the old routing. A RAG triage bot copies
  stale precedent indefinitely — Control_RAG at 0.556 *is* that bot. The fix this mechanism
  proposes: detect the post-change error cluster, distill "billing requests now go to the
  platform team," show a human the sentence.
- **Content moderation.** Policies update constantly; precedent-based labeling is
  stale-exemplar copying; appeals and overturns are the feedback stream. This is also where
  legibility stops being aesthetic: you can diff, audit, and veto a written rule — the
  stale-clause catch above is the governance story regulators actually ask for.
- **Fraud and compliance decisioning.** Thresholds move with regulation and adversaries;
  crucially, only some segments shift while others stay stable — the partial-shift design
  (requests flip, reports don't) models exactly that, and errors concentrating on one segment
  is the mean-contrast signature the detector fires on.
- **Coding agents with persistent memory** — the self-referential one. A team migrates an
  API; the agent's memory is full of old-style examples; CI failures are the feedback.
  Crystallizing "use X, not Y, as of version Z" into a memory file is literally what
  CLAUDE.md-style agent memory does; this project is a controlled study of when that beats
  pasting retrieved snippets. The economics compound here: one rule distilled once by a large
  model, executed forever by a small one, versus five exemplars shipped with every call.
- **Medical coding and claims.** Code sets update on a schedule, rejections are feedback,
  and auditability is a legal requirement, not a preference.

What the model deliberately simplifies, so the transfer claim stays honest: real shifts are
gradual and overlapping rather than a clean flip at trial 100; real feedback is delayed,
noisy, and partial rather than instant gold labels; and real failure axes are not guaranteed
to be embedding-visible — C1 was engineered true here, and the C1 ⇒ ¬C3 result is proof that
some failure structures can't be caught this way at all.

## The third wedge: annotator disagreement (a hypothesis, not a result)

Every experiment above used clean gold labels. Real feedback streams don't have those — they
have annotators, and annotators disagree. That turns out to be an argument *for* this
architecture, and a sharper one than it first looks, because disagreement attacks exemplar
memory at exactly the point where rules are immune.

Copying inherits label noise at full strength. A RAG arm that copies its nearest neighbor's
label inherits that one annotator's judgment on that one item — nearest-neighbor error
compounds label noise roughly one-for-one, forever, on every retrieval. The crystallizer is a
different kind of estimator: it pools dozens of episodes into one contrast, so idiosyncratic
labels wash out and it fits the *majority policy* — and once a correct rule is crystallized,
inference is noise-free from then on. The rule doesn't care that 15% of the buffer is
mislabeled. Disagreement therefore drives a wedge between the copy ceiling and the rule
ceiling that *grows with the disagreement rate*.

The reason this deserves its own section: the wedge works on **static** tasks. C1 ⇒ ¬C3
closed the static regime because a retrieved neighbor always carries the right label — but a
noisy neighbor doesn't, and label noise degrades copying without touching the embedding
geometry the detector uses. So the project now has three candidate wedges between rules and
retrieval: **geometry** (tried — the window closes as you open it), **time** (tried — the
pilot win above), and **noise** (untried). Each is a different answer to the same question:
what breaks the guarantee that your nearest stored neighbor knows the answer?

The gate has a principled role under disagreement, and half of it is already calibrated.
*Unstructured* disagreement — annotators randomly inconsistent — produces no stable
fail/success axis, and both gate-ROCs show the detector refuses structureless noise
(false-positive rate ≤ 0.05). *Structured* disagreement — two annotator camps applying
different implicit policies, correlated with anything embedding-visible — is precisely a
detectable location contrast. The mechanism would compress the camps' split into a sentence,
converting silent label noise into a visible policy question a team can adjudicate. The
pilot's stale clause is a preview of the failure mode: fed contradictory evidence, the
crystallizer wrote a hedged exception clause — legible enough to spot and resolve. And
annotator *turnover* is literally the Rule-Shift design: a new annotator with a different
implicit policy is a rule shift, with labels that are era-correct in their eyes.

The honest caveats, stated before any experiment runs: noisy feedback also degrades the
machinery's *inputs* — `was_correct` becomes unreliable, which blurs the fail/success
contrast and lowers the effective SNR (the mean-contrast ROC's sensitivity at half the noise
edge suggests margin; it was not measured under label noise). And "fits the majority policy"
cuts both ways — it locks in the majority camp against a possibly-legitimate minority
reading, which is a governance feature only if a human actually reviews the rule.

It's cheaply testable on the existing harness: flip p% of feedback labels, sweep p, and
measure where the copy ceiling crosses below rule accuracy and whether crystallization
precision survives. The pre-registerable claim: *there is a disagreement rate above which
compressed rules beat copying even with no shift at all* — compression as denoising. That
experiment is queued behind the ungated-trigger ablation the replication's miss motivated.

I set out to advertise a mechanism and ended up with a measured boundary — and then, on the
far side of it, a first live win: **compress into weights when your model is small; compress
into sentences when your model can read; break ties with time, because stale memories retrieve
perfectly and answer wrongly — and either way, let surprise decide what's worth keeping, and
let the eigenvalue earn the write.**

---

*The repo: a theory doc where every claim is an executable test, the planted-world simulations,
the guardrail scripts, the corrected kernel, and the full experiment. Lineage:
[Titans](https://arxiv.org/abs/2501.00663) (surprise-gated test-time memory — the inspiration),
[RepE](https://arxiv.org/abs/2310.01405) (PCA over contrastive differences),
[cPCA](https://arxiv.org/abs/1709.06716) (target-vs-background spectra),
[BBP 2005](https://arxiv.org/abs/math/0403022) (the detectability edge), and the
verbal-consolidation line (Reflexion, Generative Agents, ExpeL) this hopes to give a
rate-distortion theory.*
