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

First outing of the rebuilt kernel, on the two old tasks, with a falsifiable prediction each.
(These two were live smoke runs — the telemetry below is quoted from their run logs; the flip
experiment in the next section is the archived one, `comparison_results.flip.json`.)

- **Number-game, 80 trials** (arithmetic rule, invisible to text embeddings): the gate should
  stay closed. It did — **zero axioms**, where the old kernel emitted 15–20.
- **TREC, 120 trials** (question-type rule, embedding-visible): a real-but-weak axis might
  clear the edge. The telemetry shows λ1 hovering at the permutation edge across four checks —
  held (unstable), held (below edge), **crystallized** (detectable and stable), then the
  novelty gate marked the axis consumed. One axiom, strength 1.03:

> *"If the question asks for a quantity or measurement, predict 'NUM'; if the question asks for
> a location, person, or named entity, predict 'LOC'."*

One hundred twenty trials, lossily compressed to twenty-seven words — and the words are a
**correct rule about the task's hidden structure**, written by a 4B model. That is the artifact
this architecture exists to produce.

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

## The showdown, and the number that may have decided it in advance

100 training messages, then 45 held-out with memory frozen. Pre-registered win condition:
Eigen > max(RAG, Baseline). Plus an **Oracle arm** — no memory, the true flip-table pasted
into context — as the headroom ceiling:

| Arm | Held-out | requests | reports |
|---|---:|---:|---:|
| Baseline | 0.222 | 0.30 | 0.16 |
| Oracle_Rule (true rule in context) | 0.467 | 0.35 | 0.56 |
| **Control_RAG** | **0.533** | 0.60 | 0.48 |
| Treatment_Eigen | 0.356 | 0.55 | 0.20 |

**RAG won; H1 not supported.** But look at the instrument panel rather than the scoreboard:

- The guardrail's m (0.567) predicted RAG's score (0.533) before the run. The regime map works.
- The compressor found the planted axis, live: one axiom, strength 1.12, after two correctly
  held checks — *"if the input describes a completed action or a status update … DEFER;
  otherwise ESCALATE."* Completed-action-vs-needs-attention **is** the request/report polarity.
  Right axis.
- Wrong *function*: the 4B model collapsed the per-topic flip into one global mapping (and
  dropped the third label). The wrong half poisoned the report cells: 0.20 (n=25), below blind
  copying.
- And the most consequential number — though on one seed and 45 held-out items it is
  **suggestive rather than significant** (each proportion carries roughly ±0.15 at 95%):
  **Oracle (0.467) < m (0.567).** With the true rule in its context, the model applies it *worse
  than mindless neighbor-copying scores*. If that holds up under more seeds, it means the
  experiment was decided before training started — when the executor's rule-following capacity
  sits below the copy ceiling, rule-memory cannot win **regardless of axiom quality** — and only
  the Oracle arm could reveal it. Call it C5, a fifth pre-registerable gate: *the model must
  apply a rule better than it can copy an exemplar.* Replicating it across seeds is the first
  item on the to-do list; the effect direction, at least, is exactly what the capability-tax
  story predicts.

## What Titans gets for free, and what legibility costs

Here is the deepest thing the failed showdown taught me, and it's a direct consequence of the
substrate swap. **Titans never pays C5.** When surprise is compressed into weights, applying
the memory is just the forward pass — rule-following is free, baked in. When surprise is
compressed into *language*, application becomes a capability tax collected at inference: the
model has to read the rule, bind it to the current input, and execute the conditional. A 4B
model can't — it copies better than it follows, so RAG is the structurally *correct* memory for
it, not just a strong baseline.

So the honest scope of "lossy compressive surprise-memory → rules" is now sharp:

- **The compression side works and is demonstrably safe**: surprise-gated capture, spectral
  detection with a noise floor, one true rule per real axis, silence otherwise. That's the
  novel seam — Titans' surprise economics with a rate-gated, *legible* codec, where prior
  verbal-consolidation systems (Reflexion, Generative Agents, ExpeL) reflect on schedules and
  keep everything.
- **The decompression side is the frontier**: legibility pays off exactly when the executor
  clears C5 and coverage is sparse (m near chance) — and it buys what weights never can:
  audit, portability across models, human override, and rules that survive a model swap. The
  next experiments write themselves: same rig, same gates, a stronger model reading (and
  writing) the rules — cheap, because the detectability gate makes crystallization rare by
  design; the flip task at the ≥5 seeds its own protocol pre-registers, to promote C5 from
  suggestive to significant; and an ablation of the surprise gate itself (gated vs
  store-everything), since the banner mechanism has never been isolated as a variable.

I set out to advertise a mechanism and ended up with a measured boundary: **compress into
weights when your model is small; compress into sentences when your model can read — and
either way, let surprise decide what's worth keeping, and let the eigenvalue earn the write.**

---

*The repo: a theory doc where every claim is an executable test, the planted-world simulations,
the guardrail scripts, the corrected kernel, and the full experiment. Lineage:
[Titans](https://arxiv.org/abs/2501.00663) (surprise-gated test-time memory — the inspiration),
[RepE](https://arxiv.org/abs/2310.01405) (PCA over contrastive differences),
[cPCA](https://arxiv.org/abs/1709.06716) (target-vs-background spectra),
[BBP 2005](https://arxiv.org/abs/math/0403022) (the detectability edge), and the
verbal-consolidation line (Reflexion, Generative Agents, ExpeL) this hopes to give a
rate-distortion theory.*
