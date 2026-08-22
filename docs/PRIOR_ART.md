# Prior Art & Positioning

How this project's "eigen-memory" agent relates to published work. The agent combines three
ideas, each with substantial prior art. The contribution here is the *combination* and the
*honest experiment*, not any single mechanism.

> Sourcing note: citations were hand-verified against the arXiv abstracts. Treat this as a
> literature map, not an exhaustive survey.

## A. Surprise / prediction-error–gated memory (the "what to remember" signal)

- **Titans** (Behrouz, Zhong & Mirrokni, 2024, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) —
  **the direct inspiration for this project's surprise mechanism.** A neural long-term memory
  trained *at test time*: each input's "momentary surprise" (the gradient of an
  associative-memory loss with respect to the memory module's parameters) is blended with a
  momentum-like "past surprise" and an adaptive forgetting gate; high-surprise inputs reshape
  the memory. The store is an MLP's weights — lossy, fixed-capacity, opaque. Scales past
  2M-token contexts, outperforming Transformers and modern linear RNNs on long-context tasks.
  Nuance: only the memory *writes* happen at test time — the projections and all three gates
  are meta-learned during pretraining, so Titans is a from-scratch architecture, not a
  bolt-on for a frozen model. (This project is, precisely, the inference-time retrofit of the
  same economics.)
- **Prioritized Experience Replay** (Schaul et al., 2015, [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)) —
  the RL ancestor: replays transitions in proportion to their TD-error (a prediction-error
  signal) instead of uniformly.
- **Curiosity-driven exploration** (Pathak et al., 2017, [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)) —
  curiosity as the prediction error of a learned forward model; the canonical
  "prediction-error = surprise" formulation.

**How this project differs:** it keeps Titans' economics (surprise decides what survives; the
memory is lossy) and swaps both the signal and the substrate. The signal is read directly from
an LLM's logits (entropy = perceptual surprise; NLL of the true label = predictive surprise) —
a training-free, black-box analogue of Titans' gradient surprise, usable with any hosted model.
The substrate is not parameters but **language**: surprising episodes are lossily compressed
into legible natural-language rules (section B for that lineage), which trades Titans'
free-of-charge memory application (a forward pass) for auditability and portability — see
THEORY.md and the C5 executor gate in C1_C3_TASK.md for what that trade costs. The
memory-conditional surprise probe (surprise measured with memory in context, so solved items
stop registering) is this project's analogue of Titans' forgetting gate.

## B. Consolidating experience into natural-language rules (the "memory consolidation" step)

LLM-agent systems that compress raw experience into higher-level, reusable text:

- **Generative Agents** (Park et al., 2023, [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)) —
  a "memory stream" of raw experiences plus a **reflection** mechanism that periodically
  synthesizes them into higher-level inferences. The closest analogue to this project's
  axiom "crystallization."
- **Reflexion** (Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) —
  the agent verbally reflects on task feedback and stores the reflection in episodic memory to
  improve on later trials. Self-correction from mistakes via language, no weight updates.
- **ExpeL** (Zhao et al., 2023, [arXiv:2308.10144](https://arxiv.org/abs/2308.10144)) —
  gathers experiences across training tasks and extracts **natural-language insights** that are
  retrieved at inference. Very close in spirit to "crystallize failures into rules."
- **Voyager** (Wang et al., 2023, [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) —
  an ever-growing **skill library** of executable code; consolidation into reusable, compositional
  behaviors rather than rules.

**How this project differs:** consolidation is *triggered* by a spectral signal (a PCA
component over surprise-vectors stabilizing) and *seeded* by contrastive failure/success
examples aligned to that component — rather than time-based reflection (Generative Agents) or
per-trial reflection (Reflexion). The triggering is the novel seam; the LLM introspection that
writes the rule is standard.

## C. Spectral / PCA methods on representations (the "eigen" part)

- **Representation Engineering** (Zou et al., 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) —
  extracts concept *directions* from LLM activations. Crucially, its reading method (LAT) runs
  PCA over **difference vectors of contrastive stimulus pairs**, never over raw activations —
  the differencing is what isolates the concept from dominant nuisance variance.
- **Contrastive PCA** (Abid, Zhang, Bagaria & Zou, 2018, *Nature Communications* 9:2134,
  [arXiv:1709.06716](https://arxiv.org/abs/1709.06716)) — finds directions of high variance in
  a target dataset *relative to a background dataset*, surfacing structure ordinary PCA misses.
- **The BBP phase transition** (Baik, Ben Arous & Péché, 2005, *Ann. Probab.* 33(5); real-valued
  analogue Baik & Silverstein 2006) — for spiked covariance models, the top sample eigenvector
  carries information about a planted direction only past a critical sample count; below it, it
  is asymptotically uncorrelated noise. The principled answer to *when* an eigen-direction is
  real enough to act on (implemented as a permutation-estimated edge).

**How this project differs — and an honest correction:** the original kernel applied PCA to
*raw input embeddings* of surprising items — skipping RepE's differencing step — and a theory
review showed that mechanism is blind to a sub-dominant attribute (failure-conditioning
mildly *suppresses* it; see [THEORY.md](THEORY.md) §1). The corrected kernel is LAT/cPCA
transported to retrieval: contrastive PCA over **retrieval residuals** (query − retrieved
exemplar), with failure residuals as target and success residuals as background, triggered
past the BBP detectability threshold. A literature check found no established method applying
PCA to retrieval residuals for error analysis or rule induction (adjacent work — "PCA-RAG,"
residual quantization — shares vocabulary but targets compression/indexing), so that framing
appears to be this project's contribution; LAT and cPCA are its defensible ancestors.

## D. Does "eigen-memory" already exist as a term?

The search did not surface an established, widely-used "eigen-memory" term matching this exact
architecture. There is adjacent work on PCA/eigendecomposition over memory or activations (e.g.
representation engineering above; classical Hopfield/associative-memory spectra), but not a
canonical "eigen-memory agent." Treat the name as this project's own coinage, not an established
method — and avoid implying otherwise.

## E. Do sophisticated memory schemes actually beat simple RAG?

This is the crux for the experiment. The honest, field-wide picture: gains from elaborate
agent-memory schemes over a well-tuned retrieval (RAG) baseline are **frequently modest and
task-dependent**, and strong simple baselines often close most of the gap. This project's own
result is a small, local data point in that broader pattern — see [FINDINGS.md](FINDINGS.md).

## One-line positioning for the README

> Titans compresses an agent's surprises into neural weights; this project compresses them into
> **legible rules** — surprise-gated storage (read from logits, not gradients), spectral
> detection of shared failure structure (contrastive PCA over retrieval residuals, gated by a
> random-matrix detectability edge), and LLM introspection that writes the rule — then runs
> controlled experiments to find exactly where that legible compression beats plain retrieval.
