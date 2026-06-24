# Prior Art & Positioning

How this project's "eigen-memory" agent relates to published work. The agent combines three
ideas, each with substantial prior art. The contribution here is the *combination* and the
*honest experiment*, not any single mechanism.

> Sourcing note: papers below were surfaced by a multi-source literature search and
> cross-checked against the author's own knowledge of the field. They are well-known,
> canonical works. The search's adversarial-verification pass was interrupted, so treat the
> framing as a literature map rather than a peer-reviewed survey.

## A. Surprise / prediction-error–gated memory (the "what to remember" signal)

The idea of using prediction error to decide what is worth storing or replaying is well
established in reinforcement learning:

- **Prioritized Experience Replay** (Schaul et al., 2015, [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)) —
  replays transitions in proportion to their TD-error (a prediction-error signal) instead of
  uniformly. Directly analogous to this project's *predictive surprise* (NLL) gate on writes.
- **Curiosity-driven exploration** (Pathak et al., 2017, [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)) —
  formalizes curiosity as the prediction error of a learned forward model, used as intrinsic
  reward. The canonical "prediction-error = surprise" formulation this project echoes.

**How this project differs:** the surprise signal is read directly from an LLM's logits
(entropy of next-token distribution = perceptual surprise; NLL of the true label =
predictive surprise) rather than from a learned RL value/forward model. The gate decides
*storage*, not *replay priority*.

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
  extracts concept *directions* from population-level LLM activations (often via PCA over
  contrastive pairs) to read and steer high-level concepts. Establishes PCA-over-representations
  as a concept-extraction tool.

**How this project differs — and an honest caveat:** RepE applies PCA to *model activations*
(rich internal states). This project applies PCA to *embedding vectors of inputs* (here, single
integers). That is a much weaker substrate — see the embedding-substrate caveat in
[FINDINGS.md](../FINDINGS.md). The "eigen" framing should be read as *PCA-as-a-trigger*, not
as eigendecomposition carrying the semantic load.

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
result is a small, local data point in that broader pattern — see [FINDINGS.md](../FINDINGS.md).

## One-line positioning for the README

> This project recombines three established ideas — prediction-error–gated storage (à la
> Prioritized Experience Replay), reflection-style consolidation into natural-language rules
> (à la Generative Agents / ExpeL), and PCA-based concept extraction (à la Representation
> Engineering) — into a single local agent, and runs a controlled experiment to test whether
> the combination beats plain retrieval.
