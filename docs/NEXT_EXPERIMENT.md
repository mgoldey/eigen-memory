# What Would It Take to Actually Demonstrate Success?

> Output of a red-team brainstorm (experimental-design persona) run 2026-07-16, after the
> corrected multi-seed flip rerun showed zero axioms crystallizing and Treatment_Eigen
> degenerating to exactly Control_RAG. Lightly edited. This document is the pre-registration
> basis for the next experiment.

## 0. The organizing insight

The three negative results, read together, are close to a theorem about **static tasks**:

> In a single-session task with a **fixed** rule and a **growing** buffer, making the
> rule-attribute embedding-visible (C1) makes it retrieval-visible, so nearest-neighbor copy
> accuracy m rises toward the attribute's cross-split recoverability — and whatever attribute
> generalizes across the train/test surface split is *exactly* the attribute that dominates
> cross-split neighbor structure. **C1 ⇒ ¬C3, asymptotically in buffer density.**

The data proves this twice: the (buggy) guardrail tuned m ≈ 0.567 *within-split*, and the
corrected cross-split measurement showed polarity-match at 0.73–0.89 — the vocabulary split
deleted topic similarity and left polarity as the dominant surviving axis. No static task
geometry escapes this. **Every winnable accuracy claim must break copying with a mechanism
orthogonal to embedding geometry.** There are exactly three:

1. **Time** — exemplars go stale; the abstract rule (re-crystallized) stays current.
2. **Capacity** — exemplars get evicted; one sentence survives any budget.
3. **Executor economics** — exemplars cost tokens/attention the model can't afford; a rule is cheap.

Plus one claim family needing no accuracy win: **mechanism-quality claims** (the gate is
safe; rules are correct when they fire; rules are auditable).

## 1. The success ladder (weakest-but-honest → strongest)

- **S0 — "The detectability gate is calibrated"**: synthetic sweep, no LLM. Plant a rank-1
  contrast at SNR ∈ {0, 0.5×, 1×, 2×, 4× the noise edge}, n ∈ {50, 100, 200}, 20 replicates;
  report a gate-ROC. The specificity half already exists (0 axioms on number-game and
  corrected flip); the sensitivity arm must fire ≥90% at 2× the edge. *Winnable now; ~2–4 h
  of numpy.* Almost nobody in the memory-agent literature reports their consolidation
  trigger's false-positive rate.
- **S1 — "When crystallization fires, the rule is correct"**: score every fired axiom against
  the planted rule with a rubric; report precision with a Wilson interval. Currently n≈2
  firings — only strong bundled with S5 (which generates ~1 firing/seed).
- **S2 — "Rules match exemplars at a fraction of the token budget"**: needs an
  accuracy-parity result first (TREC was 0.75 vs 0.80 — not parity). Parked.
- **S3 — "Rules survive buffer capacity constraints"**: evaluate held-out with the buffer
  capped at N ∈ {150, 50, 15, 5, 0}; RAG decays toward Baseline, the rule stays flat. Cheap
  (~1–2 h on a trained store), one compelling plot — but presupposes a fired axiom and a
  C5-passing executor. Act 2 of the recommended play.
- **S4 — "Rules survive an encoder swap (stale index)"**: property statement, small figure,
  not a headline. (The *executor*-swap version is wrong — exemplars are text and port fine.)
- **S5 — "Rules beat exemplars when the rule shifts and exemplars go stale"**: the strongest
  winnable accuracy claim; the recommended play (§5). The attack to pre-empt: recency-weighted
  RAG — which must be a pre-registered control arm. The structural counter: recency-weighting
  shrinks the effective buffer to the post-shift window, re-creating sparse coverage — while
  the rule, crystallized from the same few post-shift failures, covers the whole space.
- **S6 — "Eigen beats RAG in a static fair fight": declared UNWINNABLE on this rig** (the §0
  theorem plus C5 jointly foreclose it). Retiring it explicitly is reviewer-credibility
  currency.

## 2. C1∧C3 paradox — escape routes scored

| Family | Copying fails held-out? | Rank-1 failure axis? | Verdict |
|---|---|---|---|
| XOR/compositional (the flip task) | ✗ (polarity dominates cross-split) | ✗ (per-topic table = high-rank) | Tried; fails both. Do not iterate. |
| K rules over K attributes | ✓ (drive m → chance with K=3–4) | ✗ (rank ≈ K) | Needs a hierarchical kernel — mechanism change. Skip. |
| Low-density / tiny buffer | ✓ (coverage) | converges to XOR case as density grows | Only as the S3 capacity demo. |
| **Continual shift / stale exemplars** | **✓ (neighbor is perfect, label is out of date — non-geometric)** | **✓ if the shift is partial (one polarity row flips → contrast = the polarity axis)** | **The one. C1 and C3 decoupled by time.** |
| Exemplar-unparseable, rule-easy | ✓ | n/a | Token-economics claim; reduces to "compression helps." Narrative only. |

Pre-registered caveat for the shift design: the staleness contrast may be null in *residual*
space (both classes retrieve same-polarity neighbors), so declare the estimator order up
front — primary: contrastive PCA on failure-vs-success **query embeddings** in the post-shift
window; secondary: residuals. Written down before running, or it's estimator-shopping.

## 3. The C5 gate — executor qualification (RFμ)

Never again burn a full run on an unqualified executor. The **Rule-Following
Microbenchmark**: 60 items (20 is too few — SE ≈ 0.11), paired, McNemar test, ~45 min per
candidate:

1. Fresh rule table, never reused in the main run.
2. **R**: true rule pasted + item → label (test table and prose formats, take the max).
3. **C**: k=5 labeled neighbors at the main run's expected polarity-match rate → the copy
   ceiling *as the model realizes it*, not the geometric m.
4. **RC**: rule + stale exemplars that contradict it — the condition the shift experiment
   actually lives in.
5. **Gate: acc(R) − acc(C) ≥ +0.10 AND acc(RC) ≥ acc(R) − 0.10, McNemar p < 0.05.**

Candidate order (stop at first pass): gemma3:12b → qwen3.5:9b → qwen3.5:27b → gemma3:27b.
If nothing local passes, that is itself the publishable boundary: "below ~X B parameters,
exemplar-copying dominates rule-following, and no rule-memory architecture can pay."

> **Amendment (2026-07-16, after the first two RFμ runs — `run_rfmu.py`,
> `rfmu.<model>.json`).** The C condition's premise broke on contact: with a global-polarity
> rule, 5 correctly-labeled exemplars are enough for a strong model to *induce* the rule, so
> C measures few-shot induction, not nearest-neighbor copying — gemma4-8B scored C = 0.933
> and qwen3.5:9b C = 0.917, far above the geometric copy ceiling (~0.8 polarity match), which
> makes the R − C ≥ +0.10 margin gate nearly unpassable for any model strong enough to matter.
> Both candidates: R (prose rule) = 1.000. RC (rule + stale exemplars): gemma4-8B 0.767
> (**fails** — trusts stale exemplars ~25% of the time), qwen3.5:9b 0.900 (passes at the
> boundary). Two consequences, adopted before any main-run spend: (1) the Rule-Shift-relevant
> qualification is **R ≥ 0.90 AND RC ≥ R − 0.10 AND McNemar(R > C-stale) p < .05**, where
> **C-stale** (exemplars only, labeled under the outdated rule, no hint they're stale)
> replaces C as the realized copy arm — that is what post-shift retrieval actually serves;
> (2) rule-format sensitivity is real and consistent (prose ≥ table by 0.12–0.23 on both
> models) — good news, since crystallized axioms are prose sentences.
>
> **Executor designated (2026-07-16, post-Ollama-upgrade).** gemma4:12b passes the amended
> gate decisively: R (prose) = 0.983, **RC = 0.983** (zero stale-exemplar seduction — RC ≡ R,
> vs gemma4-8B's −0.233 and qwen3.5:9b's −0.100), **C-stale = 0.450** with McNemar
> p < 0.0001 (R > CS). The CS number doubles as an empirical validation of the Rule-Shift
> premise: a copy arm served stale exemplars realizes ~0.45 while a rule-following executor
> realizes ~0.98 — that 0.53 gap is the headroom the experiment plays in. One sharp caveat
> for the axiom-injection template: gemma4:12b scores 0.517 on *table*-formatted rules vs
> 0.983 on prose — rules must be injected as prose sentences. G0 is satisfied; next stop G1
> (generator probe) and the pilot seed.

**Split the roles**: crystallizer = biggest model (fires rarely — quality is nearly free);
executor = cheapest RFμ-passing model (thousands of calls); surprise probe can stay small
(needs varied NLL, not competence). "Write with a big model rarely, read with a small model
always" is independently the best economic argument for rules over exemplars.

## 4. Win axes ranked for the portfolio

1. **S5 rule-shift** — the only accuracy win consistent with §0.
2. **S3 capacity scaling** — cheapest striking visual, same trained store.
3. **Auditability veto demo** — inject a known-wrong axiom → accuracy drops; veto it →
   recovery; contrast with "find the one bad exemplar among 150." Two hours; label it a
   demonstration, not an evaluation. Makes "legible beats latent" concrete vs Titans.
4. Token cost at parity / encoder-swap — paragraphs, not headlines.

## 5. The recommended play: the Rule-Shift experiment

**Task**: flip generator reused with two changes — (1) rule is **global in polarity** (one
mapping, no per-topic table → rank-1 failure axis by construction); (2) at trial 100 the
**request row flips**, reports unchanged. Disjoint train/test vocabulary kept. n_test = 90
per seed (SE ≈ 0.05; 450 paired observations across 5 seeds resolves Δ ≈ 0.08).

**Arms (6)**: Baseline · Control_RAG (k=5) · **Recency_RAG (the arm that can kill it)** ·
Treatment_Eigen (**axiom replaces exemplars when injected** — pre-registered injection
policy) · Oracle (true post-shift rule) · measured copy ceiling (analysis).

**Protocol**: 100 pre-shift trials (parity expected — negative control) → shift → 60
post-shift adaptation trials → freeze → 90 held-out post-shift items. 5 seeds, temperature 0.

**Gates in order** (each aborts before the next spends money):
- **G0**: executor passes RFμ (~2–4 h across candidates).
- **G1**: probe-AUC(polarity) ≥ 0.8 on the new generator (30 min, no LLM).
- **G2**: corrected C3 — cross-split post-shift copy accuracy against the *actual frozen
  buffer* ≤ 0.45, measured not assumed (30 min).
- **G3**: detectability fires on the pilot seed (primary estimator: query-embedding cPCA,
  post-shift window; p < .05 on 3 consecutive checks) before the 5-seed spend.
- **G4**: every fired axiom scored against the planted post-shift rule *before* unblinding
  accuracy.

> **Amendment (2026-07-17, pre-pilot, from the G1/G2 measurement — guardrail_shift.py on 5
> seeds).** G1 passes everywhere (probe-AUC 0.980–0.989). Strict G2 **fails on every seed**
> (aggregate copy_acc 0.500–0.644 vs ≤ 0.45) — and the per-row split shows the ≤ 0.45 bar
> was an arithmetic oversight in this pre-registration, not a property of the data: only the
> request row shifts, so report queries stay copyable forever (0.67–0.98 measured) while
> request-copying craters exactly as intended (0.00–0.35). An aggregate below 0.45 is
> unreachable *by construction* while retrieval works at all. Amended G2, adopted before any
> pilot spend: **G2a** copy_acc on requests (the shifted row) ≤ 0.45 [passes 5/5], and
> **G2b** the Recency_RAG policy ceiling (top-5, newest wins) ≤ executor R − 0.10 = 0.883,
> so a correct axiom retains decision-rule headroom over the kill arm [passes 5/5; ceiling
> 0.689–0.778]. The strict number is kept in every guardrail.shift.<seed>.json as g2_strict.
>
> **Amendment (2026-07-17, pre-pilot, by analysis).** G3's pre-registered estimator
> (query-embedding **cPCA**) is provably blind to the failure structure this design creates.
> Post-shift, failures concentrate on one polarity and successes on the other — a
> **location** difference between the groups. Covariance-style contrasts (cPCA included)
> subtract per-group means and cannot see it; second-moment and pooled-centering variants
> cancel it too, because the polarity clusters sit symmetrically about the global mean.
> Estimator amended to a **two-sample mean contrast** (statistic = ‖mean(fail) −
> mean(succ)‖², direction = the mean-difference axis) under the *identical* permutation-edge
> / stability / novelty gates. Two supporting kernel settings for the non-stationary stream:
> a sliding **window of 60** trials (without forgetting, 100 pre-shift records swamp the 60
> post-shift ones — the legible analogue of Titans' forgetting gate), and a sample-size-aware
> **stability threshold of 0.5** (the gate-ROC showed cos 0.95 is unreachable below ~8× edge;
> 0.5 is still p ≈ 2×10⁻⁴ against the random-direction null in the r = 50 working space).
> G3's firing criterion stays as pre-registered: detectable on 3 consecutive checks.

**Decision rule**: primary endpoint Eigen vs **Recency_RAG**, held-out post-shift,
item-level paired across 5 seeds; win = Δ ≥ +0.10, one-sided p < 0.05, AND Eigen >
Control_RAG, AND Oracle > copy ceiling. Secondary: Eigen within 0.05 of Oracle.

**Budget**: RFμ 2–4 h + guardrails 1 h + pilot ~1.5 h + 5 seeds × 6 arms × ~250 calls
≈ 7,500 calls ≈ 5–6 h on a 12B. Fits in a day.

**Named failure modes** (all detectable before the full spend): Recency_RAG wins at pilot →
pivot to S3+S0+auditability; G3 never fires → report "the gate correctly identifies this
failure structure as not spectrally compressible" + a clearly-labeled exploratory ungated
arm; no executor passes RFμ → the C5 boundary paper; crystallizer writes the pre-shift rule
→ caught at G4; executor follows stale exemplars → caught at RFμ condition RC.

**The honest fallback if everything loses**: "Across four purpose-built regimes —
embedding-blind rules, exemplar-solvable rules, static XOR rules, and shifting rules —
natural-language rule-compression never beat the best exemplar policy, and we can say
precisely why in each: substrate (C1), copy ceiling (C3), executor economics (C5), and
recency-weighting. The detectability gate posted a 0% false-positive rate throughout, and
every fired axiom was scored for correctness. This is a map of where Titans-style surprise
economics does and does not transfer to legible memory." That is a strong portfolio artifact
regardless of the coin flip.

**If only three things get done**: (1) RFμ + G0; (2) the S0 synthetic gate-ROC — it converts
"0 axioms fired" from an anticlimax into a calibration result; (3) the Rule-Shift pilot seed.
