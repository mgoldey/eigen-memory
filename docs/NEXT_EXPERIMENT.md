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
> `results/calibration/rfmu.<model>.json`).** The C condition's premise broke on contact: with a global-polarity
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
> 0.689–0.778]. The strict number is kept in every results/shift/guardrail.shift.<seed>.json as g2_strict.
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

## 6. Pilot result and threats-to-validity (2026-07-22, seed 42)

Held-out post-shift, frozen memory, disjoint vocab: Baseline 0.033 · Oracle_Post 0.967 ·
Control_RAG 0.556 · Recency_RAG 0.522 · **Treatment_Eigen 0.911** (requests 1.00, reports
0.84). Primary endpoint Δ vs Recency_RAG = **+0.389** (bar +0.10), McNemar p = 1.6e-8
(39-vs-4 discordants); vs Control_RAG p = 9.7e-9. Secondary endpoint (within 0.05 of
Oracle): 0.056 — just misses. G3 fired exactly as pre-registered (a marginal pre-shift
detection was rejected by the streak rule when λ dipped back under the edge; then three
consecutive post-shift detections, stable direction, one crystallization; surprise spiked
0.30 → 9.78 NLL at the shift boundary — this pair is read from the run log, which is
gitignored, so it is reported as live-run telemetry rather than a committed number).
Artifacts: results/shift/comparison_results.shift.42.json,
results/shift/derisk.shift.42.json, results/calibration/gate_roc_mean.json, results/shift/guardrail.shift.<seed>.json ×5.

**G4 caught a real mechanism bug on the first attempt** (archived:
results/shift/comparison_results.shift.42.cotbug.json): the crystallizer's 400-token budget ran out
inside its <thought> block and 1.4k chars of truncated CoT were stored and injected —
scoring 0.856, causally effective but not the claimed mechanism. Fix: retry for a bare
RULE line, never store scaffolding.

> **Correction (2026-08-10, found in a later audit).** That fix was incomplete, and the
> 0.911 artifact below is affected. The retry only covered replies with **no** `RULE:` at
> all; the extraction itself used `rpartition("RULE:")[2]`, which keeps the entire *suffix*
> rather than the rule's line. The stored seed-42 axiom is therefore the legible rule
> **followed by ~250 chars of trailing CoT**, truncated mid-sentence — visible in
> `results/shift/comparison_results.shift.42.json`. So the 0.911 run still injected CoT
> residue, in smaller quantity than the 0.856 cotbug run (~250 vs ~1.4k chars) and with the
> correct rule stated first, but the "clean prose rule only" claim did not hold. The
> extraction now takes the first line after `RULE:`, with a regression test built from this
> artifact's exact reply shape
> (`tests/test_kernel_consolidation.py::test_only_the_rule_line_is_stored_when_cot_trails_it`).
> **The 0.911 number stands as measured but is not a clean test of the stated mechanism; a
> rerun on the fixed extractor is required before it can be cited as one.** The pre-registered
> five-seed verdict (miss, +0.078) is unaffected in direction — seed 42 was the only firing
> seed, so a rerun moves the one number the pooled result leans on.

The rerun crystallized a genuinely legible prose rule
— "resolved / already through review → ESCALATE; pending / awaiting review → DEFER (unless
'still awaiting review', which is FILE)" — and scored **higher** (0.911). G4 scoring: both
polarity clusters map to the correct post-shift labels, expressed extensionally (marker
enumeration, not request/report intension) with one stale exception clause — a pre-rule
fragment ('still awaiting review' → FILE = the old request row) that names a train-split
marker and therefore never triggers on the disjoint held-out vocabulary. That clause is
exactly the auditability pitch: a human reads the rule, sees the stale fragment, vetoes it.

Complaints pre-empted, with the artifact that answers each:
- *"Single seed / noise"* — within-seed McNemar above; seed-level replication still
  requires the 5-seed run (open).
- *"The kill arm is a strawman"* — a copy policy with PERFECT staleness filtering (store
  restricted to the 60 post-shift trials) ceilings at 0.756 (nn) / 0.778 (top-5 majority):
  below Eigen 0.911. No exemplar policy over this buffer reaches the Treatment number.
  Separately, realized Recency_RAG (0.522) landed *below* Control_RAG (0.556) — in-context
  recency-weighting is hard to execute even with newest-first ordering and a staleness hint.
- *"You amended the G3 estimator mid-stream"* — amended BEFORE the pilot, by arithmetic
  (cPCA is blind to location differences); and now calibrated to the same standard as the
  original gate: results/calibration/gate_roc_mean.json shows noise fire-rate 0.00–0.05 and fire-rate 1.00 from
  snr 0.5 at the pilot's n_fail≈30, at live cadence with the exact pilot kernel config.
- *"45% parse-fallback = parsing luck"* — inverted: Control_RAG 0.75 and Recency_RAG 0.69
  are HIGHER (exemplar-echo is a property of exemplar-laden prompts); one identical parser
  everywhere; Treatment's post-crystallization context is the cleanest of the memory arms.
- *"The axiom replaced exemplars — maybe dropping stale exemplars did the work"* — the
  axiom was injected on all 90 held-out items, so Treatment's context was axiom-only;
  Baseline (empty context) scores 0.033. The content, not the absence, carries the effect.
- *Still open*: 5-seed replication; the axiom is extensional (markers) rather than
  intensional (polarity) — fine for the accuracy claim, worth naming in any write-up;
  surprise-gate ablation remains unrun (crystallizer consumes ungated residuals).

**If only three things get done**: (1) RFμ + G0; (2) the S0 synthetic gate-ROC — it converts
"0 axioms fired" from an anticlimax into a calibration result; (3) the Rule-Shift pilot seed.

## 7. Gate v2 amendment (2026-07-25): evidence accumulation, calibrated to a stated error budget

**Trigger** (recorded before seeds 23/7 finished): replication seeds 2 and 18 both went
gate-shut → 0 axioms → Treatment ≡ Control_RAG on all 90 held-out predictions (verified
identical). Their kernel telemetry shows the gate wasn't refusing noise — it was refusing
*evidence*: across 7 checks the contrast direction was flagged stable on 6, with λ₁ at
0.60–1.10× the edge every time. A fixed failure axis reproducing across overlapping windows
is exactly the signature the gate exists to find, and the rule scored it zero.

**Diagnosis — the v1 rule's two free parameters were never derived from an error budget:**
- the "edge" is the **max of 20 permutation draws** (implied per-check α ≈ 1/21, with
  enormous variance in the edge itself — it ranged 0.040–0.056 across checks on the same
  seed; seed 2's lone "detection" was against a lucky low draw);
- **streak = 3** consecutive binary super-edge checks, which discards all sub-edge evidence
  — a near-miss at 0.9× the edge counts exactly as much as pure noise.

Neither the max (vs a quantile) nor the 3 (vs 2 or 5) was chosen against a stated
false-fire target. The stability and novelty gates are untouched by this amendment.

**v2 rule** (one calibrated quantity replaces both free choices):
- per check, a permutation **p-value** — rank of the observed statistic among 200
  permutations, `p = (1 + #{perm ≥ obs}) / 201` — not a max;
- decayed log-evidence `E_t = 0.8·E_{t−1} + (−ln p_t)`;
- fire when `E_t ≥ θ` AND stable AND novel;
- **θ is the smallest value with run-level noise fire-rate ≤ 0.05** at live cadence
  (16 checks × 10 trials, window 60), measured by simulation with the overlapping-window
  dependence included — consecutive windows share ~50/60 residuals, which correlates even
  noise directions and inflates the stable flag, so θ must absorb that; an analytical
  threshold would be anti-conservative. Calibration: `gate_roc_v2.py` → `results/calibration/gate_roc_v2.json`
  (v1 and v2 run paired on identical observation streams).

**Integrity protocol.** The trigger came from live-seed telemetry, so this is a post-hoc
amendment and is labeled as such everywhere. To keep it from being tuning-to-the-test:
θ is selected **only from the synthetic noise null** — no live-seed data enters the
calibration; the running 5-seed replication completes under v1 unchanged and its verdict
is reported as THE pre-registered result; v2 then gets its own pre-registered re-run —
Treatment arm only per seed (the other four arms are gate-independent and are reused),
same primary endpoint (Δ vs Recency_RAG ≥ +0.10 across seeds), same G4 axiom audit —
and is reported alongside v1 as "v2, amended after 2 gate-shut seeds", never replacing it.
A v2 result that still misses is reported as a miss.

**Outcome (2026-07-26, results/calibration/gate_roc_v2.json) — the calibration cancelled the re-run.**
θ landed at 13.16 (per-cell q95 of armed noise evidence 7.44/10.14/13.15, vs a null
stationary mean of ~5 — the overlapping-window dependence roughly doubles what
independence would predict, vindicating the empirical-calibration requirement). At that
budget, v2 fires no more often than v1 anywhere on the grid (identical at n_fail 30/45,
slightly lower at 25), and v2's fresh-noise specificity check came back 0.10 at n30 —
itself over budget. Both rules fire ≥ 0.90 from snr 0.15 at live window compositions,
i.e. from a *sustained* λ/edge ≈ 1.05; pure noise averages 0.86–0.89; the shut seeds'
per-check means were 0.81–0.85. The apparent direction-stability on shut seeds is also
what overlapping-window noise produces (consecutive windows share ~5/6 of residuals).
**Conclusion: the shut seeds are signal-starved, not threshold-starved — no gate honoring
a 5% run-level false-fire budget fires on them — and the pre-registered v2 Treatment
re-run is cancelled as pointless-by-calibration** (the sim killed the spend before it
started; that is what these gates are for). Caveat: the sim plants a rank-1 shift in iid
gaussian embeddings, so the snr↔ratio mapping to live data is approximate — but the
within-sim v1-vs-v2 comparison is exact and answers the amendment's question. The live
next levers are **featurization** (a residual stream that actually carries the contrast)
and the **ungated-trigger ablation** (which now doubles as a check on this sim-to-live
mapping: if an ungated crystallizer extracts a correct rule from the shut seeds' windows,
the signal was there and the estimator missed it; if it writes garbage, the calibration
called it right).

## 8. Five-seed replication verdict (2026-07-26): pre-registered endpoint NOT met

All five seeds complete (42 pilot + 2, 18, 23, 7 run 2026-07-25/26, seeds sequential,
~3 h each; one OOM event killed llama-server mid-seed-23 and the run survived via
respawn — results/shift/comparison_results.shift.<seed>.json ×5).

| seed | gate | Eigen | Recency_RAG | Control_RAG | Δ (primary) |
|-----:|------|------:|------------:|------------:|------------:|
| 42 | **fired** | 0.911 | 0.522 | 0.556 | **+0.389** |
| 2 | shut | 0.644 | 0.544 | 0.644 | +0.100 |
| 18 | shut | 0.500 | 0.567 | 0.500 | −0.067 |
| 23 | shut | 0.600 | 0.611 | 0.611 | −0.011 |
| 7 | shut | 0.633 | 0.656 | 0.633 | −0.022 |

**Pooled (450 paired items): Eigen 0.658, Recency_RAG 0.580, Δ = +0.078 — below the
pre-registered +0.10 bar. Endpoint NOT met**, despite the direction being real (McNemar
89-vs-54 discordant, one-sided p = 2.2e-3) and Eigen > Control_RAG (0.658 vs 0.589)
holding. Verdict: **miss**; the +0.10 effect-size bar does the work the p-value can't.

Mechanistic accounting, stated precisely: on **3 of the 4** gate-shut seeds (2, 18, 7),
Treatment's 90 held-out predictions are IDENTICAL to Control_RAG's, verified item-level —
no axiom, no effect, no hidden channel. Seed 23 is the exception and is worth recording
honestly: with **zero** axioms stored and temperature 0, it still differs on exactly 1 of 90
items (0.600 vs 0.611). No memory channel can explain a difference with no axiom in the
store, so this is residual nondeterminism in the serving stack — but it means the claim is
"3 of 4 identical", not "all four", and any future run should expect ~1-item jitter rather
than treating bitwise identity as a guaranteed invariant.
The entire pooled gain is seed 42's fire (+0.355 over its own control).
So the result decomposes cleanly:
- **Conditional on firing, the mechanism wins big and legibly** (one seed: +0.389 over
  the kill arm, McNemar 1.6e-8, auditable prose rule).
- **The gate fires on 1/5 seeds at this task's real SNR.** The shut-seed telemetry shows
  why (§7): direction-stable contrasts at 0.60–1.10× a max-of-20-permutations edge,
  discarded by the binary streak. And the v2 sweep's first row adds a caution: pure noise
  averages λ/edge ≈ 0.87 under this estimator, so magnitude-wise the shut seeds sit near
  snr ≈ 0.1–0.35 — partly a threshold problem (§7's fix), possibly partly a signal problem
  (residual featurization may simply carry little contrast on these vocabularies).

This is the pre-registration's named failure mode ("G3 never fires → report that the gate
correctly identifies this structure as not spectrally compressible") landing on 4 of 5
seeds, with the pilot seed showing what happens when it does fire. Follow-ups, in order:
(1) ~~the §7 gate-v2 re-run~~ — cancelled by its own calibration (§7 outcome: the shut
seeds are signal-starved, not threshold-starved; v2 fires no more than v1 at an honest
budget); (2) an exploratory ungated arm (crystallize on a fixed schedule) — now doing
double duty as the signal-vs-estimator arbiter; (3) residual featurization work if the
ungated arm shows the signal was there; (4) the label-noise wedge, unchanged.

## 9. Ungated-trigger ablation protocol (2026-07-26, pre-registered; not yet run)

Script: `ungated_ablation.py` (written box-idle, untested until Ollama is free). Per shut
seed {2, 18, 23, 7}: rebuild the end-of-run window (embeddings + stale-copier proxy for
was_correct — a trial fails iff the similarity-nearest earlier trial's stored label ≠ its
era-correct label; per-trial live correctness wasn't persisted, and this proxy reproduces
the planted failure structure and the observed ~0.42–0.45 post-adapt accuracy), compute
the ungated mean-contrast axis, and force ONE crystallization — no detectability gate, no
streak, no stability/novelty. The clean-RULE guard stays (it's a storage-hygiene fix, not
a gate). **Readout, committed in advance**: G4 rule-text scoring only — a rule counts as
correct iff it maps BOTH polarities to the era-correct post-shift labels (the seed-42 bar).
Correct on ≥2 of 4 seeds → the signal was in the episodes and the *estimator/featurization*
missed it → featurization work follows. Correct on ≤1 → the calibration's signal-starved
verdict stands and the Rule-Shift chapter closes as reported. Accuracy impact of forced
axioms is explicitly out of scope for this ablation (a wrong forced axiom poisoning
held-out accuracy is already known from the flip experiment; the question here is signal
existence, not deployment policy). Exploratory label, reported alongside — the 5-seed
verdict (§8) is unaffected either way.
