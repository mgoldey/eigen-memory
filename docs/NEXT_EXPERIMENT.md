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
0.84). *(These are the original, residue-affected numbers, kept as the record of what was run
that day; the clean rerun below supersedes them — Treatment_Eigen 0.922.)* Primary endpoint Δ
vs Recency_RAG = **+0.389** (bar +0.10), McNemar p = 1.6e-8
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
>
> **Rerun attempted and abandoned (2026-08-10).** Started on the fixed extractor and stopped
> after 2 of 5 arms on wall-clock cost (~6 h/seed measured on this hardware; see the runtime
> note in the README). What it produced before being stopped, for whatever it is worth:
> Baseline 0.011 (archived run: 0.033) and **Oracle_Post 0.989** (archived: 0.967) — i.e. the
> executor still applies the true rule near-perfectly, so the premise the experiment rests on
> reproduces. `Treatment_Eigen` is the arm that would settle the question and it runs last, so
> it was never reached. **The caveat above stands unchanged: 0.911 is not yet a clean test.**
>
> **Rerun completed (2026-08-11) — the result holds.** All five arms on the fixed extractor,
> same seed and executor. The stored axiom is a single clean rule line with **no trailing
> CoT**: *"If the status is 'resolved' or 'already through review', label as ESCALATE; if the
> status is 'pending' or 'awaiting review', label as DEFER (unless it is a specific 'still
> awaiting review' case which is FILE)."* — same rule, same stale exception clause, same
> strength (1.31) as the residue-affected run, minus the residue. Held-out **0.922** vs the
> archived 0.911 (+0.011). The four control arms reproduced within ±0.022 — Baseline 0.033
> (=), Oracle_Post 0.978 (+0.011), Control_RAG 0.533 (−0.022), Recency_RAG 0.522 (=) — so the
> rig was stable and the comparison is like-for-like. Exact McNemar on the rerun's
> `test_correct` vectors: 39 items Eigen-only correct vs 3 Recency-only, **p = 5.6e-9**.
> Per-class, Treatment scores requests 0.949 / reports 0.902 against Recency's 0.256 / 0.725
> — the gain is concentrated in `request`, the class the shift redefined (FILE → DEFER).
> **Conclusion: the 0.911 was not an artifact of the injected CoT residue; one clean rule line
> reaches the same place.** The pre-registered five-seed verdict (miss, +0.078) is unchanged in
> direction; substituting 0.922 for 0.911 moves the pooled gain by +0.002, still short of the
> +0.10 bar. Pre-fix artifact archived as
> `results/shift/comparison_results.shift.42.cotresidue.json`.
>
> Two operational bugs surfaced during the rerun and were fixed (they cost ~4 h of wall clock,
> not any result): `memory-db` had no restart policy, so a reboot mid-run left Postgres down
> (`3be46d9`); and the Ollama client was built with no `timeout`, inheriting the SDK's 600 s
> default, so a single dropped local connection stalled the run for ten minutes — hit twice,
> now `timeout=120, max_retries=3` (`017f12c`).

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
  below Eigen 0.922. No exemplar policy over this buffer reaches the Treatment number.
  Separately, realized Recency_RAG (0.522) landed *below* Control_RAG (0.533 on the rerun,
  0.556 as originally run) — in-context recency-weighting is hard to execute even with
  newest-first ordering and a staleness hint.
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
re-run is cancelled as pointless-by-calibration**
*(Superseded on the signal-existence question by the §9 ablation, run 2026-08-11: all four
shut seeds crystallized correct rules from the same windows, so the signal was present. What
survives here is the narrower claim — no gate fires on **this live statistic** at an honest
budget. The gap between the two is the fail/succ split quality; see §9.)* (the sim killed the spend before it
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
| 42 | **fired** | 0.911 † | 0.522 | 0.556 | **+0.389** |
| 2 | shut | 0.644 | 0.544 | 0.644 | +0.100 |
| 18 | shut | 0.500 | 0.567 | 0.500 | −0.067 |
| 23 | shut | 0.600 | 0.611 | 0.611 | −0.011 |
| 7 | shut | 0.633 | 0.656 | 0.633 | −0.022 |

† Seed 42 was rerun on the fixed extractor (2026-08-11) and scored **0.922**; the table keeps
the as-run 0.911 because the pooled figures below were computed from it. Substituting the
rerun moves the pooled Δ by +0.002, which does not change the verdict.

**Pooled (450 paired items): Eigen 0.658, Recency_RAG 0.580, Δ = +0.078 — below the
pre-registered +0.10 bar. Endpoint NOT met**, despite the direction being real (McNemar
89-vs-54 discordant, one-sided p = 2.2e-3) and Eigen > Control_RAG (0.658 vs 0.589)
holding. Verdict: **miss**; the +0.10 effect-size bar does the work the p-value can't.

> *(This is the as-run v1 verdict and stays as the record of what the pre-registered
> mechanism produced. §10 reports a rebuilt pipeline — outcome-stream detection,
> change-point truncation, validation and retirement — that clears the same bar at
> **+0.242** on the same five seeds. The miss below is real and was not retroactively
> rewritten; the v4 result is a different mechanism measured against the same bar.)*

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
(1) ~~the §7 gate-v2 re-run~~ — cancelled by its own calibration (§7 outcome: v2 fires no
more than v1 at an honest budget on the live statistic); (2) ~~an exploratory ungated arm~~
— **run 2026-08-11, §9: correct on 4 of 4, the signal WAS in the episodes**; (3) persist
per-trial correctness so the live gate sees the split the ablation reconstructs — now the
immediate item, and a prerequisite for (4); (4) residual featurization work; (5) the
label-noise wedge, unchanged.

## 9. Ungated-trigger ablation protocol (2026-07-26, pre-registered) — RUN 2026-08-11

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

**Result (2026-08-11): correct on 4 of 4 — the pre-registered "signal was there" branch,
cleared with margin (bar was ≥2).** Artifacts: `results/shift/ungated_ablation.<seed>.json` ×4.

| seed | crystallized rule | planted post-rule | G4 |
|---|---|---|---|
| 2 | completed→ESCALATE; incomplete→DEFER | request→DEFER, report→ESCALATE | both ✓ |
| 18 | incomplete/pending→FILE; completed→DEFER | request→FILE, report→DEFER | both ✓ |
| 23 | completed/handled→FILE; pending→DEFER | request→DEFER, report→FILE | both ✓ |
| 7 | resolved/handled→FILE; pending/awaiting→DEFER | request→DEFER, report→FILE | both ✓ |

All four are clean prose (101–165 chars, no CoT residue). Each expresses the
completed-vs-incomplete distinction extensionally, as seed 42 did — marker enumeration
rather than the request/report intension — and each maps both polarities correctly.

Under the ablation's reconstruction λ₁ exceeds the noise edge on all four (ratio 1.02–1.46),
where the live runs' final checks read 0.75–0.91:

| seed | live ratio (last check) | ablation ratio |
|---|---|---|
| 2 | 0.88 | 1.14 |
| 18 | 0.87 | 1.02 |
| 23 | 0.75 | 1.46 |
| 7 | 0.91 | 1.18 |

**What this establishes:** the shut seeds were *not* signal-starved. The failure structure is
real and rank-1-recoverable from the same episodes.

**What it does not establish:** that the live gate is simply miscalibrated. The proxy for
`was_correct` is *cleaner* than live reality — live failures include executor mistakes that
are noise rather than rule-shift signal — so an unknown part of the ratio gap is the proxy
denoising the fail/succ split rather than a featurization defect. This is the sim-to-live
mapping the protocol above named as the open question, and it is narrowed, not closed.

**Consequent next step**, replacing "featurization work follows" as the immediate item:
persist per-trial correctness in the harness so the live gate sees the same split the
ablation reconstructs. That is a harness change, and it is a prerequisite for interpreting
any featurization experiment — without it, one cannot tell a bad featurization from a noisy
correctness signal. §7's "signal-starved" conclusion is superseded on the signal-existence
question; its narrower finding (v2 fires no more than v1 at an honest false-fire budget on
the *live* statistic) is unaffected.

### 9b. Sequential (e-value) trigger — fires more, writes worse rules (2026-08-12)

Built as `sequential_gate=True` (off by default): permutation p-values instead of
a max-edge comparison, power-calibrated to e-values, accumulated multiplicatively,
firing at 1/alpha under Ville's inequality. Type-I control is genuine and measured
(null fire rate 0.000–0.017 across n_fail 25→10, `tests/test_sequential_gate.py`).
It fires where the streak rule cannot — seed 7 (0→1 axiom), seed 42 (1→2).

**The extra fires are not a win. G4-scoring the rule text reverses the verdict.**

Seed 42, planted post-shift `request → DEFER, report → ESCALATE` (stale pre-shift
`request → FILE`):

| gate | axiom | verdict |
|---|---|---|
| streak | *"unresolved task requiring action → **DEFER**; resolved task being tracked → **ESCALATE**"* | **both branches correct** |
| sequential [1] | *"still pending or awaiting action → **FILE**; resolution or confirmation → **ESCALATE**"* | request branch is the **pre-shift** label |
| sequential [2] | *"explicitly stated urgency → **ESCALATE**; otherwise → **FILE**"* | **both branches wrong** |

Seed 7 showed the same failure: *"ongoing need for attention → ESCALATE; otherwise
→ FILE"* against a planted `request → DEFER`, where ESCALATE is again the pre-shift
label. Three sequential axioms across two seeds, every one with a wrong branch,
against the streak rule's one clean axiom.

**Why: it fires before the shift.** The shift lands at batch 11. The sequential
gate's evidence path on seed 42 is E = 0.8, 1.5, 2.4, 11.8, **30.2 (fired, batch 7)**,
then collapses to 0.3 by batch 11 and rebuilds to **78.5 (fired, batch 15)**. Axiom
[1] was written at batch 7 — four batches *before the rule changed*. It is not a
hallucination; it is an accurate statement of the **pre-shift** rule, which became
false five batches later. Axiom [2] then fired post-shift with the good direction
already consumed, landing on a worse axis.

The streak rule's checks on the same stream: 1.00, 0.78, 0.81, 0.82, 1.45, 1.51,
**1.48 (fired, batch 16)**. Its three-consecutive requirement acted as a *delay*
that let the post-shift signal dominate before committing. That is a real function,
not the pure waste §9a treated it as — the earlier framing ("two brittle knobs where
one principled test would do") was wrong about what the second knob was doing.

**The structural gap this exposes: the crystallizer has no notion of when a rule
stopped being true.** On a non-stationary target, firing faster produces confidently
stale rules — the exact failure this project set out to solve, reproduced inside the
proposed fix. Detection speed and rule validity are in tension, and nothing in the
current design mediates them.

Two candidate resolutions, neither built:

1. **Validate axioms before injection** — score a candidate against recent trials,
   reject if it does not beat the current policy. Makes early firing *safe* rather
   than preventing it, and the §9 ablation (4/4 correct rules when forced) suggests
   the accept rate would be usable.
2. **Recency-weight the evidence** — decay contributions so pre-shift evidence
   cannot fund a post-shift fire. Closer to what the streak rule was doing by
   accident, but with a stated budget rather than a hand-tuned count.

### 9a. Correctness persistence + gate replay (2026-08-11, built; awaiting data)

`run_shift_experiment.py` now writes `trial_correct` — item-level trial-stream correctness —
on each of the three training arms. Previously only the two summary means (`pre_shift_acc`,
`post_adapt_acc`) survived, and per-item correctness died with the Postgres container; it was
written to `episodic_buffer.was_correct` but never serialized. This is the only genuinely
unrecoverable input to the gate: residuals are deterministic from the trial text and can be
rebuilt at will, which is exactly what the §9 ablation does.

`gate_replay.py` consumes it. For a given seed it computes the end-of-run contrast statistic
twice over identical featurization — once with the ablation's stale-copier proxy, once with
the real split — and prints both beside the live run's own last check. The comparison is
designed to separate the two candidate causes that §9 left entangled:

| pattern | reading |
|---|---|
| live ≈ replay < proxy | **featurization** — a clean split doesn't help; the embedding-mean contrast is the bottleneck |
| live < replay ≈ proxy | **label noise** — the gate was starved by executor error polluting the fail/succ split |

**Result (seed 23, 2026-08-11): featurization, not label noise.** Seed 23's `Treatment_Eigen`
arm was rerun to capture `trial_correct` (the four control arms were reused from the committed
run via the driver's resume path — they don't feed a gate). Artifact:
`results/shift/gate_replay.23.json`.

| split | λ₁ | edge | ratio |
|---|---|---|---|
| proxy (stale-copier, as §9 used) | 0.0777 | 0.0531 | **1.46** |
| **live (real per-trial labels)** | **0.0418** | **0.0439** | **0.95** |
| live run's own last check | 0.0418 | 0.0508 | 0.82 |

The live split tracks the live run, not the proxy — and λ₁ is *identical* (0.0418) between the
replay and the run's own telemetry, confirming the replay reproduces what the gate actually
saw. Handed real correctness labels over unchanged featurization, the statistic still sits
below the noise edge.

**So the label-noise hypothesis is retired.** The proxy's 1.46 came from the proxy being an
easier problem: it splits 43 fail / 17 succ on a deterministic copying rule, where reality
splits 32/28 with genuinely mixed failure causes. §9's rules were correct because the proxy
hands cPCA a cleanly separable problem — not because executor noise was starving the live
gate. The bottleneck is the embedding-mean contrast itself, which is where the next work
belongs (§9's "featurization work follows", now the immediate item rather than a conditional).

Scope: one seed. Seed 23 was picked as the sharpest test (widest proxy-vs-live gap in §9), so
it is the most informative single seed, but confirming the other three shut seeds needs three
more `Treatment_Eigen` reruns (~1 h each).

**All four shut seeds replayed (2026-08-12). The one-seed reading did not generalize, and the
more important finding is about the noise edge, not the split.**

| seed | proxy | live | live run | λ₁ (replay = run) | edge replay | edge run | edge spread |
|---|---|---|---|---|---|---|---|
| 2 | 1.14 | 0.89 | 0.84 | 0.04159 | 0.04664 | 0.04956 | 1.06× |
| 7 | 1.18 | 1.13 | 1.28 | 0.06549 | 0.05804 | 0.05110 | 1.14× |
| 18 | 1.02 | **1.03** | 0.78 | 0.04107 | 0.03998 | 0.05232 | **1.31×** |
| 23 | 1.46 | 0.95 | 0.82 | 0.04182 | 0.04392 | 0.05080 | 1.16× |

**λ₁ agrees between the replay and the live run to ≤2×10⁻⁷ relative on every seed** — i.e.
floating-point identical, not merely close. The replay reproduces
exactly what the gate saw, so *every* live-vs-run ratio difference in this table is the
permutation edge moving, not the signal. On seed 18 that alone flips the verdict: the same
λ₁ = 0.04107 reads as detectable against the replay's edge (0.03998) and as below-edge against
the run's (0.05232). One of four seeds has its outcome decided by which permutation draw came
up.

Consequences for the §9a hypothesis test:

- Seeds 2 and 23 behave as the one-seed reading predicted — real labels pull the statistic
  down toward the live run and away from the proxy. Featurization, not label noise.
- Seed 18 does the opposite (live 1.03 tracks proxy 1.02, against a run of 0.78), but for a
  reason that has nothing to do with the split: its edge estimate is 31 % apart on identical
  data.
- Seed 7 sits above the edge on all three statistics (1.18 / 1.13 / 1.28) and in the live run
  reached **streak = 2 of the 3** consecutive detections required, with λ₁ (0.065–0.070)
  well above its committed run's 0.042–0.052. It did not fire only because the trial stream
  ended before a third check. Its committed run never exceeded 0.93.

**Revised conclusion.** The clean "featurization is the bottleneck" claim from the seed-23
replay is *not* supported across four seeds. What the four seeds share is that every relevant
quantity sits within a few percent of the decision boundary: λ₁/edge on real labels spans
0.78–1.28 across all four seeds, and the edge itself varies up to 1.31× on identical data
(the proxy ratios run higher, to 1.46, which is the point of §9a). The gate is
not obviously mis-featurized *or* mis-thresholded — it is operating with essentially no margin,
so seed-level outcomes are decided by estimator variance rather than by the presence or absence
of structure. That is a weaker and less tidy claim than §9a's first draft, and it is the one
the data supports.

Two consequent work items, in order:

1. **Stabilize the noise edge** — more permutation draws, or pool/smooth the edge across
   checks rather than re-estimating it independently each time. Cheap, and it is a prerequisite
   for interpreting any featurization change: right now a featurization improvement and a
   favorable edge draw are indistinguishable at these effect sizes.
2. **Replace the threshold-plus-streak rule with an anytime-valid sequential test** (e-values /
   testing-by-betting). The current design tunes two brittle knobs — a permutation quantile and
   a consecutive-detection count — where a single sequential test with a stated Type-I budget
   would do, and it would remove exactly the estimator-variance sensitivity this table exposes.

Held-out accuracy on the reruns (all four still `n_axioms=0`, matching their committed runs):
seed 2 0.644→0.656, seed 7 0.633→0.567, seed 18 0.500→0.533, seed 23 0.600→0.544. Moves in
both directions on shut-gate runs, i.e. held-out executor noise. Pooled with seed 42's clean
0.922, the five-seed Eigen mean goes 0.660→0.644 and Δ vs Recency_RAG goes +0.080→+0.064 —
still a miss against the +0.10 bar. §8's table keeps the as-run numbers its pooled figures
were computed from; originals are preserved as
`results/shift/comparison_results.shift.<seed>.pre-trialcorrect.json`.

Incidental: the rerun's `Treatment_Eigen` scored 0.544 against the committed 0.600. Both are
shut-gate runs with `n_axioms=0`, so this is executor noise on the held-out set, not a
mechanism difference. The committed artifact now carries 0.544; the original is preserved at
`results/shift/comparison_results.shift.23.pre-trialcorrect.json`. Substituting it (together
with seed 42's clean-extractor 0.922) moves the pooled five-seed Eigen mean to 0.649 and Δ to
+0.069, against 0.660 / +0.080 with seed 23's as-run 0.600 — still a miss against the +0.10
bar. §8's table keeps the as-run numbers its pooled figures were computed from.

---

## 10. Rebuilt pipeline, five seeds (2026-08-13): endpoint MET at +0.242

§8 missed at +0.078 with the trigger firing on 1 of 5 seeds — and on rerun that one seed
did not fire again (§9a), so the honest v1 summary is closer to "fired once, not
reproducibly". §9–§9c traced that to four separate defects. Fixing them changes the
verdict.

| seed | v4 | v1 | Δ | axiom | request | report | kill arm | oracle |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.956 | 0.656 | **+0.300** | ✓ | **1.000** | 0.915 | 0.544 | 0.989 |
| 7 | 0.567 | 0.567 | −0.000 | — | 0.353 | 0.846 | 0.656 | 0.956 |
| 18 | **0.978** | 0.533 | **+0.445** | ✓ | **1.000** | 0.957 | 0.567 | 0.900 |
| 23 | 0.644 | 0.544 | +0.100 | — | 0.480 | 0.850 | 0.611 | 0.867 |
| 42 | 0.967 | 0.556 | +0.411 | ✓ | **1.000** | 0.941 | 0.522 | 0.978 |

**Pooled: Eigen 0.822 vs Recency_RAG 0.580 — Δ = +0.242 against the +0.10 bar. Endpoint
MET.** Fired on 3 of 5 seeds; every axiom it wrote was correct on both polarities.

### What changed, and why each mattered

**Detection moved off the spectral statistic.** The contrast λ₁/edge sits at 0.78–1.28 on
real labels while the permutation edge itself varies 1.31× on identical data, so seed-level
outcomes were decided by estimator variance (§9a). Recomputed from the committed telemetry,
the streak rule fires **0 of 5**. The outcome stream carries the same event far more
cleanly — accuracy drops 0.12–0.26 at the shift on every seed — and a betting e-process
plus an unseen-label check fires **5 of 5 at +1 to +6 trials**, zero pre-shift.

**Change-point truncation was the difference between a correct rule and a stale one.**
Pre-change *successes* are successes under the OLD rule; leaving them in the contrast pulls
the success mean toward the old regime, which is exactly what let the crystallizer write
the pre-shift rule back out. On seed 42, same data and same statistic:

| run | window when crystallizing | λ₁ | rule |
|---|---|---|---|
| v3 | 40 pre / 20 post | 0.042 | stale (`request → FILE`) |
| v3b | 40 pre / 20 post | 0.042 | stale (`request → FILE`) |
| v4 | 0 pre / 50 post | 0.080–0.110 | correct, both branches |

This **inverts** the offline finding in `2813fe6`, which had `full` and `truncated` tied at
5/5 and nearly retired the idea. That test crystallized at *end of stream*, where the buffer
is already all post-shift and truncation has nothing to remove. Mid-stream — where a
deployment lives — it is the whole difference.

**Validation and retirement are the safety net, and both earned their place.** v3 stored a
half-stale axiom and scored 0.500, *worse* than injecting nothing (0.556), with `request` at
0.077. v3b wrote the same class of rule and retired it a batch later, scoring 0.611. The
accept bar needed two corrections found the same way: a margin scaled to the tail size
(0.40-vs-0.33 on ten items is one item of noise), and a per-class bar floored at **chance**
rather than at the agent's own rate — on the class whose rule just changed the agent has
already collapsed to 0.00, so comparing only to the agent puts the bar on the floor exactly
where it matters.

### The two misses are stream length, not misjudgement

Neither seed wrote a stale rule. Both wrote **nothing** and scored at or above their
no-axiom baselines.

- **Seed 7** — 49 failures in 160 trials, the lowest of any seed. After truncation it never
  refilled to 25 failures plus 40 post-change trials before the stream ended.
- **Seed 23** — reached readiness at batch 15 with λ₁ = 0.044, but the direction was
  `unstable` and the stream ended before a confirming check.

Both are the 60-trial post-shift stretch expiring against a 60-trial window. A longer
post-change stretch is the obvious next rig change, and it is a rig fix rather than a
mechanism one.

### Scope and honest caveats

- v4's parameters (truncation, readiness 40) were chosen while watching **seed 42** fail in
  v3 and v3b. Seed 42 is therefore selection-affected. Seeds 2, 7, 18 and 23 are clean
  tests, and the pooled figure includes all five.
- **Seed 18 is the load-bearing generalization evidence**: its rule mapping is *inverted*
  relative to seeds 2 and 42 (completed → DEFER, not ESCALATE), so the crystallizer is not
  reproducing a memorized rule shape. It scored 0.978 against its own oracle of 0.900 — the
  self-written axiom beat pasting the true rule in prose.
- `request` accuracy is **1.000 on every seed that fired** — the class the shift redefines,
  and the class that collapsed to 0.077 in v3 when a stale axiom went live.
- One configuration, five seeds, one task. Nothing here shows the mechanism transfers to a
  shift that *permutes* existing labels rather than introducing a new one — the unseen-label
  signal carries most of the detection on this task and would not exist there.

Artifacts: `results/shift/comparison_results.shift.v4.<seed>.json` ×5,
`results/shift/outcome_detection.json`. Config: `run_shift_experiment.py --v4`.
