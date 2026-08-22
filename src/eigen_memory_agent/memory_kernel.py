"""Residual-error spectral consolidation kernel (docs/THEORY.md).

This replaces the original kernel, which implemented a mechanism the theory review
disproved (PCA over raw failure embeddings, crystallize-every-batch, cosine-to-
eigenvector axiom retrieval, an arithmetic-hint prompt). The corrected mechanism:

- **Input = retrieval residuals** (query embedding minus retrieved-neighbor embedding),
  split by outcome into failure and success buffers. Retrieval cancels the dominant
  similarity axes; what survives in the residual is what retrieval failed to control.
- **Estimator = contrastive eigendecomposition**: top eigenvector of
  Cov(failure residuals) - Cov(success residuals). Successes share retrieval's leakage
  and noise but carry no failure-specific spike, so the shared structure cancels.
- **Trigger = detectability + stability**, not a schedule: crystallize only when the top
  eigenvalue clears a permutation-estimated noise edge (shuffle fail/success labels; the
  empirical analogue of the Marchenko-Pastur bulk edge) AND the direction is stable
  across successive checks AND it has not already been crystallized.
- **Contrast sets by projection**: the introspection prompt is seeded with failures at
  extreme opposite projections along the detected axis, plus matched successes — and is
  task-neutral (no arithmetic hints).

Verified end-to-end on a planted world in tests/test_kernel_theory.py; unit-tested with
fakes in tests/test_kernel_consolidation.py.
"""

import json
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

# Working dimension for the eigenanalysis. Residuals are projected onto the top
# R_COMPONENTS principal components of the pooled residual cloud first: the spike
# survives any projection that keeps its neighborhood, and reducing d moves the
# BBP detectability transition earlier (THEORY.md section 5).
R_COMPONENTS = 50


# ---- Sample floors ----
# Floors before the spectral machinery runs at all. Below these, even the
# permutation edge is too noisy to trust.
MIN_FAIL_RESIDUALS = 25
# Eligibility floor when the SEQUENTIAL gate is on. The check already runs once
# per batch, but a 60-trial window holding >=25 failures is only satisfiable at a
# high failure rate: measured on the Rule-Shift seeds, the window+floor pair
# allowed 2 checks on seeds 7 and 23 (49 and 59 failures) against 9 on seed 2
# (79 failures) -- so low-failure seeds got structurally fewer chances to detect,
# independent of whether their signal was real.
#
# Lowering it costs POWER, not VALIDITY: the permutation p-value is exact in
# finite samples at any n, and the measured null fire rate over 12 checks stays
# at or below 0.017 for n_fail down to 10 (test_type_i_holds_at_small_fail_counts).
# 15 keeps a usable sample for a 40-component contrast while roughly doubling the
# number of eligible checks on the starved seeds.
MIN_FAIL_RESIDUALS_SEQ = 15
MIN_SUCC_RESIDUALS = 10


# ---- Axiom validation (§9b) ----
# How many of the MOST RECENT trials a candidate axiom is scored on, and the
# minimum needed to judge at all. One batch of 10 is the natural unit -- long
# enough to separate a live rule from a superseded one, short enough that it is
# genuinely "recent" on a stream where the rule can change every 60 trials.
VALIDATION_WINDOW = 10
VALIDATION_MIN_ITEMS = 5
# Minimum examples of a class before its per-class score can veto a candidate.
# Below this a single unlucky item in a rare class would block every rule.
VALIDATION_MIN_CLASS_ITEMS = 3


# ---- Outcome trigger (§9d) ----
# Trials used to establish the pre-change hit rate before the detector is allowed
# to fire, and the betting fraction of the e-process. Measured offline on all
# five Rule-Shift streams: 5/5 detection, 0 pre-shift false fires, median delay
# 1 trial -- against 0/5 for the spectral streak rule on the same data.
OUTCOME_BURN_IN = 30
OUTCOME_BET = 0.5
# Post-change trials required before a rule may be WRITTEN. Detection stays
# immediate; only formation waits. v3 fired at the shift and crystallized two
# batches later from a 60-trial window still mostly pre-shift, producing a rule
# whose changed-class branch was the pre-shift label -- and scoring 0.077 on
# that class against 0.359 for injecting nothing. 20 is a third of the window:
# enough for the contrast to be dominated by post-change evidence without
# waiting so long that a short stream never qualifies.
FORMATION_MIN_POST_CHANGE = 20


# ---- Permutation edge ----
# Number of label-shuffles used to estimate the noise edge. The edge is the max
# top-eigenvalue over shuffles: anything a random fail/success split can produce
# is noise by construction. With max-over-N as the edge, a pure-noise lambda1
# clears it with probability ~1/(N+1) per check — 20 shuffles keeps that under
# 5% (the old value of 5 allowed ~17%, and the gate is re-checked every batch).
N_PERMUTATIONS = 20
# Permutations for the SEQUENTIAL gate's p-value. Far more than the max-edge rule
# needs, because a p-value's resolution floor is 1/(1+N): at N=20 the smallest
# attainable p is 0.048, which cannot supply meaningful evidence. 199 gives a
# floor of 0.005 and is still cheap (the permutation loop is O(N) mean contrasts).
N_PERMUTATIONS_SEQ = 199
# Power-calibrator exponent mapping p-values to e-values: e = k * p**(k-1), which
# integrates to exactly 1 against a uniform p (so E[e] = 1 under the null). 0.4
# keeps sensitivity to small p while capping one check at ~9.6x evidence.
E_CALIBRATOR_KAPPA = 0.4
# Anytime-valid firing threshold: the e-process fires when its running product
# reaches 1/ALPHA. Ville's inequality bounds the all-time false-fire probability
# by ALPHA for any stopping time, which is the honest version of the false-fire
# budget the permutation-quantile-plus-streak rule only approximated.
E_PROCESS_ALPHA = 0.05


# ---- Direction gates ----
# The direction must persist across consecutive checks (|cos| above this) before
# it is trusted — the failure stream is non-stationary, so a one-off axis is noise.
STABILITY_COS = 0.95
# A new direction too close to an already-crystallized one (|cos| above this) is
# the same axiom again; skip it. Deduplication falls out of geometry, not bookkeeping.
NOVELTY_COS = 0.8


# ---- Contrast-set sizes for the introspection prompt ----
N_CONTRAST_PER_SIDE = 3
N_MATCHED_SUCCESSES = 3


# ---------------------------------------------------------------------------
# KernelConfig — the 22 keyword arguments collapsed into a data object.
# Named constructors provide the two pre-registered configurations so the valid
# flag combinations are self-documenting rather than implicit.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelConfig:
    """Configuration for EigenMemoryKernel, replacing 22 keyword arguments."""

    model: str = "gemma3:4b"
    min_fail_residuals: int = MIN_FAIL_RESIDUALS
    min_succ_residuals: int = MIN_SUCC_RESIDUALS
    n_permutations: int = N_PERMUTATIONS
    stability_cos: float = STABILITY_COS
    novelty_cos: float = NOVELTY_COS
    rng_seed: int = 0
    extra_body: Optional[dict] = None
    window: Optional[int] = None
    contrast_on: str = "residual"
    consecutive_detections: int = 1
    sequential_gate: bool = False
    n_permutations_seq: int = N_PERMUTATIONS_SEQ
    e_process_alpha: float = E_PROCESS_ALPHA
    validate_axioms: bool = False
    retire_stale_axioms: bool = False
    outcome_trigger: bool = False
    formation_min_post_change: int = FORMATION_MIN_POST_CHANGE
    truncate_at_change: bool = False
    labels: Optional[list] = None

    @classmethod
    def static_task(cls, **overrides):
        """The static-task configuration (number-game, TREC, flip)."""
        return cls(**overrides)

    @classmethod
    def shift_v4(cls, seed=0, **overrides):
        """The v4 Rule-Shift pipeline: outcome trigger, validation, retirement,
        change-point truncation, and formation readiness gating."""
        defaults = dict(
            rng_seed=seed,
            window=60,
            contrast_on="embedding_mean",
            consecutive_detections=3,
            stability_cos=0.5,
            outcome_trigger=True,
            validate_axioms=True,
            retire_stale_axioms=True,
            truncate_at_change=True,
            formation_min_post_change=40,
        )
        return cls(**{**defaults, **overrides})


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _top_contrast_eig(fail_r, succ_r):
    """Top eigenpair of Cov(fail) - Cov(succ) in the given coordinates."""
    C = np.cov(fail_r, rowvar=False) - np.cov(succ_r, rowvar=False)
    w, V = np.linalg.eigh(C)
    return float(w[-1]), V[:, -1], w


def _mean_contrast(fail_r, succ_r):
    """Two-sample mean contrast: statistic = squared distance between the group
    means, direction = the mean-difference axis. The shift-regime estimator —
    a location difference between failures and successes that a covariance
    contrast cannot see (see __init__ notes on contrast_on)."""
    d = fail_r.mean(axis=0) - succ_r.mean(axis=0)
    return float(d @ d), _unit(d)


def _evalue_from_null(fail_r, succ_r, n_perm, rng, contrast="mean"):
    """Permutation e-value for "failures differ from successes along one axis".

    The streak rule compared lam1 against max(null) -- one bit per check, and a
    threshold whose sampling noise decided seed 18's verdict outright (edge
    0.03998 vs 0.05232 on identical data, docs/NEXT_EXPERIMENT.md §9a). The same
    permutations yield a far more informative quantity: the RANK of the observed
    statistic in the null distribution, i.e. a p-value.

    Uses the (1 + #{null >= obs}) / (1 + n_perm) estimator, which is exactly
    valid in finite samples (the +1s account for the observed value itself), so
    p is super-uniform under the null and e = 1/p is a genuine e-value with
    E[e] <= 1. Independent e-values multiply, which is what lets evidence
    accumulate across checks instead of being discarded.

    Note e = 1/p is NOT an e-value: E[1/p] = int_0^1 dp/p diverges, and it shows
    up empirically as a null mean around 7 with a max at the permutation floor.
    Instead p is passed through the standard power calibrator

        e = kappa * p**(kappa - 1),   kappa in (0, 1)

    which integrates to exactly 1 against a uniform p, so E[e] = 1 under the
    null by construction. kappa = 0.4 is a common default: it keeps useful
    sensitivity to small p while bounding a single check's contribution (at the
    1/200 floor it yields ~9.6x, not 200x).
    """
    stat = _mean_contrast if contrast == "mean" else (
        lambda a, b: _top_contrast_eig(a, b)[:2])
    obs, _ = stat(fail_r, succ_r)
    pooled = np.vstack([fail_r, succ_r])
    n_fail = len(fail_r)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        lam, _ = stat(pooled[idx[:n_fail]], pooled[idx[n_fail:]])
        if lam >= obs:
            ge += 1
    p = (1.0 + ge) / (1.0 + n_perm)
    return float(E_CALIBRATOR_KAPPA * p ** (E_CALIBRATOR_KAPPA - 1.0))


class _EProcess:
    """Multiplicative e-process with an anytime-valid firing rule.

    Fires when the running product of e-values reaches 1/alpha. Ville's
    inequality gives P(sup_t E_t >= 1/alpha) <= alpha for ANY stopping time,
    so unlike a fixed-window permutation test re-run every batch, checking as
    often as we like costs nothing in false-fire budget. That is the property
    the "3 consecutive detections" heuristic was standing in for.

    Tracks log-evidence for numerical stability over long streams.

    Per-check evidence is capped just below the firing threshold, so no single
    check can fire the process on its own. Seed 2 crossed the old edge once per
    run (ratios 1.08 and 1.02) and correctly stayed shut both times; a single
    lucky draw should not be more decisive than that.

    Do NOT floor log_e at 0. An earlier version did, reasoning that a run of
    uninformative checks should not have to be "paid back" before a real shift
    can fire. That turns the process into a random walk with a reflecting
    barrier, which reaches any threshold eventually: it fired on 100% of pure-noise
    streams in tests/test_sequential_gate.py. Uninformative checks MUST be able
    to spend accumulated evidence -- that is what pays for the anytime-valid
    guarantee.

    Firing LATCHES. Ville bounds the probability that the process EVER crosses
    the threshold, so "has it crossed" is the anytime-valid statement; whether it
    happens to be above the line at the moment you look is not. Without latching
    the same evidence reads as fired or not-fired depending on when the caller
    asks, which is exactly the estimator-variance sensitivity this replaces.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.log_threshold = float(np.log(1.0 / alpha))
        self.log_e = 0.0
        self.max_log_e = 0.0
        self.history = []

    def update(self, e):
        # Cap one check's contribution strictly below the firing threshold.
        log_e = float(np.log(max(e, 1e-12)))
        log_e = min(log_e, self.log_threshold - 1e-9)
        self.log_e += log_e
        self.max_log_e = max(self.max_log_e, self.log_e)
        self.history.append(e)
        return self.fired

    @property
    def fired(self):
        return self.max_log_e >= self.log_threshold

    @property
    def evidence(self):
        """Running product, for logging. Clipped to avoid overflow display."""
        return float(np.exp(min(self.log_e, 700.0)))

    def reset(self):
        self.log_e = 0.0
        self.max_log_e = 0.0


def _match_label(raw, labels):
    """First label mentioned in the reply, else None. Local copy of agent.py's
    cleaner: importing it here would be circular (agent imports this module)."""
    up = (raw or "").upper()
    hits = [(up.find(l.upper()), l) for l in labels if l.upper() in up]
    return min(hits)[1] if hits else None


def _validate_axiom(axiom, recent, client, model, labels, window=VALIDATION_WINDOW,
                    min_items=VALIDATION_MIN_ITEMS, extra_body=None):
    """Score a candidate axiom on the most RECENT trials before storing it.

    Returns (accepted, axiom_accuracy, baseline_accuracy).

    This free-function form is kept for backward compatibility with tests that
    import it directly.  The implementation lives in
    ``EigenMemoryKernel._validate_axiom_impl`` (below), which has access to
    ``self.client``, ``self.model``, etc. and therefore needs only the axiom
    text as an explicit argument.

    §9b: the crystallizer has no notion of when a rule stopped being true. The
    sequential trigger fired at batch 7 for a shift landing at batch 11 and wrote
    an accurate statement of the PRE-shift rule, false four batches later. A rule
    describing a superseded regime fails on recent data by construction, so
    scoring the tail is what separates "was correct" from "is correct".

    The bar is the agent's own recent hit rate, not the true rule -- there is no
    oracle at run time. An axiom costs context on every future call, so a tie
    goes to the status quo.

    Only the tail is scored: validating on the whole history would pass a rule
    that was right for most of the run even after the regime changed.

    Failures reject. Failing open would silently restore the unvalidated
    behaviour this exists to prevent.
    """
    return _validate_axiom_core(
        axiom, recent, client, model, labels, window, min_items, extra_body)


def _validate_axiom_core(axiom, recent, client, model, labels, window, min_items, extra_body):
    """Shared implementation for both the free function and the method."""
    tail = recent[-window:]
    if len(tail) < min_items:
        return False, 0.0, 0.0

    chance = 1.0 / max(len(labels), 1)
    baseline = max(sum(1 for r in tail if r["was_correct"]) / len(tail), chance)
    margin = float(np.sqrt(max(baseline * (1.0 - baseline), 0.01) / len(tail)))
    hits = 0
    preds = []
    for r in tail:
        prompt = (
            f"{axiom}\n\n"
            f"Apply the rule above to this input. Answer with exactly one of "
            f"{', '.join(labels)} and nothing else.\n\nInput: {r['input']}\nAnswer:"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, seed=0, max_tokens=8, extra_body=extra_body,
            )
            pred = _match_label(resp.choices[0].message.content, labels)
        except Exception as e:
            print(f"[EIGEN] axiom validation call failed: {e}; rejecting")
            return False, None, baseline
        preds.append(pred)
        if pred == r["actual"]:
            hits += 1
    acc = hits / len(tail)

    by_class = {}
    for r, pred in zip(tail, preds):
        c = by_class.setdefault(r["actual"], [0, 0, 0])
        c[0] += 1
        c[1] += (pred == r["actual"])
        c[2] += bool(r["was_correct"])
    for lab, (n_c, rule_hits, agent_hits) in by_class.items():
        if n_c < VALIDATION_MIN_CLASS_ITEMS:
            continue
        rule_acc = rule_hits / n_c
        bar = max(agent_hits / n_c, chance)
        if rule_acc + 1e-9 < bar:
            print(f"[EIGEN] axiom rejected on class {lab!r}: "
                  f"{rule_acc:.2f} vs bar {bar:.2f} "
                  f"(agent {agent_hits / n_c:.2f}, chance {chance:.2f})")
            return False, acc, baseline

    return acc > baseline + margin, acc, baseline


def crystallization_prompt(side_a, side_b, successes):
    """Task-neutral contrastive prompt. side_a/side_b: (input, predicted, actual)
    tuples from opposite extremes of the detected axis; successes: (input, actual)."""
    def fmt_f(rows):
        return "\n".join(
            f"- Input: {i} | I predicted: {p} | Actual: {a}" for i, p, a in rows
        )

    fmt_s = "\n".join(f"- Input: {i} | Actual: {a}" for i, a in successes) or "- (none available)"
    return f"""I am a classification agent making systematic mistakes. My failures vary along a single hidden axis. Below are failures from the two opposite ends of that axis, plus similar cases I got right.

Failures (side A of the axis):
{fmt_f(side_a)}

Failures (side B of the axis):
{fmt_f(side_b)}

Similar cases I got RIGHT:
{fmt_s}

Task: Inside a <thought> block, work out what single property distinguishes side A from side B, and how that property determines the correct label.

Your final line must be exactly:
RULE: [one concise, testable rule mapping the property to the labels]"""


class EigenMemoryKernel:
    def __init__(self, db_conn, openai_client, config=None, **kwargs):
        """Create a kernel from a KernelConfig or from keyword arguments.

        Both forms are supported: ``EigenMemoryKernel(conn, client, config=cfg)``
        and the original keyword form for backward compatibility with experiment
        scripts and tests.
        """
        if config is None:
            config = KernelConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass config OR keyword arguments, not both")

        self.config = config
        self.conn = db_conn
        self.client = openai_client
        self.model = config.model
        self.min_fail_residuals = config.min_fail_residuals
        self.min_succ_residuals = config.min_succ_residuals
        self.n_permutations = config.n_permutations
        self.stability_cos = config.stability_cos
        self.novelty_cos = config.novelty_cos
        self.rng = np.random.default_rng(config.rng_seed)
        self.extra_body = config.extra_body
        self.window = config.window
        self.contrast_on = config.contrast_on
        self.consecutive_detections = config.consecutive_detections
        self._t = 0
        self._detect_streak = 0

        # Residual records, split by outcome. Every retrieval-bearing trial feeds
        # these (NOT gated on surprise: the covariance contrast needs unbiased
        # samples of both outcomes).
        self.fail_records = []  # {residual, embedding, input, prediction, actual}
        self.succ_records = []  # {residual, embedding, input, actual}

        # Running mean of query embeddings, used to center projections when
        # selecting axioms at inference time.
        self._embed_sum = None
        self._embed_count = 0

        # Trigger state: last observed top direction (full space) and the
        # directions already crystallized.
        self.prev_direction = None
        self.consumed_directions = []

        # Telemetry: (lambda1, permutation edge) per check, and top-3 eigenvalue
        # shares per check for the spectrum heatmap in scripts/analysis/plot_results.py.
        self.detectability_history = []
        # Sequential (e-value) trigger. Off by default so every committed result
        # stays reproducible; the streak rule remains the as-run mechanism.
        self.sequential_gate = config.sequential_gate
        self.n_permutations_seq = config.n_permutations_seq
        self._eproc = _EProcess(alpha=config.e_process_alpha)
        self.evalue_history = []
        # Validate a candidate axiom against recent trials before storing it.
        # Off by default: it changes what gets written, so every committed
        # result stays reproducible. See §9b.
        self.validate_axioms = config.validate_axioms
        self.labels = config.labels
        self.recent_trials = []
        self.validation_history = []
        # Re-validate stored axioms and retire the ones that stopped being true.
        # Off by default alongside the rest of §9b/§9c.
        self.retire_stale_axioms = config.retire_stale_axioms
        self.retirement_history = []
        # Outcome-stream change trigger (§9d), off by default.
        self.outcome_trigger = config.outcome_trigger
        self.outcome_change_detected = False
        self.outcome_change_at = None
        self.outcome_change_reason = None
        self._outcome_n = 0
        self._outcome_hits = 0
        self._outcome_seen_labels = set()
        self._outcome_logw = 0.0
        self._outcome_consumed = False
        # Formation readiness: detection is immediate, rule WRITING waits for
        # enough post-change evidence. v3 crystallized two batches after the
        # trigger, against a window still dominated by pre-shift trials, and
        # wrote the pre-shift rule.
        self.formation_min_post_change = config.formation_min_post_change
        self._post_change_observed = 0
        # Drop pre-change records from the contrast window when a change is
        # detected. Pre-change SUCCESSES are successes under the old rule, and
        # they pull the success mean toward the old regime -- which is what lets
        # the crystallizer write the pre-shift rule back out mid-stream.
        self.truncate_at_change = config.truncate_at_change
        # axiom text -> its consumed direction, so retiring a rule can release
        # the axis. Without this the novelty gate would retire the wrong rule
        # and then forbid the right one on the same structure.
        self.axiom_directions = {}
        self.spectrum_history = []

    # ------------------------------------------------------------------ ingest

    def observe(self, embedding, residual, was_correct, context_input, prediction, actual):
        """Record one trial's retrieval residual, keyed on outcome (correctness),
        not on surprise.  Delegates to: append, trigger, prune."""
        embedding = np.asarray(embedding, dtype=float)
        residual = np.asarray(residual, dtype=float)
        if self._embed_sum is None:
            self._embed_sum = np.zeros_like(embedding)
        self._embed_sum += embedding
        self._embed_count += 1

        self._t += 1
        rec = {
            "residual": residual,
            "embedding": embedding,
            "input": context_input,
            "prediction": prediction,
            "actual": actual,
            "t": self._t,
        }
        (self.succ_records if was_correct else self.fail_records).append(rec)
        self.recent_trials.append(
            {"input": context_input, "actual": actual, "was_correct": was_correct})

        self._process_outcome_trigger(was_correct, actual)
        self._prune_window()

    def _process_outcome_trigger(self, was_correct, actual):
        """Update the outcome-stream change detector and handle truncation."""
        if not self.outcome_trigger:
            return
        was_detected = self.outcome_change_detected
        self._update_outcome_trigger(was_correct, actual)
        if was_detected and self.outcome_change_detected:
            self._post_change_observed += 1
        elif self.outcome_change_detected and not was_detected:
            self._post_change_observed = 1
            if self.truncate_at_change:
                dropped = len(self.fail_records) + len(self.succ_records) - 1
                cutoff = self._t - 1
                self.fail_records = [r for r in self.fail_records
                                     if r["t"] > cutoff]
                self.succ_records = [r for r in self.succ_records
                                     if r["t"] > cutoff]
                print(f"[EIGEN] change-point truncation: dropped {dropped} "
                      f"pre-change records from the contrast window")

    def _prune_window(self):
        """Keep the contrast buffers and recent-trial tail bounded."""
        if len(self.recent_trials) > 4 * VALIDATION_WINDOW:
            self.recent_trials = self.recent_trials[-4 * VALIDATION_WINDOW:]
        if self.window:
            cutoff = self._t - self.window
            self.fail_records = [r for r in self.fail_records if r["t"] > cutoff]
            self.succ_records = [r for r in self.succ_records if r["t"] > cutoff]

    def embedding_mean(self):
        if not self._embed_count:
            return None
        return self._embed_sum / self._embed_count

    # ---------------------------------------------------------------- spectral

    def _reduced_residuals(self):
        """Project pooled residuals onto their top principal components.

        Returns (fail_reduced, succ_reduced, basis) with basis rows orthonormal in
        the full space, so directions can be mapped back via basis.T @ v.
        """
        key = "embedding" if self.contrast_on == "embedding_mean" else "residual"
        F = np.array([r[key] for r in self.fail_records])
        S = np.array([r[key] for r in self.succ_records])
        pooled = np.vstack([F, S])
        r = min(R_COMPONENTS, pooled.shape[0] - 1, pooled.shape[1])
        pca = PCA(n_components=r).fit(pooled)
        basis = pca.components_
        return F @ basis.T, S @ basis.T, basis

    def _permutation_edge(self, F_red, S_red):
        """Noise edge: max top-eigenvalue over random fail/success relabelings."""
        pooled = np.vstack([F_red, S_red])
        n_fail = len(F_red)
        edge = 0.0
        for _ in range(self.n_permutations):
            idx = self.rng.permutation(len(pooled))
            if self.contrast_on == "embedding_mean":
                lam, _ = _mean_contrast(pooled[idx[:n_fail]], pooled[idx[n_fail:]])
            else:
                lam, _, _ = _top_contrast_eig(pooled[idx[:n_fail]], pooled[idx[n_fail:]])
            edge = max(edge, lam)
        return edge

    def check_and_crystallize(self):
        """Run once per batch: eigenanalysis + the detectability/stability/novelty
        gates. Crystallizes at most one axiom per call."""
        min_fail = (MIN_FAIL_RESIDUALS_SEQ if self.sequential_gate
                    else self.min_fail_residuals)
        if (
            len(self.fail_records) < min_fail
            or len(self.succ_records) < self.min_succ_residuals
        ):
            return

        F_red, S_red, basis = self._reduced_residuals()
        if self.contrast_on == "embedding_mean":
            lam1, v_red = _mean_contrast(F_red, S_red)
            w = np.array([lam1])
        else:
            lam1, v_red, w = _top_contrast_eig(F_red, S_red)
        edge = self._permutation_edge(F_red, S_red)
        v_full = _unit(basis.T @ v_red)

        self.detectability_history.append((lam1, edge))
        top3 = np.sort(np.clip(w, 0, None))[::-1][:3]
        total = top3.sum()
        self.spectrum_history.append((top3 / total if total > 0 else top3).tolist())

        detectable = lam1 > edge
        self._detect_streak = self._detect_streak + 1 if detectable else 0
        stable = self.prev_direction is not None and (
            abs(float(v_full @ self.prev_direction)) > self.stability_cos
        )
        novel = all(
            abs(float(v_full @ c)) < self.novelty_cos for c in self.consumed_directions
        )
        self.prev_direction = v_full

        if self.outcome_trigger:
            # Detection comes from the outcome stream; the spectral axis is
            # demoted to what it is actually good at -- picking the contrast
            # examples for the prompt. Crystallize once per detected change,
            # with stability/novelty still applying so a noise axis is not used.
            triggered = (self.outcome_change_detected
                         and not self._outcome_consumed
                         and self.formation_ready)
            print(f"[EIGEN-OUT] lam1={lam1:.3f} "
                  f"change={'YES@' + str(self.outcome_change_at) if self.outcome_change_detected else 'no'} "
                  f"({'FIRE' if triggered else 'hold'}, "
                  f"{'stable' if stable else 'unstable'}, "
                  f"{'novel' if novel else 'consumed'}) "
                  f"fail={len(self.fail_records)} succ={len(self.succ_records)}")
            strength = lam1 / edge if edge > 0 else 1.0
        elif self.sequential_gate:
            # Accumulate evidence instead of re-testing from scratch. See
            # _EProcess / _evalue_from_null for why this replaces both the
            # permutation-quantile threshold and the consecutive-detection count.
            e = _evalue_from_null(
                F_red, S_red, self.n_permutations_seq, self.rng,
                contrast="mean" if self.contrast_on == "embedding_mean" else "cov",
            )
            self._eproc.update(e)
            self.evalue_history.append(e)
            triggered = self._eproc.fired
            print(
                f"[EIGEN-SEQ] lam1={lam1:.3f} e={e:.2f} E={self._eproc.evidence:.1f} "
                f"(need {1.0 / self._eproc.alpha:.0f}, {'FIRED' if triggered else 'accumulating'}, "
                f"{'stable' if stable else 'unstable'}, {'novel' if novel else 'consumed'}) "
                f"fail={len(self.fail_records)} succ={len(self.succ_records)}"
            )
            strength = self._eproc.evidence
        else:
            triggered = (detectable
                         and self._detect_streak >= self.consecutive_detections)
            print(
                f"[EIGEN] lam1={lam1:.3f} edge={edge:.3f} "
                f"({'detectable' if detectable else 'below edge'}, "
                f"streak={self._detect_streak}, "
                f"{'stable' if stable else 'unstable'}, "
                f"{'novel' if novel else 'consumed'}) "
                f"fail={len(self.fail_records)} succ={len(self.succ_records)}"
            )
            strength = lam1 / edge if edge > 0 else 1.0

        if triggered and stable and novel:
            self._crystallize(v_full, strength=strength)
            if self.sequential_gate:
                # Evidence for THIS direction is spent; a further axiom must earn
                # its own. Without this the latched process fires every check.
                self._eproc.reset()
            if self.outcome_trigger:
                # One crystallization per detected change. The trigger latches
                # (the rule really did change), so without this it would fire on
                # every subsequent check. A LATER change re-arms it below.
                self._outcome_consumed = True

    # ---------------------------------------------------------- axiom validation

    def _validate_axiom_impl(self, axiom_text):
        """Score a candidate axiom against recent trials using self's state."""
        labels = self.labels or sorted(
            {r["actual"] for r in self.recent_trials if r.get("actual")})
        return _validate_axiom_core(
            axiom_text, self.recent_trials, self.client, self.model, labels,
            VALIDATION_WINDOW, VALIDATION_MIN_ITEMS, self.extra_body)

    # ----------------------------------------------------------- crystallization

    def _contrast_sets(self, v_full):
        """Failures at extreme opposite projections along v_full, plus successes
        matched to them by embedding similarity.

        Project the vector the AXIS was derived from. `_reduced_residuals` keys
        on "embedding" in embedding_mean mode and "residual" otherwise, so
        projecting residuals unconditionally ordered failures by a quantity
        largely unrelated to the axis whenever contrast_on="embedding_mean" --
        the Rule-Shift configuration. Measured on the real seed windows: the two
        orderings correlate 0.11-0.46, and on seeds 23 and 42 the residual
        ordering returned six contrast examples ALL sharing one label, i.e. a
        prompt asserting a contrast its own examples do not contain.
        """
        key = "embedding" if self.contrast_on == "embedding_mean" else "residual"
        proj = np.array([float(r[key] @ v_full) for r in self.fail_records])
        order = np.argsort(proj)
        side_a = [self.fail_records[i] for i in order[:N_CONTRAST_PER_SIDE]]
        side_b = [self.fail_records[i] for i in order[::-1][:N_CONTRAST_PER_SIDE]]

        successes = []
        if self.succ_records:
            succ_emb = np.array([_unit(r["embedding"]) for r in self.succ_records])
            used = set()
            for f in side_a + side_b:
                sims = succ_emb @ _unit(f["embedding"])
                for j in np.argsort(sims)[::-1]:
                    if j not in used:
                        used.add(int(j))
                        successes.append(self.succ_records[j])
                        break
                if len(successes) >= N_MATCHED_SUCCESSES:
                    break
        return side_a, side_b, successes

    @property
    def formation_ready(self):
        """Whether enough post-change evidence exists to WRITE a rule.

        Always True when the outcome trigger is off: the streak-rule and
        sequential paths have no change point, so there is nothing to wait for
        and their behaviour must be unchanged.
        """
        if not self.outcome_trigger:
            return True
        if not self.outcome_change_detected:
            return False
        return self._post_change_observed >= self.formation_min_post_change

    def _update_outcome_trigger(self, was_correct, actual):
        """Detect the rule change from outcomes rather than embedding geometry.

        The spectral statistic on these streams sits at 0.78-1.28x a threshold
        that itself varies 1.31x on identical data, and fires 0/5. The outcome
        stream carries the same event at far higher SNR: accuracy drops
        0.12-0.26 on every seed at the shift.

        Two signals, both checkable every trial:

          - an unseen label. Under a stable rule the label set is closed, so a
            label that never occurred before IS the change. On this task the
            shift introduces exactly that, which is where most of the detection
            power comes from -- a shift PERMUTING existing labels would not
            produce it and would fall back to the e-process below.
          - a betting e-process against "the hit rate is still p0", with p0 the
            burn-in rate. Ville bounds the all-time false-fire probability, so
            checking every trial costs nothing.

        Latches: once the rule has changed it has changed, and the agent
        adapting afterwards must not un-fire it.
        """
        # NB this counts OBSERVED trials, which lag stream position: agent.py
        # skips observe() when a trial has no retrieval neighbour (empty buffer
        # early on), so on seed 42 kernel trial 91 is stream trial 100. Read
        # outcome_change_at as "after N observed trials", not as a stream index.
        self._outcome_n += 1
        if self._outcome_n <= OUTCOME_BURN_IN:
            self._outcome_seen_labels.add(actual)
            self._outcome_hits += bool(was_correct)
            return
        if self.outcome_change_detected:
            # Already fired and already crystallized: re-arm so a SECOND change
            # later in the stream can be detected too. The label set is
            # refreshed to the post-change regime, and the e-process restarts.
            if self._outcome_consumed:
                self._outcome_seen_labels.add(actual)
                self.outcome_change_detected = False
                self._outcome_consumed = False
                self._outcome_logw = 0.0
            return

        if actual not in self._outcome_seen_labels:
            # Record it NOW. Without this the label stays "unseen" and every
            # later occurrence re-triggers detection once the first axiom is
            # stored and the trigger re-arms -- observed live on seed 42, which
            # reported a second "unseen label 'DEFER'" eleven trials after the
            # first, for a label it had already fired on.
            self._outcome_seen_labels.add(actual)
            self.outcome_change_detected = True
            self.outcome_change_at = self._outcome_n
            self.outcome_change_reason = f"unseen label {actual!r}"
            print(f"[OUTCOME] change detected at trial {self._outcome_n}: "
                  f"{self.outcome_change_reason}")
            return

        # Bet a fraction OUTCOME_BET of capital that the failure rate has RISEN.
        # Payoff (1 + b) on a failure, (1 - b * q0/p0) on a success, which has
        # mean exactly 1 under the null -- a valid e-value, so Ville bounds the
        # all-time false-fire probability by alpha.
        #
        # An earlier form multiplied the whole bet by q0, which silently zeroed
        # the update when the burn-in was perfect (p0 = 1 -> q0 = 0 -> factor
        # identically 1.0). The detector then could not fire on a total accuracy
        # collapse -- the case it exists for. Clamp p0 instead so a perfect
        # burn-in leaves a usable null.
        p0 = min(max(self._outcome_hits / OUTCOME_BURN_IN, 1e-3), 0.99)
        q0 = 1.0 - p0
        factor = (1.0 + OUTCOME_BET) if not was_correct else (
            1.0 - OUTCOME_BET * q0 / p0)
        self._outcome_logw = max(
            self._outcome_logw + float(np.log(max(factor, 1e-12))), 0.0)
        if self._outcome_logw >= np.log(1.0 / E_PROCESS_ALPHA):
            self.outcome_change_detected = True
            self.outcome_change_at = self._outcome_n
            self.outcome_change_reason = (
                f"accuracy decay (baseline {p0:.2f})")
            print(f"[OUTCOME] change detected at trial {self._outcome_n}: "
                  f"{self.outcome_change_reason}")

    @staticmethod
    def axiom_select_sql():
        """The SELECT the agent uses to fetch injectable axioms.

        Lives here so the retirement filter cannot drift away from the
        retirement writer. Retired rows stay in the table as the record of what
        the agent believed; they are excluded from injection, not deleted.
        """
        return ("SELECT axiom_content, eigen_vector FROM semantic_core "
                "WHERE NOT COALESCE(retired, FALSE)")

    def revalidate_axioms(self):
        """Re-score stored axioms against recent trials; retire the stale ones.

        Validation at write time cannot catch a rule that is true when written
        and false later -- seed 42's "pending -> FILE" scored 0.80 vs 0.70
        honestly at batch 7 for a shift that landed at batch 11. Asking the same
        question again once the regime has moved is what catches it.

        Failures do NOT retire. An unreachable model is not evidence a rule went
        stale, and retiring on error would delete working memory on any blip.
        This is the opposite of the write path, where rejecting on error is the
        safe direction.
        """
        if not self.retire_stale_axioms:
            return []
        tail = self.recent_trials[-VALIDATION_WINDOW:]
        if len(tail) < VALIDATION_MIN_ITEMS:
            return []

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, axiom_content FROM semantic_core "
                            "WHERE NOT COALESCE(retired, FALSE)")
                rows = cur.fetchall()
        except Exception as e:
            print(f"[EIGEN] axiom re-validation SELECT failed: {e}")
            return []

        retired = []
        for axiom_id, content in rows:
            ok, acc, base = self._validate_axiom_impl(content)
            # acc is None when the rule could not be scored at all (model
            # unreachable). Only retire when it actually answered and lost.
            if ok or acc is None or acc >= base:
                continue
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE semantic_core SET retired = TRUE, "
                        "retired_at = NOW() WHERE id = %s", (axiom_id,))
                self.conn.commit()
            except Exception as e:
                print(f"[EIGEN] retiring axiom {axiom_id} failed: {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                continue
            retired.append(axiom_id)
            # Release the axis so the CORRECTED rule for this same failure
            # structure can crystallize: it lies along essentially the same
            # direction, and the novelty gate would otherwise forbid it.
            v = self.axiom_directions.pop(axiom_id, None)
            if v is None:
                v = self.axiom_directions.pop(content, None)
            if v is not None:
                v = np.asarray(v, dtype=float)
                self.consumed_directions = [
                    c for c in self.consumed_directions
                    if abs(float(v @ c)) <= self.novelty_cos
                ]
            self.retirement_history.append(
                {"axiom": content, "accuracy": acc, "baseline": base})
            print(f"[EIGEN] axiom RETIRED ({acc:.2f} vs baseline {base:.2f}): "
                  f"{content[:70]}")
        return retired

    def _crystallize(self, v_full, strength=1.0):
        """Translate the detected axis into a linguistic rule via contrastive
        introspection, and store it with its (full-space) direction."""
        side_a, side_b, successes = self._contrast_sets(v_full)
        prompt = crystallization_prompt(
            [(r["input"], r["prediction"], r["actual"]) for r in side_a],
            [(r["input"], r["prediction"], r["actual"]) for r in side_b],
            [(r["input"], r["actual"]) for r in successes],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=0,
                max_tokens=400,
                extra_body=self.extra_body,
            )
            raw = response.choices[0].message.content or ""
            if "RULE:" not in raw:
                # The whole budget went into the <thought> block (seed-42 shift
                # pilot: a 1.4k-char truncated CoT was stored and injected —
                # the truncation cousin of the original full-CoT-axiom bug).
                # One follow-up asking for the final line alone.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "Now give ONLY your final "
                         "line, in the exact form 'RULE: ...' — no other text."},
                    ],
                    temperature=0.0,
                    seed=0,
                    max_tokens=120,
                    extra_body=self.extra_body,
                )
                raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[EIGEN] crystallization LLM call failed: {e}")
            return

        # Store ONLY the final RULE: line. The <thought> block is scaffolding for
        # the introspection call; storing the full reply used to inject ~1.2k
        # chars of CoT rambling into every context the axiom was selected for —
        # sabotaging the very arm under test.
        #
        # Take the first LINE after the last "RULE:", not the whole suffix.
        # rpartition(...)[2] kept everything downstream, so a reply that stated
        # the rule and then kept reasoning ("Wait, let me refine that...") stored
        # the rule PLUS ~250 chars of trailing CoT — the truncation cousin of the
        # original bug, and what the seed-42 Rule-Shift pilot actually stored.
        rule = raw.rpartition("RULE:")[2].strip().splitlines()[0].strip() if "RULE:" in raw else ""
        if not rule or "<thought" in rule:
            # Never store scaffolding as memory. The axis stays unconsumed, so
            # crystallization retries at the next check.
            print("[EIGEN] no clean RULE line after retry; axiom NOT stored")
            return
        axiom = f"RULE: {rule}"

        if self.validate_axioms:
            ok, acc, base = self._validate_axiom_impl(axiom)
            self.validation_history.append(
                {"axiom": axiom, "accuracy": acc, "baseline": base, "accepted": ok})
            print(f"[EIGEN] axiom validation: {acc:.2f} vs baseline {base:.2f} "
                  f"-> {'ACCEPT' if ok else 'REJECT'}")
            if not ok:
                # Leave the axis UNCONSUMED: the direction may be real and simply
                # early (§9b's batch-7 fire for a batch-11 shift). Rejecting the
                # text without burning the direction lets a later check retry once
                # the recent evidence has caught up.
                return

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_core (axiom_content, eigen_vector, strength_score)
                    VALUES (%s, %s, %s)
                    """,
                    (axiom, v_full.tolist(), float(strength)),
                )
            self.conn.commit()
        except Exception as e:
            # Roll back so the connection is not left in an aborted-transaction
            # state that would poison every later DB call on it. The axis stays
            # unconsumed, so crystallization retries at the next check.
            self.conn.rollback()
            print(f"[EIGEN] axiom store failed: {e}")
            return

        self.consumed_directions.append(v_full)
        # Remember which axis this axiom claimed, so retiring it can release the
        # axis for a corrected rule on the same failure structure.
        self.axiom_directions[axiom] = v_full
        print(f"[AXIOM+] {axiom[:80].strip()}...")

    # ------------------------------------------------------------- inference use

    def score_axioms(self, query_embedding, axiom_rows):
        """Relevance of stored axioms to a query: |projection| of the centered
        query embedding onto each axiom's axis. Sign-invariant, unlike the old
        cosine-to-eigenvector ranking (an eigenvector's sign is arbitrary).

        axiom_rows: iterable of (axiom_content, eigen_vector) where eigen_vector
        may be a pgvector text literal or a sequence. Returns rows sorted by
        descending relevance as (score, axiom_content).
        """
        mu = self.embedding_mean()
        q = np.asarray(query_embedding, dtype=float)
        centered = q - mu if mu is not None else q
        scored = []
        for content, vec in axiom_rows:
            if isinstance(vec, str):
                # pgvector's text literal ("[0.1,0.2,...]") happens to be valid
                # JSON — a format coincidence, named here so it reads as a choice.
                vec = json.loads(vec)
            v = np.asarray(vec, dtype=float)
            scored.append((abs(float(centered @ v)), content))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored
