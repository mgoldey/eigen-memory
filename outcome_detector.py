"""Detect the rule shift from the CORRECTNESS stream instead of embedding space.

The spectral gate watches a 768-d contrast statistic that on real seeds sits at
0.78-1.28x a permutation threshold which itself varies 1.31x on identical data
(docs/NEXT_EXPERIMENT.md §9a). Meanwhile the outcome stream carries the same
event at far higher SNR and is now persisted as `trial_correct`: under stale
copying the shifted class stops being answered correctly almost entirely, while
the unshifted class is untouched.

This script asks whether that is enough, offline, with no LLM calls:

  - Does a change detector on outcomes fire on all five seeds?
  - Does it stay silent over the pre-shift stretch (the false-fire budget)?
  - How many batches after the shift does it take?
  - Does it locate the change point well enough to truncate the crystallizer's
    window to post-shift evidence only?

Two detectors, both anytime-valid so "checked every trial" costs nothing:

  GLOBAL   a betting e-process on the overall correctness stream.
  PER-LABEL the same, run separately per true label, firing when ANY label's
            process fires. The shift moves ONE class, so pooling dilutes it --
            this is the version the design predicts should win.

Usage: uv run python outcome_detector.py [seed ...]     (default: all five)
Writes results/shift/outcome_detection.json
"""

import json
import sys

import numpy as np

from src.dataset import load_shift, shift_rules
from src import paths

SEEDS = [int(a) for a in sys.argv[1:]] or [2, 7, 18, 23, 42]
SHIFT_TRIAL = 100          # rule changes here, by construction
ALPHA = 0.05
BURN_IN = 30               # trials used to estimate the pre-shift rate


def _eprocess(outcomes, p0, alpha=ALPHA, lam=0.5):
    """Betting e-process against 'success rate is still p0'.

    Bets a fraction lam of capital that each trial fails. Wealth grows when
    failures arrive faster than p0 predicts. Ville's inequality bounds the
    all-time false-fire probability by alpha, so this can be checked after every
    single trial rather than once per batch -- which is the property the
    3-consecutive-detections rule was hand-approximating.

    Returns (fired_at_index or None, wealth trace).
    """
    q0 = 1.0 - p0                       # expected failure rate under the null
    logw, trace, fired = 0.0, [], None
    for i, ok in enumerate(outcomes):
        # payoff > 1 on a failure, < 1 on a success, mean 1 under the null
        x = (0.0 if ok else 1.0)
        factor = 1.0 + lam * ((x - q0) / max(q0 * (1 - q0), 1e-6)) * q0
        logw += np.log(max(factor, 1e-12))
        logw = max(logw, 0.0)           # reset to 1: change DETECTION, not testing
        trace.append(logw)
        if fired is None and logw >= np.log(1.0 / alpha):
            fired = i
    return fired, trace


def run_seed(seed):
    art = paths.shift(f"comparison_results.shift.{seed}.json")
    with open(art) as f:
        eigen = json.load(f)["arms"]["Treatment_Eigen"]
    correct = eigen.get("trial_correct")
    if not correct:
        print(f"seed {seed}: no trial_correct; skipping")
        return None

    trials, _ = load_shift(seed=seed)
    n = min(len(correct), len(trials))
    correct, trials = correct[:n], trials[:n]
    labels = [t["label"] for t in trials]
    pre, post = shift_rules(seed)

    # Null rate from the pre-shift burn-in only: the agent's own early hit rate.
    p0 = max(sum(correct[:BURN_IN]) / BURN_IN, 1e-3)

    fired_g, _ = _eprocess(correct[BURN_IN:], p0)
    fired_g = None if fired_g is None else fired_g + BURN_IN

    # Per-label. Note the shifted class's NEW label has no pre-shift examples at
    # all -- on this task the shift introduces a label that never occurred
    # before (seed 2: request FILE -> DEFER, and DEFER is absent pre-shift). A
    # first-version of this detector required a burn-in per label and therefore
    # skipped exactly the class that changes, watching only the untouched one.
    # It fired 0/5. Treat an unseen label as its own detection: under a stable
    # rule the label set is closed, so a new one appearing IS the change.
    per = {}
    novel_at = None
    seen = set(labels[:BURN_IN])
    for i in range(BURN_IN, n):
        if labels[i] not in seen:
            novel_at = i
            break
    for lab in sorted(set(labels)):
        idx = [i for i in range(n) if labels[i] == lab]
        burn = [i for i in idx if i < BURN_IN]
        rest = [i for i in idx if i >= BURN_IN]
        if len(burn) < 5 or len(rest) < 5:
            continue
        p0_l = max(sum(correct[i] for i in burn) / len(burn), 1e-3)
        f, _ = _eprocess([correct[i] for i in rest], p0_l)
        per[lab] = None if f is None else rest[f]
    candidates = [v for v in per.values() if v is not None]
    if novel_at is not None:
        candidates.append(novel_at)
    fired_p = min(candidates, default=None)

    def _fmt(x):
        if x is None:
            return "never"
        return f"trial {x} ({'PRE-shift!' if x < SHIFT_TRIAL else f'+{x - SHIFT_TRIAL} after'})"

    print(f"\n=== seed {seed}  (pre-shift hit rate {p0:.2f}) ===")
    print(f"  global   : {_fmt(fired_g)}")
    print(f"  per-label: {_fmt(fired_p)}   " +
          " ".join(f"{k}={'-' if v is None else v}" for k, v in per.items()))

    return {"seed": seed, "p0": p0, "global_fired": fired_g,
            "per_label_fired": fired_p, "per_label": per,
            "novel_label_at": novel_at,
            "shift_trial": SHIFT_TRIAL, "n_trials": n,
            "pre_rule": pre, "post_rule": post}


def main():
    rows = [r for r in (run_seed(s) for s in SEEDS) if r]
    if not rows:
        print("nothing to do")
        return

    print(f"\n{'seed':>5} {'global':>22} {'per-label':>22}")
    for r in rows:
        def f(x):
            if x is None:
                return "never"
            return f"{x} ({'PRE' if x < SHIFT_TRIAL else '+' + str(x - SHIFT_TRIAL)})"
        print(f"{r['seed']:>5} {f(r['global_fired']):>22} {f(r['per_label_fired']):>22}")

    for name, key in (("global", "global_fired"), ("per-label", "per_label_fired")):
        fired = [r for r in rows if r[key] is not None]
        false_fires = [r for r in fired if r[key] < SHIFT_TRIAL]
        delays = [r[key] - SHIFT_TRIAL for r in fired if r[key] >= SHIFT_TRIAL]
        print(f"\n{name}: fired {len(fired)}/{len(rows)}, "
              f"pre-shift false fires {len(false_fires)}, "
              f"median delay {int(np.median(delays)) if delays else '--'} trials "
              f"({(np.median(delays) / 10):.1f} batches)" if delays else
              f"\n{name}: fired {len(fired)}/{len(rows)}, "
              f"pre-shift false fires {len(false_fires)}")

    with open(paths.shift("outcome_detection.json"), "w") as f:
        json.dump({"alpha": ALPHA, "burn_in": BURN_IN, "seeds": rows}, f, indent=2)
    print("\nwrote results/shift/outcome_detection.json")


if __name__ == "__main__":
    main()
