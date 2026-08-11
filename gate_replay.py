"""Replay the crystallization gate against LIVE correctness (not the proxy).

The §9 ungated ablation showed all four gate-shut seeds carry recoverable
signal — but it had to reconstruct which trials failed, using a stale-copier
proxy, because per-trial live correctness was never persisted. That proxy is
*cleaner* than live reality: live failures include ordinary executor mistakes,
which are noise rather than rule-shift signal. So the ablation established that
the signal exists without establishing WHY the live gate missed it. Two
candidate causes remain entangled:

  (a) featurization — the estimator contrasts raw embedding means, and that
      stream carries the shift weakly; or
  (b) label noise — the fail/succ split the gate is handed is polluted by
      executor error, blurring a contrast that is otherwise there.

This script separates them. Runs with `trial_correct` in their artifact can be
replayed with the REAL split, holding featurization fixed. Comparing the three
statistics answers it directly:

  live λ₁/edge   ≈ replay λ₁/edge  <  proxy λ₁/edge   -> (a) featurization
  live λ₁/edge   <  replay λ₁/edge ≈  proxy λ₁/edge   -> (b) label noise

Needs Ollama (embeddinggemma) to rebuild embeddings; no executor calls, so it
is cheap. Artifacts without `trial_correct` are skipped with a message — only
runs made after that field was added can be replayed.

Usage: uv run python gate_replay.py [seed ...]   (default: every shift seed)
Writes results/shift/gate_replay.<seed>.json per seed.
"""

import json
import sys

import numpy as np
from openai import OpenAI

from gate_roc import _NullConn
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_shift, shift_rules
from src.eigen_memory_agent.memory_kernel import (
    EigenMemoryKernel, _mean_contrast,
)
from src import paths

KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}
SEEDS = [int(a) for a in sys.argv[1:]] or [2, 7, 18, 23, 42]


def _statistic(client, trials, embs, correct_fn, seed):
    """Build a kernel over the stream and return (lam1, edge) at end-of-run."""
    conn = _NullConn()
    kernel = EigenMemoryKernel(conn, client, rng_seed=seed, **KCFG)
    for i, t in enumerate(trials):
        if i == 0:
            continue  # no earlier neighbour to copy from
        nn = int(np.argmax(embs[:i] @ embs[i]))
        residual = embs[i] - embs[nn]
        kernel.observe(embedding=embs[i], residual=residual,
                       was_correct=correct_fn(i, nn, t),
                       context_input=t["input"],
                       prediction=trials[nn]["label"], actual=t["label"])
    if (len(kernel.fail_records) < kernel.min_fail_residuals
            or len(kernel.succ_records) < kernel.min_succ_residuals):
        return None, None, len(kernel.fail_records), len(kernel.succ_records)
    F_red, S_red, _ = kernel._reduced_residuals()
    lam1, _ = _mean_contrast(F_red, S_red)
    edge = kernel._permutation_edge(F_red, S_red)
    return lam1, edge, len(kernel.fail_records), len(kernel.succ_records)


def run_seed(client, seed):
    art = paths.shift(f"comparison_results.shift.{seed}.json")
    with open(art) as f:
        run = json.load(f)
    eigen = run.get("arms", {}).get("Treatment_Eigen", {})
    live_correct = eigen.get("trial_correct")
    if not live_correct:
        print(f"seed {seed}: no `trial_correct` in artifact — run predates the "
              f"field; skipping (rerun the arm to replay it)")
        return None

    trials, _ = load_shift(seed=seed)
    pre_rule, post_rule = shift_rules(seed)
    print(f"\n===== seed {seed}  pre={pre_rule}  post={post_rule} =====")
    if len(live_correct) != len(trials):
        print(f"  WARNING: {len(live_correct)} correctness flags vs "
              f"{len(trials)} trials — replaying over the overlap")

    embs = np.array([
        client.embeddings.create(input=t["input"], model=EMBEDDING_MODEL)
        .data[0].embedding for t in trials
    ])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    # Proxy split (what the ablation used) vs live split (what actually happened).
    proxy = lambda i, nn, t: trials[nn]["label"] == t["label"]
    live = lambda i, nn, t: bool(live_correct[i]) if i < len(live_correct) else True

    p_lam, p_edge, p_f, p_s = _statistic(client, trials, embs, proxy, seed)
    l_lam, l_edge, l_f, l_s = _statistic(client, trials, embs, live, seed)

    def _fmt(lam, edge):
        if lam is None:
            return "insufficient window"
        return f"lam1={lam:.4f} edge={edge:.4f} ratio={lam / edge:.2f}"

    print(f"  proxy split (stale-copier): {_fmt(p_lam, p_edge)}  fail={p_f} succ={p_s}")
    print(f"  live  split (as measured):  {_fmt(l_lam, l_edge)}  fail={l_f} succ={l_s}")

    live_det = eigen.get("detectability") or []
    last = live_det[-1] if live_det else None
    if last:
        print(f"  live run's own last check:  lam1={last[0]:.4f} edge={last[1]:.4f} "
              f"ratio={last[0] / last[1]:.2f}")

    out = {"seed": seed, "kernel_cfg": KCFG,
           "proxy": {"lam1": p_lam, "edge": p_edge,
                     "ratio": (p_lam / p_edge) if p_lam else None,
                     "n_fail": p_f, "n_succ": p_s},
           "live": {"lam1": l_lam, "edge": l_edge,
                    "ratio": (l_lam / l_edge) if l_lam else None,
                    "n_fail": l_f, "n_succ": l_s},
           "live_run_last_check": last,
           "n_axioms_live_run": eigen.get("n_axioms")}
    with open(paths.shift(f"gate_replay.{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                    timeout=120.0, max_retries=3)
    done = [r for r in (run_seed(client, s) for s in SEEDS) if r]
    if not done:
        print("\nNothing replayed: no artifact carries `trial_correct` yet. That "
              "field is written by run_shift_experiment.py from this commit on, "
              "so a seed must be rerun before its gate can be replayed.")
        return
    print("\nReading: if the live split's ratio tracks the PROXY's, the gate was "
          "starved by a noisy correctness signal. If it tracks the LIVE RUN's "
          "own checks, the featurization is the bottleneck.")


if __name__ == "__main__":
    main()
