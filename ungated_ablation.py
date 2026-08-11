"""Ungated-trigger ablation (docs/NEXT_EXPERIMENT.md §8 follow-up, protocol §9).

The gate-v2 calibration concluded the four gate-shut replication seeds are
signal-starved: their mean-contrast statistic sits at noise level (lam/edge
0.81-0.85 vs noise 0.87), so no gate honoring a false-fire budget fires. This
ablation is the arbiter for that conclusion's sim-to-live mapping: bypass the
gate entirely, force one crystallization from each shut seed's end-of-run
window, and G4-score the rule text against the planted post-shift rule.

  - Correct rule  -> the signal WAS in the episodes and the estimator/
                     featurization missed it (featurization work follows).
  - Garbage rule  -> the calibration called it right; the windows genuinely
                     don't separate failure from success.

Reconstruction caveat (pre-registered): per-trial live correctness was not
persisted by the harness, so was_correct uses a STALE-COPIER PROXY — a trial
is a failure iff the similarity-nearest EARLIER trial's stored label differs
from the era-correct label. This mirrors the copying failure mode the design
plants (post-shift request rows flip; reports stay stable) and matches the
observed Treatment post-adapt accuracy (~0.42-0.45). Residuals are rebuilt the
same way (query embedding minus nearest earlier embedding). The readout is the
rule TEXT, printed beside the planted pre/post rules for G4 scoring — accuracy
impact is deliberately out of scope until the rule-text verdict is in.

Needs Ollama (embeddinggemma + gemma4:12b). Written 2026-07-26 while the box
was reserved; not yet run.

Usage: uv run python ungated_ablation.py [seed ...]   (default: 2 18 23 7)
Writes ungated_ablation.<seed>.json per seed.
"""

import contextlib
import io
import json
import sys

import numpy as np
from openai import OpenAI

from gate_roc import _NullConn
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_shift, shift_rules
from src.eigen_memory_agent.memory_kernel import (
    EigenMemoryKernel, _mean_contrast, _unit,
)
from src import paths

EXECUTOR = "gemma4:12b"
EXTRA_BODY = {"reasoning_effort": "none"}
KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}
SEEDS = [int(a) for a in sys.argv[1:]] or [2, 18, 23, 7]


class _AxiomConn(_NullConn):
    """Also captures the inserted axiom row so the rule text is recoverable."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def cursor(self):
        conn = self
        base = super().cursor()

        class _Cur(type(base)):
            def execute(self, sql, params=None):
                if "INSERT INTO semantic_core" in sql:
                    conn.inserts += 1
                    conn.rows.append(params)

        return _Cur()


def run_seed(client, seed):
    trials, _ = load_shift(seed=seed)
    pre_rule, post_rule = shift_rules(seed)
    print(f"\n===== seed {seed}  pre={pre_rule}  post={post_rule} =====")

    embs = np.array([
        client.embeddings.create(input=t["input"], model=EMBEDDING_MODEL)
        .data[0].embedding for t in trials
    ])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    conn = _AxiomConn()
    kernel = EigenMemoryKernel(conn, client, model=EXECUTOR,
                               extra_body=EXTRA_BODY, rng_seed=seed, **KCFG)

    n_fail = 0
    for i, t in enumerate(trials):
        if i == 0:
            continue  # no earlier neighbor to copy from
        nn = int(np.argmax(embs[:i] @ embs[i]))
        copied = trials[nn]["label"]
        residual = embs[i] - embs[nn]
        fail = copied != t["label"]
        n_fail += fail
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.observe(embedding=embs[i], residual=residual,
                           was_correct=not fail, context_input=t["input"],
                           prediction=copied, actual=t["label"])

    F_red, S_red, basis = kernel._reduced_residuals()
    lam1, v_red = _mean_contrast(F_red, S_red)
    edge = kernel._permutation_edge(F_red, S_red)
    v_full = _unit(basis.T @ v_red)
    print(f"window: fail={len(kernel.fail_records)} succ={len(kernel.succ_records)} "
          f"(copier fail rate {n_fail / (len(trials) - 1):.2f} over full stream)")
    print(f"ungated statistic: lam1={lam1:.3f} edge={edge:.3f} "
          f"ratio={lam1 / edge:.2f}  (gate would {'FIRE' if lam1 > edge else 'refuse'})")

    kernel._crystallize(v_full, strength=lam1 / edge if edge > 0 else 1.0)
    rule = conn.rows[-1][0] if conn.inserts else None
    if rule is not None and not isinstance(rule, str):  # param order safety
        rule = next((p for p in conn.rows[-1] if isinstance(p, str) and "RULE" in p),
                    str(conn.rows[-1]))
    print(f"crystallized: {rule!r}")
    print(f"G4 targets — post rule: request→{post_rule['request']} "
          f"report→{post_rule['report']} | stale pre: request→{pre_rule['request']}")

    out = {"seed": seed, "executor": EXECUTOR, "kernel_cfg": KCFG,
           "proxy": "stale-copier", "lam1": lam1, "edge": edge,
           "ratio": lam1 / edge, "n_fail_window": len(kernel.fail_records),
           "n_succ_window": len(kernel.succ_records),
           "rule": rule, "pre_rule": pre_rule, "post_rule": post_rule}
    with open(paths.shift(f"ungated_ablation.{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    # Same timeout/retry as the agent's client: the SDK default is 600 s, so a
    # dropped local connection stalls the run for ten minutes (see 017f12c).
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                    timeout=120.0, max_retries=3)
    for seed in SEEDS:
        run_seed(client, seed)
    print("\nG4 scoring is a human read: does each rule map BOTH polarities to "
          "the post-shift labels? Score before looking at anything else.")


if __name__ == "__main__":
    main()
