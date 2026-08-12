"""Replay BOTH triggers over the real seed streams and compare what fires.

The streak rule crystallized 1 axiom across 5 seeds. The ungated ablation showed
all four shut seeds carry a recoverable rule, and the §9a replay showed real
correctness labels do not rescue the streak rule -- it runs with no margin, and
its permutation threshold varies enough to decide a seed's verdict by itself.

This script asks the only question that matters for the mechanism: does an
anytime-valid sequential trigger fire where the streak rule did not, on the same
data, with an honest false-fire budget?

Both triggers see IDENTICAL inputs -- same embeddings, same residuals, same
real per-trial correctness from `trial_correct`, same featurization. The only
difference is the firing rule. Seeds whose artifact predates `trial_correct` are
skipped rather than reconstructed with a proxy.

Needs Ollama (embeddinggemma) to rebuild embeddings; no executor calls.

Usage: uv run python gate_sequential_replay.py [seed ...]   (default: all 5)
Writes results/shift/gate_sequential.<seed>.json per seed.
"""

import json
import sys

import numpy as np
from openai import OpenAI

from gate_roc import _NullConn
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_shift
from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from src import paths

KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}
SEEDS = [int(a) for a in sys.argv[1:]] or [2, 7, 18, 23, 42]
BATCH = 10


def _run(client, trials, embs, correct, seed, sequential):
    """Stream the trials through a kernel, checking once per batch as the live
    driver does. Returns (n_axioms, per-check log)."""
    conn = _NullConn()
    kernel = EigenMemoryKernel(conn, client, rng_seed=seed,
                               sequential_gate=sequential, **KCFG)
    log = []
    for i, t in enumerate(trials):
        if i > 0:
            nn = int(np.argmax(embs[:i] @ embs[i]))
            kernel.observe(embedding=embs[i], residual=embs[i] - embs[nn],
                           was_correct=bool(correct[i]),
                           context_input=t["input"],
                           prediction=trials[nn]["label"], actual=t["label"])
        # Check on the same cadence the driver uses: once per completed batch.
        if (i + 1) % BATCH == 0:
            before = conn.inserts
            n_e_before = len(kernel.evalue_history)
            n_d_before = len(kernel.detectability_history)
            # Snapshot evidence BEFORE the call so a post-fire reset() does not
            # make the firing check look like it fired on E=1.
            e_before = kernel._eproc.log_e
            kernel.check_and_crystallize()
            if sequential:
                # Only log when a check actually ran; the eligibility floor
                # returns early on batches with too few failures in-window.
                if len(kernel.evalue_history) > n_e_before:
                    e = kernel.evalue_history[-1]
                    log.append({"batch": (i + 1) // BATCH, "e": e,
                                "E": float(np.exp(min(e_before + np.log(e), 700.0))),
                                "fired": conn.inserts > before})
            elif len(kernel.detectability_history) > n_d_before:
                lam, edge = kernel.detectability_history[-1]
                log.append({"batch": (i + 1) // BATCH, "lam1": lam, "edge": edge,
                            "ratio": lam / edge if edge else None,
                            "fired": conn.inserts > before})
    return conn.inserts, log


def run_seed(client, seed):
    art = paths.shift(f"comparison_results.shift.{seed}.json")
    with open(art) as f:
        run = json.load(f)
    eigen = run.get("arms", {}).get("Treatment_Eigen", {})
    correct = eigen.get("trial_correct")
    if not correct:
        print(f"seed {seed}: no `trial_correct`; skipping (rerun the arm first)")
        return None

    trials, _ = load_shift(seed=seed)
    n = min(len(correct), len(trials))
    trials, correct = trials[:n], correct[:n]
    print(f"\n===== seed {seed}  ({n} trials, {sum(1 for c in correct if not c)} failures) =====")

    embs = np.array([
        client.embeddings.create(input=t["input"], model=EMBEDDING_MODEL)
        .data[0].embedding for t in trials
    ])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    streak_n, streak_log = _run(client, trials, embs, correct, seed, False)
    seq_n, seq_log = _run(client, trials, embs, correct, seed, True)

    print(f"  streak rule:     {streak_n} axiom(s)")
    if streak_log:
        print("    ratios: " + " ".join(
            f"{c['ratio']:.2f}" for c in streak_log if c.get("ratio")))
    print(f"  sequential gate: {seq_n} axiom(s)")
    if seq_log:
        print("    evidence: " + " ".join(f"{c['E']:.1f}" for c in seq_log))
        first = next((c["batch"] for c in seq_log if c["fired"]), None)
        if first:
            print(f"    first fired at batch {first} of {len(seq_log)} checks")

    out = {"seed": seed, "kernel_cfg": KCFG, "n_trials": n,
           "n_failures": sum(1 for c in correct if not c),
           "streak": {"n_axioms": streak_n, "checks": streak_log},
           "sequential": {"n_axioms": seq_n, "checks": seq_log},
           "n_axioms_live_run": eigen.get("n_axioms")}
    with open(paths.shift(f"gate_sequential.{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                    timeout=120.0, max_retries=3)
    rows = [r for r in (run_seed(client, s) for s in SEEDS) if r]
    if not rows:
        print("\nNothing replayed: no artifact carries `trial_correct` yet.")
        return
    print(f"\n{'seed':>5} {'streak':>7} {'sequential':>11} {'live run':>9}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} {r['streak']['n_axioms']:>7} "
              f"{r['sequential']['n_axioms']:>11} {str(r['n_axioms_live_run']):>9}")
    print("\nThe streak column should reproduce the live run's axiom count (same "
          "rule, same data). The sequential column is the question.")


if __name__ == "__main__":
    main()
