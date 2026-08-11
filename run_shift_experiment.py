"""The Rule-Shift experiment (docs/NEXT_EXPERIMENT.md §5) — pilot / single seed.

Protocol: 100 pre-shift trials (negative control — all memory arms should be at
parity) -> the request row of the global polarity rule flips -> 60 post-shift
adaptation trials -> memory FROZEN -> 90 held-out post-shift items in disjoint
test vocabulary. Feedback labels are always era-correct, so a RAG buffer ends
up holding 100 stale-request-label episodes plus 60 current ones — the exact
condition RFμ's CS arm measured (stale copy realizes 0.450 vs rule 0.983 on
the designated executor).

Arms (the 6th, the measured copy ceiling, lives in guardrail.shift.<seed>.json):
  Baseline       — no memory, held-out only.
  Oracle_Post    — no memory, true POST-shift rule as PROSE in context
                   (RFμ: prose 0.983 vs table 0.517 on gemma4:12b).
  Control_RAG    — episodic retrieval, k=5, similarity order; trained.
  Recency_RAG    — the kill arm: same retrieval, presented newest-first with a
                   staleness hint; trained.
  Treatment_Eigen— retrieval + kernel in the shift configuration; a selected
                   axiom REPLACES exemplars (pre-registered injection policy).

Executor/crystallizer/probe: gemma4:12b via Ollama's OpenAI-compat endpoint
with reasoning_effort="none" — a thinking stream otherwise burns the whole
budget and returns empty content (RFμ bug five). Verified live: content and
logprobs both present, label tokens appear as prefixes (ES/DE/FILE), which the
prefix-matching _extract_nll handles.

Kernel shift configuration (dated amendments in docs/NEXT_EXPERIMENT.md):
  window=60                  — forgetting; without it 100 pre-shift records
                               swamp the 60 post-shift ones.
  contrast_on="embedding_mean" — two-sample mean contrast; the pre-registered
                               query-embedding cPCA is variance-based and is
                               blind to a location difference by construction.
  consecutive_detections=3   — G3 as pre-registered.
  stability_cos=0.5          — sample-size-aware relaxation (gate-ROC: 0.95 is
                               unreachable below ~8x edge; 0.5 is still
                               p~2e-4 against the random-direction null in
                               the r=50 working space).

G4 ordering: fired axioms are printed and written BEFORE the held-out summary.
Writes comparison_results.shift.<seed>.json.
Usage: uv run python run_shift_experiment.py [seed]
"""

import json
import os
import sys

import numpy as np
import psycopg2

from src.config import get_db_string
from src.dataset import (N_SHIFT_PRE, get_labels, load_shift, shift_oracle_text,
                         shift_rules)
from src.eigen_memory_agent.agent import AgenticMemoryLoop, parse_prediction
from src import paths

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
BATCH = 10
LABELS = get_labels("flip")
EXECUTOR = "gemma4:12b"
EXTRA_BODY = {"reasoning_effort": "none"}
RETRIEVAL_K = 5
KERNEL_SHIFT_CFG = {"window": 60, "contrast_on": "embedding_mean",
                    "consecutive_detections": 3, "stability_cos": 0.5}


def reset_db(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE episodic_buffer, semantic_core;")
    conn.commit()


def run_phase(agent, data, learn, fallback_counter):
    """One pass over data in batches. learn=False -> frozen: predict only."""
    for b in range(0, len(data), BATCH):
        batch = data[b:b + BATCH]
        inputs = [d["input"] for d in batch]
        truths = [d["label"] for d in batch]
        preds, embs, sal, _, ctxs, resids = agent.run_batch(inputs)
        if learn:
            agent.learn_batch(inputs, preds, truths, embs, sal, ctxs, resids)
        parsed = [parse_prediction(p, LABELS) for p in preds]
        cleaned = [lab for lab, _ in parsed]
        fallback_counter[0] += sum(fb for _, fb in parsed)
        fallback_counter[1] += len(parsed)
        correct = [p == t for p, t in zip(cleaned, truths)]
        yield batch, cleaned, correct


def run_arm(name, conn, trials, heldout, *, retrieval, eigen, static_context="",
            recency=False, train=True):
    print(f"\n===== ARM: {name} (seed={SEED}) =====", flush=True)
    reset_db(conn)
    agent = AgenticMemoryLoop(
        get_db_string(), enable_retrieval=retrieval, enable_eigen_memory=eigen,
        labels=LABELS, static_context=static_context,
        model=EXECUTOR, thought_model=EXECUTOR, extra_body=EXTRA_BODY,
        retrieval_k=RETRIEVAL_K, recency_rerank=recency,
        axiom_replaces_exemplars=eigen,
        kernel_kwargs=KERNEL_SHIFT_CFG if eigen else None,
    )

    result = {"train_accs": [], "checks_after_batch": [], "axioms_after_batch": []}
    fallbacks = [0, 0]

    trial_correct = []
    if train:
        n_batches = len(trials) // BATCH
        for i, (_, _, correct) in enumerate(
                run_phase(agent, trials, learn=True, fallback_counter=fallbacks)):
            trial_correct.extend(correct)
            acc = float(np.mean(correct))
            result["train_accs"].append(acc)
            if eigen:
                result["checks_after_batch"].append(len(agent.kernel.detectability_history))
                result["axioms_after_batch"].append(len(agent.kernel.consumed_directions))
            phase = "pre" if (i + 1) * BATCH <= N_SHIFT_PRE else "POST"
            print(f"[{name}] trial batch {i+1}/{n_batches} ({phase}-shift): acc={acc:.2f}", flush=True)
        # Item-level trial correctness, not just the two means. The gate's
        # fail/succ split is built from exactly this signal, so without it an
        # offline gate re-analysis has to RECONSTRUCT which trials failed --
        # which is what ungated_ablation.py does with its stale-copier proxy,
        # and why that ablation can show the signal was recoverable without
        # establishing whether the live gate's miss is a featurization problem
        # or merely a noisy correctness signal. Persisting it makes the two
        # distinguishable. Cheap: 160 bools per arm. Present only on the three
        # arms that train (the memory arms); Baseline and Oracle_Post skip the
        # trial stream entirely, and neither feeds a gate. The bool() cast is
        # load-bearing: run_phase yields numpy arrays and np.bool_ is not
        # JSON-serializable.
        result["trial_correct"] = [bool(c) for c in trial_correct]
        result["pre_shift_acc"] = float(np.mean(trial_correct[:N_SHIFT_PRE]))
        result["post_adapt_acc"] = float(np.mean(trial_correct[N_SHIFT_PRE:]))
        print(f"[{name}] trial-stream acc: pre-shift {result['pre_shift_acc']:.3f} "
              f"post-shift adaptation {result['post_adapt_acc']:.3f}", flush=True)

    # Health telemetry before anything is interpreted (this repo's history is
    # three silent constant-surprise bugs).
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM episodic_buffer")
        result["buffer_size"] = cur.fetchone()[0]
    result["embed_failures"] = agent.embed_failures
    result["nll_missing_rate"] = (
        agent.nll_missing / agent.nll_probes if agent.nll_probes else None)

    # G4 ordering: surface fired axioms BEFORE held-out accuracy is computed.
    if eigen:
        result["detectability"] = [(float(l), float(e))
                                   for l, e in agent.kernel.detectability_history]
        result["n_axioms"] = len(agent.kernel.consumed_directions)
        with conn.cursor() as cur:
            cur.execute("SELECT axiom_content, strength_score FROM semantic_core")
            result["axioms"] = [
                {"strength": float(s),
                 "rule": (a.split("RULE:")[-1].strip() if "RULE:" in a else a)[:500]}
                for a, s in cur.fetchall()
            ]
        print(f"[{name}] G4 — score these against the planted post-shift rule "
              f"BEFORE reading held-out accuracy:", flush=True)
        for ax in result.get("axioms", []) or [{"strength": 0, "rule": "(none fired)"}]:
            print(f"[{name}]   AXIOM (strength {ax['strength']:.2f}): {ax['rule'][:300]}", flush=True)

    # Held-out phase: feedback off, memory frozen.
    by_pol = {"request": [], "report": []}
    all_correct = []
    for batch, _, correct in run_phase(agent, heldout, learn=False, fallback_counter=fallbacks):
        for item, c in zip(batch, correct):
            by_pol[item["meta"]["polarity"]].append(c)
            all_correct.append(c)
    result["test_acc"] = float(np.mean(all_correct))
    result["test_correct"] = [bool(c) for c in all_correct]  # item-level, for paired tests
    result["test_by_polarity"] = {k: float(np.mean(v)) for k, v in by_pol.items() if v}
    result["parse_fallback_rate"] = fallbacks[0] / fallbacks[1] if fallbacks[1] else None
    print(f"[{name}] HELD-OUT accuracy: {result['test_acc']:.3f} "
          f"by polarity: {result['test_by_polarity']}", flush=True)
    print(f"[{name}] health: buffer={result['buffer_size']} "
          f"embed_failures={result['embed_failures']} "
          f"nll_missing_rate={result['nll_missing_rate']} "
          f"parse_fallback_rate={result['parse_fallback_rate']:.2f}", flush=True)
    return result


def main():
    conn = psycopg2.connect(get_db_string())
    trials, heldout = load_shift(seed=SEED)
    pre, post = shift_rules(seed=SEED)
    oracle = shift_oracle_text(SEED)
    print(f"seed={SEED}  pre-rule={pre}  post-rule={post}")
    print(f"Oracle (post-shift, prose): {oracle}\n", flush=True)

    results = {"seed": SEED, "executor": EXECUTOR, "retrieval_k": RETRIEVAL_K,
               "kernel_cfg": KERNEL_SHIFT_CFG,
               "pre_rule": pre, "post_rule": post, "arms": {}}

    # Crash resilience: each arm resets the DB and is independent, so results
    # are dumped after every arm and completed arms are skipped on relaunch
    # (the first pilot attempt died to a mid-run reboot with nothing on disk).
    out_path = paths.shift(f"comparison_results.shift.{SEED}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("seed") == SEED and prev.get("executor") == EXECUTOR:
            results["arms"] = prev.get("arms", {})
            print(f"Resuming; completed arms: {sorted(results['arms'])}", flush=True)

    arm_specs = [
        ("Baseline", dict(retrieval=False, eigen=False, train=False)),
        ("Oracle_Post", dict(retrieval=False, eigen=False,
                             static_context=oracle, train=False)),
        ("Control_RAG", dict(retrieval=True, eigen=False)),
        ("Recency_RAG", dict(retrieval=True, eigen=False, recency=True)),
        ("Treatment_Eigen", dict(retrieval=True, eigen=True)),
    ]
    for name, kwargs in arm_specs:
        if name in results["arms"]:
            continue
        results["arms"][name] = run_arm(name, conn, trials, heldout, **kwargs)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== HELD-OUT SUMMARY (frozen memory, post-shift labels, disjoint vocab) ===")
    for arm, r in results["arms"].items():
        print(f"  {arm:16s}: {r['test_acc']:.3f}  {r['test_by_polarity']}")
    eig = results["arms"]["Treatment_Eigen"]
    rec = results["arms"]["Recency_RAG"]["test_acc"]
    print(f"\nG3 (detectability fired -> axiom): {eig.get('n_axioms', 0)} axiom(s)")
    print(f"Primary endpoint (pilot read only — 5 seeds decide): Eigen {eig['test_acc']:.3f} "
          f"vs Recency_RAG {rec:.3f} (win needs Δ >= +0.10 across seeds)")
    conn.close()


if __name__ == "__main__":
    main()
