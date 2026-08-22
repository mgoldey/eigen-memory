"""The C1-and-C3 experiment (docs/C1_C3_TASK.md), demonstration scale.

Protocol (single seed; Guardrail 1 must PASS for the seed first — run
scripts/analysis/guardrail_flip.py):

  Train phase (memory arms only): N_TRAIN flip messages with feedback; memory built.
  Test phase (all arms): N_TEST held-out messages (disjoint objects, contexts,
  shells, markers), feedback OFF, memory FROZEN (no writes, no crystallization —
  run_batch only). Held-out accuracy is the headline metric.

Arms:
  Baseline      — no memory, test only.
  Oracle_Rule   — no memory, true flip-table in context, test only (headroom ceiling).
  Control_RAG   — episodic retrieval; trained.
  Treatment_Eigen — retrieval + corrected kernel (residual cPCA, gated); trained.

Outputs comparison_results.flip.<seed>.json with per-arm train curves, held-out
accuracy, per-polarity breakdown (the H2 per-cell signal), the eigen telemetry,
and per-arm health counters (buffer size, missing-NLL rate, parse-fallback rate)
so a silently degraded signal is loud in the artifact, not just in the console.

Usage: uv run python run_flip_experiment.py [seed]
"""

import json
import sys

import numpy as np
import psycopg2

from src.config import get_db_string
from src.dataset import flip_oracle_text, get_labels, load_dataset
from src.eigen_memory_agent.agent import AgenticMemoryLoop, parse_prediction
from src import paths

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRAIN = 100
N_TEST = 45
BATCH = 10
LABELS = get_labels("flip")


def reset_db(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE episodic_buffer, semantic_core;")
    conn.commit()


def run_phase(agent, data, learn, fallback_counter):
    """One pass over data. learn=False -> frozen memory: predict only."""
    accs = []
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
        accs.append(float(np.mean(correct)))
        yield batch, cleaned, correct, accs[-1]


def run_arm(name, conn, train_data, test_data, *, retrieval, eigen, static_context="", train=True):
    print(f"\n===== ARM: {name} (seed={SEED}) =====", flush=True)
    reset_db(conn)
    agent = AgenticMemoryLoop(
        get_db_string(), enable_retrieval=retrieval, enable_eigen_memory=eigen,
        labels=LABELS, static_context=static_context,
    )

    result = {"train_accs": [], "test_acc": None, "test_by_polarity": {}, "n_axioms": 0}
    fallbacks = [0, 0]  # [fired, total] across both phases

    if train:
        for i, (_, _, _, acc) in enumerate(run_phase(agent, train_data, learn=True, fallback_counter=fallbacks)):
            print(f"[{name}] train batch {i+1}/{N_TRAIN//BATCH}: acc={acc:.2f}", flush=True)
            result["train_accs"].append(acc)

    # Health telemetry, captured after training so a degraded signal is loud in
    # the artifact (this repo's history is three silent constant-surprise bugs).
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM episodic_buffer")
        result["buffer_size"] = cur.fetchone()[0]
    result["embed_failures"] = agent.embed_failures
    result["nll_missing_rate"] = (
        agent.nll_missing / agent.nll_probes if agent.nll_probes else None
    )

    # Test phase: feedback off, memory frozen (run_batch never writes).
    by_pol = {"request": [], "report": []}
    all_correct = []
    for batch, _, correct, acc in run_phase(agent, test_data, learn=False, fallback_counter=fallbacks):
        for item, c in zip(batch, correct):
            by_pol[item["meta"]["polarity"]].append(c)
            all_correct.append(c)
    result["test_acc"] = float(np.mean(all_correct))
    result["test_by_polarity"] = {k: float(np.mean(v)) for k, v in by_pol.items() if v}
    result["parse_fallback_rate"] = fallbacks[0] / fallbacks[1] if fallbacks[1] else None
    print(f"[{name}] HELD-OUT accuracy: {result['test_acc']:.3f} "
          f"by polarity: {result['test_by_polarity']}", flush=True)
    print(f"[{name}] health: buffer={result.get('buffer_size', 0)} "
          f"embed_failures={result['embed_failures']} "
          f"nll_missing_rate={result['nll_missing_rate']} "
          f"parse_fallback_rate={result['parse_fallback_rate']:.2f}", flush=True)

    if eigen:
        result["detectability"] = [(float(lam), float(edge)) for lam, edge in agent.kernel.detectability_history]
        result["n_axioms"] = len(agent.kernel.consumed_directions)
        with conn.cursor() as cur:
            cur.execute("SELECT axiom_content, strength_score FROM semantic_core")
            result["axioms"] = [
                {"strength": float(s),
                 "rule": (a.split("RULE:")[-1].strip() if "RULE:" in a else a)[:500]}
                for a, s in cur.fetchall()
            ]
        for ax in result.get("axioms", []):
            print(f"[{name}] AXIOM (strength {ax['strength']:.2f}): {ax['rule'][:200]}", flush=True)
    return result


def main():
    conn = psycopg2.connect(get_db_string())
    train_data = load_dataset(task="flip", split="train", num_samples=N_TRAIN, seed=SEED)
    test_data = load_dataset(task="flip", split="test", num_samples=N_TEST, seed=SEED)
    oracle = flip_oracle_text(SEED)
    print(f"Oracle rule for seed {SEED}:\n{oracle}\n", flush=True)

    results = {"seed": SEED, "n_train": N_TRAIN, "n_test": N_TEST, "arms": {}}
    results["arms"]["Baseline"] = run_arm(
        "Baseline", conn, train_data, test_data, retrieval=False, eigen=False, train=False)
    results["arms"]["Oracle_Rule"] = run_arm(
        "Oracle_Rule", conn, train_data, test_data, retrieval=False, eigen=False,
        static_context=oracle, train=False)
    results["arms"]["Control_RAG"] = run_arm(
        "Control_RAG", conn, train_data, test_data, retrieval=True, eigen=False)
    results["arms"]["Treatment_Eigen"] = run_arm(
        "Treatment_Eigen", conn, train_data, test_data, retrieval=True, eigen=True)

    with open(paths.flip(f"comparison_results.flip.{SEED}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== HELD-OUT SUMMARY (frozen memory, disjoint surface vocabulary) ===")
    for arm, r in results["arms"].items():
        print(f"  {arm:16s}: {r['test_acc']:.3f}  {r['test_by_polarity']}")
    base = results["arms"]["Baseline"]["test_acc"]
    orac = results["arms"]["Oracle_Rule"]["test_acc"]
    rag = results["arms"]["Control_RAG"]["test_acc"]
    eig = results["arms"]["Treatment_Eigen"]["test_acc"]
    print(f"\nGuardrail 3 ceiling gate: Oracle - Baseline = {orac - base:+.3f} "
          f"({'OK (>= 0.20)' if orac - base >= 0.20 else 'GATE FAILED - comparison moot'})")
    print(f"H1 (Eigen > max(RAG, Baseline)): {eig:.3f} vs {max(rag, base):.3f} "
          f"-> {'SUPPORTED' if eig > max(rag, base) else 'NOT SUPPORTED'} (single seed, demo scale)")
    conn.close()


if __name__ == "__main__":
    main()
