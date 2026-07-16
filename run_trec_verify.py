"""Re-verify the rebuilt kernel's TREC crystallization claim on the FIXED code.

The claim under test (from the 2026-07-14 smoke run, which predates the
review-blocker fixes in commit d135462): running the Treatment_Eigen arm on
TREC for 120 trials crystallizes exactly ONE axiom, and the rule is TRUE
(quantity -> NUM; location/person/entity -> LOC/HUM). TREC's labels are
single-token so that run dodged the _extract_nll bug, but the CoT-storage
blocker and temperature=0.8 sampling were live — this script re-tests the
claim on clean instrumentation and archives the result.

Runs ONE arm (retrieval + eigen), training only — the claim is about what
crystallizes, not about beating RAG (TREC fails C3: one exemplar suffices).

Usage: uv run python run_trec_verify.py [seed]   (default 42)
Writes trec_verify.<seed>.json.
"""

import json
import sys

import numpy as np
import psycopg2

from src.config import get_db_string
from src.dataset import get_labels, load_dataset
from src.eigen_memory_agent.agent import AgenticMemoryLoop, parse_prediction

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_TRIALS = 120
BATCH = 10
LABELS = get_labels("trec")


def main():
    conn = psycopg2.connect(get_db_string())
    with conn.cursor() as cur:
        cur.execute("TRUNCATE episodic_buffer, semantic_core;")
    conn.commit()

    data = load_dataset(task="trec", split="train", num_samples=N_TRIALS, seed=SEED)
    agent = AgenticMemoryLoop(get_db_string(), enable_retrieval=True,
                              enable_eigen_memory=True, labels=LABELS)

    accs, fallbacks = [], [0, 0]
    for b in range(0, N_TRIALS, BATCH):
        batch = data[b:b + BATCH]
        inputs = [d["input"] for d in batch]
        truths = [d["label"] for d in batch]
        preds, embs, sal, _, ctxs, resids = agent.run_batch(inputs)
        agent.learn_batch(inputs, preds, truths, embs, sal, ctxs, resids)
        parsed = [parse_prediction(p, LABELS) for p in preds]
        fallbacks[0] += sum(fb for _, fb in parsed)
        fallbacks[1] += len(parsed)
        acc = float(np.mean([lab == t for (lab, _), t in zip(parsed, truths)]))
        accs.append(acc)
        print(f"batch {b // BATCH + 1}/{N_TRIALS // BATCH}: acc={acc:.2f}", flush=True)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM episodic_buffer")
        buffer_size = cur.fetchone()[0]
        cur.execute("SELECT axiom_content, strength_score FROM semantic_core")
        axioms = [{"strength": float(s), "rule": a} for a, s in cur.fetchall()]

    result = {
        "seed": SEED,
        "n_trials": N_TRIALS,
        "train_accs": accs,
        "buffer_size": buffer_size,
        "embed_failures": agent.embed_failures,
        "nll_missing_rate": agent.nll_missing / agent.nll_probes if agent.nll_probes else None,
        "parse_fallback_rate": fallbacks[0] / fallbacks[1],
        "detectability": [(float(l1), float(e)) for l1, e in agent.kernel.detectability_history],
        "n_axioms": len(axioms),
        "axioms": axioms,
    }
    with open(f"trec_verify.{SEED}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== TREC VERIFY (seed {SEED}, fixed code) ===")
    print(f"axioms crystallized: {len(axioms)} (claim: exactly 1, and true)")
    for ax in axioms:
        print(f"  (strength {ax['strength']:.2f}) {ax['rule']}")
    print(f"health: buffer={buffer_size} nll_missing_rate={result['nll_missing_rate']:.3f} "
          f"parse_fallback_rate={result['parse_fallback_rate']:.3f}")
    conn.close()


if __name__ == "__main__":
    main()
