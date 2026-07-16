"""Guardrail 1 for the flip task (docs/C1_C3_TASK.md): measure the regime
statistics with the REAL embedding model, under the experiment's EXACT
protocol conditions, before any agent code runs.

  - probe-AUC(B) >= 0.8 : the polarity attribute must be linearly recoverable (C1)
  - m in [0.45, 0.60]   : nearest-neighbor polarity-copying must be near chance (C3)

m and the copy ceiling are measured the way the experiment will actually
retrieve: store = the seed's N_TRAIN train messages (the buffer's upper bound),
queries = the seed's N_TEST held-out messages. An earlier version measured m
with train-split queries against a 150-item store — the wrong queries at the
wrong buffer size (review finding; see docs/C1_C3_TASK.md).

The C5 executor gate compares the Oracle arm against `copy_acc` (3-class
nearest-neighbor LABEL-copy accuracy on the held-out queries) — that, not the
2-way polarity match m, is the honest blind-copying ceiling.

Writes guardrail.flip.<seed>.json for aggregate_flip.py to consume.

Usage: uv run python guardrail_flip.py [seed]
Requires Ollama with the embedding model in src/config.py. No LLM, no Postgres.
"""

import json
import sys

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_flip

N_PROBE = 300      # train-distribution sample for the linear probes
N_TRAIN = 100      # must match run_flip_experiment.N_TRAIN (the memory store)
N_TEST = 45        # must match run_flip_experiment.N_TEST (the queries)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    probe_data = load_flip(split="train", num_samples=N_PROBE, seed=SEED)
    train_data = load_flip(split="train", num_samples=N_TRAIN, seed=SEED)
    test_data = load_flip(split="test", num_samples=N_TEST, seed=SEED)
    # The generator streams deterministically, so the experiment's train set is
    # the probe sample's prefix — verify rather than assume, then reuse embeddings.
    assert [d["input"] for d in train_data] == [d["input"] for d in probe_data[:N_TRAIN]]

    def embed_all(items, what):
        print(f"Embedding {len(items)} {what} messages (seed={SEED})...", flush=True)
        return np.array([
            client.embeddings.create(input=d["input"], model=EMBEDDING_MODEL).data[0].embedding
            for d in items
        ])

    X = embed_all(probe_data, "train-distribution")
    X_test = embed_all(test_data, "held-out")

    pol = np.array([d["meta"]["polarity"] == "request" for d in probe_data])
    topic = np.array([d["meta"]["topic"] for d in probe_data])
    label = np.array([d["label"] for d in probe_data])

    # --- C1: linear probe for B (polarity), on the train distribution ---
    probe_b = cross_val_score(LogisticRegression(max_iter=2000), X, pol, cv=5).mean()
    probe_topic = cross_val_score(LogisticRegression(max_iter=2000), X, topic, cv=5).mean()
    probe_label = cross_val_score(LogisticRegression(max_iter=2000), X, label, cv=5).mean()

    # --- C3 + the copy ceiling: protocol conditions (train store, test queries) ---
    Xs = X[:N_TRAIN]
    Xs = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    Xq = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    nn = np.argmax(Xq @ Xs.T, axis=1)

    pol_q = np.array([d["meta"]["polarity"] == "request" for d in test_data])
    topic_q = np.array([d["meta"]["topic"] for d in test_data])
    label_q = np.array([d["label"] for d in test_data])

    m_pol = float(np.mean(pol[nn] == pol_q))
    m_topic = float(np.mean(topic[nn] == topic_q))
    copy_acc = float(np.mean(label[nn] == label_q))

    print(f"\nprobe(B=polarity):  {probe_b:.3f}   (gate: >= 0.80)")
    print(f"probe(topic):       {probe_topic:.3f}   (diagnostic: should be high)")
    print(f"probe(label):       {probe_label:.3f}   (diagnostic: should be low — XOR)")
    print(f"nn topic match:     {m_topic:.3f}   (diagnostic: retrieval topic-faithful)")
    print(f"m (nn polarity):    {m_pol:.3f}   (gate: 0.45 - 0.60)")
    print(f"copy ceiling:       {copy_acc:.3f}   (nn label-copy acc; C5 compares Oracle to THIS)")

    ok_c1 = probe_b >= 0.80
    ok_c3 = 0.45 <= m_pol <= 0.60
    print(f"\nC1 {'PASS' if ok_c1 else 'FAIL'} | C3 {'PASS' if ok_c3 else 'FAIL'}"
          f" -> {'TASK IS IN THE EIGEN WINDOW' if ok_c1 and ok_c3 else 'DEAD ON ARRIVAL — retune the generator'}")

    with open(f"guardrail.flip.{SEED}.json", "w") as f:
        json.dump({
            "seed": SEED, "n_probe": N_PROBE, "n_store": N_TRAIN, "n_query": N_TEST,
            "embedding_model": EMBEDDING_MODEL,
            "probe_polarity": probe_b, "probe_topic": probe_topic, "probe_label": probe_label,
            "m_polarity": m_pol, "nn_topic_match": m_topic, "copy_acc": copy_acc,
            "c1_pass": bool(ok_c1), "c3_pass": bool(ok_c3),
        }, f, indent=2)
    print(f"Wrote guardrail.flip.{SEED}.json")
    return 0 if (ok_c1 and ok_c3) else 1


if __name__ == "__main__":
    sys.exit(main())
