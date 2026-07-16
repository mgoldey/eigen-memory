"""Guardrail 1 for the flip task (docs/C1_C3_TASK.md): measure the two regime
statistics with the REAL embedding model before any agent code runs.

  - probe-AUC(B) >= 0.8 : the polarity attribute must be linearly recoverable (C1)
  - m in [0.45, 0.60]   : nearest-neighbor label-copying must be near chance at the
                          buffer size the agent will actually have (C3)

Also reports (diagnostics, not gates): topic probe accuracy (should be high — the
dominant axis), label probe accuracy (should be LOW — the XOR is not linearly
decodable), and nn topic-match rate (retrieval should be topic-faithful).

Usage: uv run python guardrail_flip.py [seed]
Requires Ollama with embeddinggemma. No LLM, no Postgres.
"""

import sys

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import OLLAMA_BASE_URL
from src.dataset import load_flip

N_ITEMS = 300      # embedded sample of the train distribution
BUFFER = 150       # matches the protocol's train-phase memory size
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    data = load_flip(split="train", num_samples=N_ITEMS, seed=SEED)

    print(f"Embedding {N_ITEMS} messages (seed={SEED})...", flush=True)
    X = np.array([
        client.embeddings.create(input=d["input"], model="embeddinggemma:latest").data[0].embedding
        for d in data
    ])
    pol = np.array([d["meta"]["polarity"] == "request" for d in data])
    topic = np.array([d["meta"]["topic"] for d in data])
    label = np.array([d["label"] for d in data])

    # --- C1: linear probe for B (polarity) ---
    probe_b = cross_val_score(LogisticRegression(max_iter=2000), X, pol, cv=5).mean()
    probe_topic = cross_val_score(LogisticRegression(max_iter=2000), X, topic, cv=5).mean()
    probe_label = cross_val_score(LogisticRegression(max_iter=2000), X, label, cv=5).mean()

    # --- C3: m at the protocol buffer size ---
    rng = np.random.default_rng(SEED)
    store = rng.choice(N_ITEMS, size=BUFFER, replace=False)
    query = np.setdiff1d(np.arange(N_ITEMS), store)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = Xn[query] @ Xn[store].T
    nn = store[np.argmax(sims, axis=1)]

    m_pol = float(np.mean(pol[nn] == pol[query]))
    m_topic = float(np.mean(topic[nn] == topic[query]))
    copy_acc = float(np.mean(label[nn] == label[query]))

    print(f"\nprobe(B=polarity):  {probe_b:.3f}   (gate: >= 0.80)")
    print(f"probe(topic):       {probe_topic:.3f}   (diagnostic: should be high)")
    print(f"probe(label):       {probe_label:.3f}   (diagnostic: should be low — XOR)")
    print(f"nn topic match:     {m_topic:.3f}   (diagnostic: retrieval topic-faithful)")
    print(f"m (nn polarity):    {m_pol:.3f}   (gate: 0.45 - 0.60)")
    print(f"nn label-copy acc:  {copy_acc:.3f}   (should track m; chance = 0.33)")

    ok_c1 = probe_b >= 0.80
    ok_c3 = 0.45 <= m_pol <= 0.60
    print(f"\nC1 {'PASS' if ok_c1 else 'FAIL'} | C3 {'PASS' if ok_c3 else 'FAIL'}"
          f" -> {'TASK IS IN THE EIGEN WINDOW' if ok_c1 and ok_c3 else 'DEAD ON ARRIVAL — retune the generator'}")
    return 0 if (ok_c1 and ok_c3) else 1


if __name__ == "__main__":
    sys.exit(main())
