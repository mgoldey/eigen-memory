"""Pilot derisk analyses (docs/NEXT_EXPERIMENT.md §6): the complaint-driven
checks that need no new LLM arms.

  1. Paired McNemar (exact binomial, one-sided) between Treatment_Eigen and
     each comparison arm on the shared 90 held-out items.
  2. Idealized copy ceilings: what could a copy policy with PERFECT staleness
     filtering realize (store restricted to the 60 post-shift trials) — the
     strongest possible version of the Recency_RAG complaint.

Reads comparison_results.shift.<seed>.<tag>.json (tag optional), writes
derisk.shift.<seed>.json.
Usage: uv run python derisk_pilot.py [seed] [results-path]
"""

import json
import sys

import numpy as np
from openai import OpenAI
from scipy.stats import binomtest

from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import N_SHIFT_PRE, load_shift

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
PATH = sys.argv[2] if len(sys.argv) > 2 else f"comparison_results.shift.{SEED}.json"


def mcnemar(a, b):
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5, alternative="greater").pvalue) if n else 1.0
    return {"treatment_only": a_only, "other_only": b_only, "p_one_sided": p}


def main():
    arms = json.load(open(PATH))["arms"]
    eig = arms["Treatment_Eigen"]["test_correct"]
    paired = {}
    for other in ["Recency_RAG", "Control_RAG", "Oracle_Post", "Baseline"]:
        paired[other] = mcnemar(eig, arms[other]["test_correct"])
        print(f"Eigen vs {other:12s}: {paired[other]}")

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    trials, heldout = load_shift(seed=SEED)

    def embed(items):
        return np.array([
            client.embeddings.create(input=d["input"], model=EMBEDDING_MODEL).data[0].embedding
            for d in items
        ])

    X, Xq = embed(trials), embed(heldout)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
    stored = np.array([d["label"] for d in trials])
    true = np.array([d["label"] for d in heldout])

    post = slice(N_SHIFT_PRE, len(trials))
    sims = Xq @ X[post].T
    nn = np.argmax(sims, axis=1)
    ceil_nn = float(np.mean(stored[post][nn] == true))
    top5 = np.argsort(-sims, axis=1)[:, :5]
    maj = [max(set(stored[post][t]), key=list(stored[post][t]).count) for t in top5]
    ceil_maj = float(np.mean(np.array(maj) == true))
    print(f"perfect-staleness-filter copy ceilings: nn={ceil_nn:.3f} top5-majority={ceil_maj:.3f}")

    out = {"seed": SEED, "results_path": PATH, "paired_mcnemar": paired,
           "ideal_copy_ceiling_nn": ceil_nn, "ideal_copy_ceiling_top5maj": ceil_maj}
    with open(f"derisk.shift.{SEED}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote derisk.shift.{SEED}.json")


if __name__ == "__main__":
    main()
