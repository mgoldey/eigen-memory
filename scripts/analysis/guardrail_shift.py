"""Rule-Shift gates G1 + G2 (docs/NEXT_EXPERIMENT.md §5). No LLM, no Postgres —
just the real embedding model over the real generator, before any agent money
is spent.

  G1: probe-AUC(polarity) >= 0.8 on the new generator. The shift generator
      reuses the flip surface text verbatim, so this should replicate flip's
      C1 — measured here on the shift protocol's own samples, not assumed.
  G2: cross-split post-shift copy accuracy vs the ACTUAL frozen buffer
      <= 0.45. Buffer = the 160 era-correct-labeled trials (exactly what a
      RAG store holds at freeze: 100 stale-request-label items + 60 current),
      queries = the 90 held-out post-shift items.

G2 amendment (2026-07-17, measured on 5 seeds BEFORE any pilot spend): the
aggregate <= 0.45 bar is unachievable BY CONSTRUCTION — only the request row
shifts, so report queries stay copyable forever (measured 0.67-0.98) while
request-copying craters as intended (0.00-0.35); the aggregate can't drop
below ~0.5 while retrieval works at all. The strict result is kept in the
JSON as g2_strict. Amended G2 binds on what the gate was for:
  G2a: copy_acc on REQUESTS (the shifted sub-task) <= 0.45, and
  G2b: the Recency_RAG policy ceiling (k=5, newest wins) <= executor
       rule-following R - 0.10, so a correct axiom has decision-rule headroom
       over the kill arm. R comes from the executor's RFμ artifact.

Also measured (diagnostic, not a gate): the recency-copy ceiling — top-5
neighbors, most recent wins — which is what the Recency_RAG kill arm can
realize by policy alone.

Writes guardrail.shift.<seed>.json.
Usage: uv run python guardrail_shift.py [seed]
"""

import json
import sys

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_shift, shift_rules
from src import paths

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
EXECUTOR_RFMU = str(paths.calibration("rfmu.gemma4_12b.json"))  # the designated executor's artifact


def executor_r():
    with open(EXECUTOR_RFMU) as f:
        return json.load(f)["acc"]["R"]


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    trials, heldout = load_shift(seed=SEED)
    pre, post = shift_rules(seed=SEED)
    print(f"seed={SEED}  pre-rule={pre}  post-rule={post}")

    def embed_all(items, what):
        print(f"Embedding {len(items)} {what} messages...", flush=True)
        return np.array([
            client.embeddings.create(input=d["input"], model=EMBEDDING_MODEL).data[0].embedding
            for d in items
        ])

    X = embed_all(trials, "trial-stream")
    Xq = embed_all(heldout, "held-out")

    # --- G1: polarity probe-AUC on the shift generator's own trial stream ---
    pol = np.array([d["meta"]["polarity"] == "request" for d in trials])
    probe_auc = cross_val_score(
        LogisticRegression(max_iter=2000), X, pol, cv=5, scoring="roc_auc"
    ).mean()

    # --- G2: copy accuracy vs the frozen buffer, held-out queries ---
    Xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xn = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
    sims = Xn @ Xs.T

    stored_label = np.array([d["label"] for d in trials])       # era-correct = stale for pre
    true_label = np.array([d["label"] for d in heldout])        # post rule
    pol_q = np.array([d["meta"]["polarity"] == "request" for d in heldout])

    nn = np.argmax(sims, axis=1)
    copy_acc = float(np.mean(stored_label[nn] == true_label))
    m_pol = float(np.mean(pol[nn] == pol_q))
    nn_stale = float(np.mean(nn < 100))  # fraction of nearest neighbors from the pre window

    # Diagnostic: what the Recency_RAG policy realizes with k=5, newest wins.
    top5 = np.argsort(-sims, axis=1)[:, :5]
    recency_pick = top5[np.arange(len(heldout)), np.argmax(top5, axis=1)]
    recency_copy = float(np.mean(stored_label[recency_pick] == true_label))

    # Per-polarity split of copy accuracy: the report row never shifted, so
    # copying should stay strong there and crater on requests.
    req = pol_q
    copy_req = float(np.mean((stored_label[nn] == true_label)[req]))
    copy_rep = float(np.mean((stored_label[nn] == true_label)[~req]))

    r = executor_r()
    g1 = probe_auc >= 0.80
    g2_strict = copy_acc <= 0.45
    g2a = copy_req <= 0.45
    g2b = recency_copy <= r - 0.10
    g2 = g2a and g2b

    print(f"\nG1 probe-AUC(polarity): {probe_auc:.3f}   (gate: >= 0.80) -> {'PASS' if g1 else 'FAIL'}")
    print(f"G2 strict copy_acc (nn, frozen buffer): {copy_acc:.3f}   (original gate: <= 0.45)"
          f" -> {'PASS' if g2_strict else 'FAIL — see amendment'}")
    print(f"G2a copy_acc on requests (shifted row): {copy_req:.3f}   (gate: <= 0.45)"
          f" -> {'PASS' if g2a else 'FAIL'}   [reports: {copy_rep:.3f}, never shifted]")
    print(f"G2b recency ceiling {recency_copy:.3f} <= executor R - 0.10 = {r - 0.10:.3f}"
          f" -> {'PASS' if g2b else 'FAIL'}")
    print(f"   nn polarity match m: {m_pol:.3f}   nn-from-pre-window: {nn_stale:.3f}")

    verdict = g1 and g2
    print(f"\n{'G1+G2(amended) PASS — proceed to the pilot seed' if verdict else 'GATE FAILED — stop before the pilot'}")

    with open(paths.shift(f"guardrail.shift.{SEED}.json"), "w") as f:
        json.dump({
            "seed": SEED, "embedding_model": EMBEDDING_MODEL,
            "pre_rule": pre, "post_rule": post,
            "probe_auc_polarity": probe_auc,
            "copy_acc": copy_acc, "copy_acc_requests": copy_req,
            "copy_acc_reports": copy_rep, "m_polarity": m_pol,
            "nn_from_pre_window": nn_stale, "recency_copy_k5": recency_copy,
            "executor_r": r,
            "g1_pass": bool(g1), "g2_strict_pass": bool(g2_strict),
            "g2a_pass": bool(g2a), "g2b_pass": bool(g2b),
            "g2_amended_pass": bool(g2),
        }, f, indent=2)
    print(f"Wrote guardrail.shift.{SEED}.json")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
