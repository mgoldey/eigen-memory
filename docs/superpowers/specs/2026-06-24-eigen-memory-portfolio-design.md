# Eigen-Memory Agent → Portfolio Piece — Design

**Date:** 2026-06-24
**Status:** Approved (pending spec review)
**Audience:** Recruiters / general technical readers

## 1. Goal

Transform this research repo into a recruiter-friendly portfolio piece that showcases a
from-scratch **learning agent with a self-correcting memory system**, validated by a
controlled experiment whose result is reported honestly — win or lose.

The headline narrative is **capability-first**: "an agent that learns hidden rules from its
own mistakes, using a Postgres/pgvector memory system." The controlled experiment is the
validation section, not the headline.

We give the Eigen-Memory ("Treatment") arm a genuinely fair shot at beating plain RAG, but
do not manufacture a win.

### Definition of "fair shot"

In scope (fair):
- Fix bugs that artificially sabotage the Treatment arm.
- Tune surprise thresholds and axiom counts to *reasonable, documented* values.
- Let the Eigen arm actually build and inject axioms as designed.
- Run a few seeds informally so a single noisy run does not decide the outcome.

Out of bounds (cheating):
- Cherry-picking a favorable seed.
- Hand-writing the hidden rules the LLM was supposed to discover on its own.
- Tuning thresholds per-arm to flatter Treatment specifically.

## 2. Current State (as explored)

- **What it is:** A 3-tier agentic memory system tested on a "hidden rules" classification
  game. The agent classifies integers 1–100 as RED (prime) / BLUE (÷5) / GREEN (else),
  learning the hidden rule only from Correct/Incorrect feedback.
- **Three arms** (`simulate.py`): Baseline (no retrieval), Control_RAG (retrieve top-3
  episodes), Treatment_Eigen (RAG + PCA-crystallized "axioms").
- **Current result:** Treatment_Eigen (~40–50% final accuracy) *loses* to Control_RAG
  (~80–90%) and also stores *more* memory — worse and more expensive.
- **Infra reality (config drift):** `gemma3:4b` and `embeddinggemma:latest` are present in
  Ollama (experiment is runnable), but no Postgres is listening on the code's hardcoded
  port 5432. A stray pgvector container runs on 5433. The experiment is not runnable as-is
  without a config fix.

### Suspected bugs that may be sabotaging Treatment

1. **Fragile surprise extraction** (`src/eigen_memory_agent/agent.py:113-138`): the NLL path
   loops `top_logprobs` for the true-label token; if absent, `target_lp` is left
   **undefined** and the except path silently fires. Surprise scales can differ between arms.
2. **Threshold mismatch** (`agent.py:141` episodic write `s_pred > 1.5` vs `agent.py:151`
   crystallize `s_pred > 2.0`): if NLL rarely exceeds 2.0, the Eigen arm barely builds any
   axioms — so the hypothesis is never really exercised, only noise is added.
3. **Unvalidated axiom quality**: a wrong LLM-written axiom actively *poisons* the retrieved
   context, which could explain Treatment underperforming RAG.

## 3. Plan — Three Phases

### Phase 1 — Debug to a valid experiment ("mechanism fires + fair tuning")

1. **Harden surprise extraction** (`agent.py:113-138`): handle the missing-token case
   explicitly (initialize `target_lp`, deterministic + logged fallback). No silent undefined.
2. **Reconcile thresholds** (`agent.py:141` vs `:151`): choose consistent, documented cutoffs
   so the Eigen arm actually crystallizes axioms on this task's surprise distribution.
3. **Instrument the pipeline**: log every axiom crystallized and every axiom retrieved into
   context. This is the *proof the mechanism fired* — the non-negotiable exit gate.
4. **Fix infra config drift**: bring up a clean Postgres on a known port via `.env`; verify
   `gemma3:4b` + `embeddinggemma` present; load `schema.sql`.
5. **Fair tuning pass**: sweep surprise thresholds to reasonable values (same logic applied to
   both arms where applicable); run ~3 seeds informally to confirm the direction of the result
   is stable, not single-run noise.

**Exit criterion:** logs demonstrate axioms being written AND injected, on sane thresholds.
Only then do we record the honest accuracy result. The result may remain "RAG wins" — that is
an acceptable, reportable outcome.

### Phase 2 — Reproducibility & code hygiene ("clean README + config")

1. Move port/secrets into `.env` consumed via `python-dotenv` (already a dependency). Remove
   hardcoded `5432` / `password` from `simulate.py` and `agent.py`.
2. Archive or delete `simulate_baseline.py` (superseded standalone prototype). `simulate.py`
   becomes the single canonical entry point.
3. Rewrite `README.md`: lead with the capability narrative, then Prerequisites → Step-by-step
   Run → Results. Readable by a non-specialist; runnable by someone willing to follow steps.
4. `git init` the repo (currently untracked) and make a clean initial commit. Add a
   `.gitignore` (exclude `.venv`, `.env`, `__pycache__`, large lock churn as appropriate).

### Phase 3 — Visuals & narrative

Regenerate `plot_results.py` to produce four artifacts:

1. **Cumulative accuracy / learning curve** — smoothed/cumulative accuracy per arm. The money
   shot: "who learned faster."
2. **Memory cost comparison** — rows stored / context size per arm; shows the efficiency
   tradeoff.
3. **A real crystallized axiom** — pull an actual `RULE: ...` from `semantic_core`, render as a
   quote block in the README. Concrete proof the mechanism works.
4. **Eigen-spectrum heatmap** — eigenvector evolution (the EDD's planned visual). A
   research-depth flex; clearly labeled as supplementary.

Add a short **"What I found / What I'd do next"** section stating the result plainly.

## 4. Success Criteria

- A stranger understands in ~60 seconds: what the agent does, that it was tested against a
  fair baseline, and what the honest result was.
- A stranger can clone and reproduce in ~10 minutes by following the README.
- The portfolio signal is **engineering judgment + intellectual honesty** (and systems
  engineering: Postgres/pgvector, surprise-gated memory, PCA consolidation) — not a
  manufactured win.
- The Eigen-Memory mechanism is provably exercised (logged axioms written + injected), so the
  experiment is a valid test of the hypothesis regardless of outcome.

## 5. Out of Scope (YAGNI)

- Full multi-seed statistical significance testing (we do informal multi-seed only).
- Makefile / one-command automation.
- New task environments beyond the integer "hidden rules" game.
- Redesigning the contrastive axiom-synthesis approach.

These are noted as "future work" in the README, not built now.
