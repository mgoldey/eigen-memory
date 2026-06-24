# Eigen-Memory Portfolio Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. This is a research repo; "tests" are primarily *does the experiment run* and *does the Eigen mechanism provably fire*, plus targeted assertions on bug fixes. Adapt TDD pragmatically.

**Goal:** Turn the eigen-memory research repo into a recruiter-friendly portfolio piece with a *valid* experiment (Eigen mechanism provably fires on fair thresholds), clean reproducibility, and clear visuals — reporting the honest result.

**Architecture:** Three phases — (1) debug the agent so surprise/crystallization actually work and are logged; (2) reproducibility + hygiene; (3) visuals + narrative. Git commit after each task. No push.

**Tech Stack:** Python 3.10+, OpenAI SDK → local Ollama (`gemma3:4b`, `embeddinggemma`), Postgres 16 + pgvector, scikit-learn IncrementalPCA, matplotlib, uv.

## Global Constraints

- Postgres: `pgvector/pgvector:pg16`, db `memory_agent`, user `postgres`, on host port **5432** (compose). Connection config from `.env`, never hardcoded secrets.
- Models: `gemma3:4b` (chat, logprobs), `embeddinggemma:latest` (768-dim embeddings). Confirmed present.
- Embedding dim must stay **768** to match `schema.sql`.
- "Fair shot" rule: fix bugs + tune thresholds reasonably + multi-seed; never cherry-pick seeds or hand-write the hidden rules.
- Commit after every task. No `git push`.
- Python interpreter: `.venv/bin/python` (uv-managed venv already present).

---

### Task 1: Capture reproducible before-baseline

**Files:**
- Create: `comparison_results.BEFORE.json` (snapshot, already copied)
- Reference: `simulate.py`

- [ ] **Step 1:** Confirm infra up: `docker exec memory-db pg_isready -U postgres` → "accepting connections".
- [ ] **Step 2:** Run `.venv/bin/python simulate.py`; save log. This is the unmodified "before" result for honest comparison.
- [ ] **Step 3:** Commit the before-snapshot.

---

### Task 2: Add config via .env (kill hardcoded port/secret)

**Files:**
- Create: `src/config.py`
- Modify: `simulate.py` (DB_STRING), `src/eigen_memory_agent/agent.py` (default db usage stays via passed string)
- Modify: `.env` (add `DB_HOST`, `DB_PORT`, `OLLAMA_BASE_URL`)

**Interfaces:**
- Produces: `src/config.py` exposing `get_db_string() -> str` and `OLLAMA_BASE_URL: str`, reading from env with sane defaults (`localhost:5432`, `http://localhost:11434/v1`).

- [ ] **Step 1:** Write `src/config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "memory_agent")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def get_db_string() -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```

- [ ] **Step 2:** In `simulate.py`, replace hardcoded `DB_STRING = "postgresql://..."` with `from src.config import get_db_string` and `DB_STRING = get_db_string()`.
- [ ] **Step 3:** In `agent.py`, default the Ollama base url from config instead of literal.
- [ ] **Step 4:** Add `DB_HOST=localhost`, `DB_PORT=5432`, `OLLAMA_BASE_URL=http://localhost:11434/v1` to `.env`.
- [ ] **Step 5:** Smoke: `.venv/bin/python -c "from src.config import get_db_string; print(get_db_string())"`.
- [ ] **Step 6:** Commit.

---

### Task 3: Fix surprise extraction bug (undefined target_lp)

**Files:**
- Modify: `src/eigen_memory_agent/agent.py:93-158` (`learn_batch`)

**Interfaces:**
- Produces: `learn_batch` always assigns a defined `s_pred` (NLL when true-label token found in top-k; explicit, logged fallback otherwise). No undefined-variable path.

- [ ] **Step 1:** Write a focused test `tests/test_surprise.py` that constructs a fake logprobs object where the true label is absent from top-k, and asserts `_extract_nll` returns a finite, large surprise (not a crash).

```python
# tests/test_surprise.py
from src.eigen_memory_agent.agent import _extract_nll

class _TL:
    def __init__(self, token, logprob): self.token=token; self.logprob=logprob

def test_nll_missing_token_returns_high_finite():
    tops = [_TL("GREEN", -0.1), _TL("RED", -2.0)]
    s = _extract_nll(tops, "BLUE")  # BLUE absent
    assert s is not None and s > 0 and s < 1e9
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_surprise.py -v` → FAIL (`_extract_nll` not defined).
- [ ] **Step 3:** Extract the inline NLL logic into a module-level helper `_extract_nll(top_logprobs, true_label) -> float` with a defined miss-path:

```python
import numpy as np

# Surprise (in nats) assigned when the true label is not in the top-k logprobs.
# Floor below top_logprobs[-1] so a "missing" token is at least as surprising
# as the least-likely observed token, without being infinite.
_MISSING_TOKEN_NLL = 7.0

def _extract_nll(top_logprobs, true_label):
    for tl in top_logprobs:
        if true_label.upper() in tl.token.upper():
            return float(-tl.logprob)
    return _MISSING_TOKEN_NLL
```

- [ ] **Step 4:** Replace the fragile loop in `learn_batch` to call `_extract_nll(...)`; remove the embedding-cosine fallback's role in masking missing logprobs (keep it only when logprobs object itself is absent).
- [ ] **Step 5:** Run pytest → PASS.
- [ ] **Step 6:** Commit.

---

### Task 4: Reconcile thresholds & instrument the mechanism

**Files:**
- Modify: `src/eigen_memory_agent/agent.py` (thresholds at lines ~141, ~151; add logging)
- Modify: `src/eigen_memory_agent/memory_kernel.py` (`_crystallize_axiom` logging)

**Interfaces:**
- Produces: named module constants `EPISODIC_WRITE_NLL`, `EIGEN_CRYSTALLIZE_NLL` (documented). Crystallization and axiom-injection emit `[AXIOM+]` / `[AXIOM→]` log lines so the mechanism is provably exercised.

- [ ] **Step 1:** Add constants at top of `agent.py`. Set both to a value the *measured* NLL distribution from Task 1's log actually crosses (inspect run_before.log NLL averages). Default proposal: `EPISODIC_WRITE_NLL = 1.0`, `EIGEN_CRYSTALLIZE_NLL = 1.5` — adjust to the observed distribution so axioms actually form. Document the reasoning in a comment.
- [ ] **Step 2:** Replace literal `1.5` / `2.0` with the constants.
- [ ] **Step 3:** In `kernel._crystallize_axiom`, on successful insert print `[AXIOM+] <first 60 chars>`. In `agent._retrieve_context`, when axioms are injected print `[AXIOM→] injected N axioms`.
- [ ] **Step 4:** Smoke-run a *single phase* (Treatment only) for ~30 trials via a throwaway snippet; confirm `[AXIOM+]` and `[AXIOM→]` appear in output. This is the **mechanism-fires exit gate**.
- [ ] **Step 5:** Commit.

---

### Task 5: Fair-tuning run + record honest result

**Files:**
- Create: `comparison_results.json` (regenerated), `results_seed*.json`
- Reference: `simulate.py`

- [ ] **Step 1:** Add a `--seed` / loop so `simulate.py` can run ~3 seeds (e.g., 42, 7, 123) and aggregate. Keep dataset generation seed-parameterized (already supported in `dataset.py`).
- [ ] **Step 2:** Run all 3 arms across 3 seeds. Capture per-seed and mean accuracy curves.
- [ ] **Step 3:** Inspect: does Eigen now beat / tie / lose to RAG on the mean? Record the honest answer in results JSON + a note file `FINDINGS.md`.
- [ ] **Step 4:** Commit results.

---

### Task 6: Reproducibility hygiene

**Files:**
- Delete: `simulate_baseline.py` (superseded prototype)
- Possibly delete: `simple_test.py`, `check_dim.py`, `test_logprobs.py` (ad-hoc scratch) — move to `scratch/` or remove if pure throwaway
- Modify: `README.md`

- [ ] **Step 1:** Remove `simulate_baseline.py`. Move ad-hoc scripts to `scratch/` (keep history, declutter root).
- [ ] **Step 2:** Rewrite README: capability-first narrative → Prerequisites (Docker, Ollama models) → Step-by-step run → Results (with images) → "What I found / future work".
- [ ] **Step 3:** Verify README commands actually work by following them mentally against current files.
- [ ] **Step 4:** Commit.

---

### Task 7: Visuals

**Files:**
- Modify: `plot_results.py`
- Create: `learning_curve.png`, `memory_cost.png`, `eigen_spectrum.png`

**Interfaces:**
- Consumes: `comparison_results.json` (multi-seed mean curves), `semantic_core` table (for a real axiom).

- [ ] **Step 1:** Generate cumulative/smoothed accuracy learning curve (mean across seeds, shaded variance band). Save `learning_curve.png`.
- [ ] **Step 2:** Generate memory-cost comparison bar/line. Save `memory_cost.png`.
- [ ] **Step 3:** Query `semantic_core` for a real crystallized axiom; write the text into README as a quote block.
- [ ] **Step 4:** Generate eigen-spectrum heatmap (top-k explained-variance evolution). Save `eigen_spectrum.png`. Label as supplementary.
- [ ] **Step 5:** Embed all images in README. Commit.

---

### Task 8: Final review & polish

- [ ] **Step 1:** Re-read README as a stranger: 60-second comprehension? 10-minute reproduce path correct?
- [ ] **Step 2:** Confirm `.env` not tracked; no secrets in committed files (`git grep -i password -- ':!*.md'`).
- [ ] **Step 3:** Final commit. Summarize outcome for the user.
```
