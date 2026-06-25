# Candidate Datasets for a Fair Re-Test

The number-game failed condition **C1** (the rule must be visible in embedding space; see
[USE_CASES.md](USE_CASES.md)) and **C2** (must generalize to held-out inputs). A fair re-test
needs a **hidden-rule short-text classification** task: short inputs that embed well, a small
label set, a learnable hidden rule, and a clean train/test split.

This document lists candidates, prioritizing ones that are **tiny, permissively licensed, and
trivially loadable**. Options marked ✅ **verified** were actually loaded on this machine.

## Recommended: TREC question classification ✅ verified

The best fit found. Short questions, a genuine hidden rule (question *type*), and an official
held-out test set.

| Property | Value |
|----------|-------|
| Task | classify a question by what it asks for |
| Classes | 6 coarse: `DESC, ENTY, ABBR, HUM, NUM, LOC` (or use a 3-class subset to mirror RED/BLUE/GREEN) |
| Avg length | **10.2 words/question** — ideal for embedding |
| Size | 5,452 train / **500 official test** |
| Split | separate train/test files → real generalization test (C2 ✓) |
| Loadable | one HTTP fetch, no library beyond stdlib; also on HF as `trec` |
| License | research use (UIUC Cognitive Computation Group) |

Why it fits: question *type* is a semantic property that sentence embeddings represent well
(C1 ✓); the official test set is unseen at train time (C2 ✓); 6 small stable classes (C4 ✓).

Loading (stdlib only, verified):

```python
import urllib.request, ssl
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

def load_trec(url):
    txt = urllib.request.urlopen(url, timeout=10, context=ctx).read().decode("latin-1")
    rows = []
    for line in txt.strip().split("\n"):
        label, _, q = line.partition(" ")
        rows.append((label.split(":")[0], q))   # (coarse_label, question)
    return rows

train = load_trec("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label")
test  = load_trec("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label")
```

To mirror the current 3-class loop, keep e.g. `{HUM, LOC, NUM}` and map to three labels.

## Fallback: 20 Newsgroups (3-class subset) ✅ verified

Bundled with scikit-learn (already installed) — works fully offline, zero new dependencies.

| Property | Value |
|----------|-------|
| Task | classify a post by topic |
| Classes | pick any subset, e.g. `rec.sport.baseball, sci.space, talk.politics.guns` |
| Avg length | ~179 words (median 80) — **longer than ideal** |
| Size | ~1,700 train (3-class subset) + a test split |
| Split | `subset='train'` / `subset='test'` built in (C2 ✓) |
| License | public, ships with sklearn |

Honest caveat: posts are long and rambling even with headers/quotes stripped, so they embed
less cleanly than TREC's one-line questions. Topic is still a strong embedding-space signal, so
C1 holds, but TREC is the better substrate match. Use this only if offline / no-network is a
hard requirement.

```python
from sklearn.datasets import fetch_20newsgroups
cats = ["rec.sport.baseball", "sci.space", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers", "footers", "quotes"))
```

## Other strong short-text options (require `pip install datasets`)

Not loaded here (HF `datasets` isn't in this venv), but well-known good fits — listed for when
network + HF are available. Verify IDs/licenses before use.

| HF id | Task | ~Classes | Length | Fit note |
|-------|------|---------|--------|----------|
| `ag_news` | news topic | 4 | ~30 words (headline+blurb) | Strong; large, subsample it |
| `emotion` | tweet emotion | 6 | ~15 words | Strong; clean short text |
| `tweet_eval` (subsets) | sentiment/emotion/etc. | varies | tweet-length | Strong; many sub-tasks |
| `banking77` | banking intent | 77 | one sentence | Great C1; 77 classes may be too many |
| `clinc_oos` | intent detection | 150 (+oos) | one sentence | Great C1; designed for few-shot |
| `dbpedia_14` | ontology topic | 14 | ~1-2 sentences | Strong; large, subsample it |

## How any of these slots into the current code

`src/dataset.py` is the only task-specific file. Swap `generate_dataset()` to return
`[{"input": text, "label": coarse_label}, ...]` from one of the above, pick a 3–6 class subset,
and the agent loop, surprise gating, and PCA kernel work unchanged. The crucial difference from
the number-game: **evaluate on the held-out test split with memory frozen** (see
[VALID_EXPERIMENT.md](VALID_EXPERIMENT.md)), so generalization — not memorization — is measured.

<!-- RESEARCH_APPENDIX: enriched from the dataset deep-research run when it completes -->
