"""RFμ — the Rule-Following Microbenchmark (docs/NEXT_EXPERIMENT.md §3).

Qualifies an executor for the Rule-Shift experiment BEFORE burning a full run.
gemma3:4b failed C5 (applies a true rule worse than copying scores); this asks,
cheaply and paired, whether a candidate clears the bar.

60 items (flip-generator surface text, fresh GLOBAL polarity rule — the same
rule family Rule-Shift will use), three conditions on identical items:

  R  — the true rule pasted (two formats: table and prose; scored as the max),
       no exemplars.
  C  — 5 labeled exemplars, no rule. Exemplar polarity matches the query at the
       cross-split rate measured by the corrected guardrail (~0.8), so this is
       the copy ceiling as the model actually realizes it.
  RC — the true rule PLUS stale exemplars labeled under an outdated rule whose
       request-row differs (the exact condition Rule-Shift creates). A model
       that trusts exemplars over the rule fails here.

Qualification gate (pre-registered):
  acc(R) - acc(C) >= +0.10  AND  acc(RC) >= acc(R) - 0.10,
  with McNemar (exact binomial on discordant pairs) p < 0.05 for R vs C.

Usage: uv run python run_rfmu.py [model] [seed]
       (default model gemma4:12b, seed 7 — a seed never used by the main runs)
Writes rfmu.<model>.json. ~240 short calls; minutes, not hours.
"""

import json
import random
import sys
import urllib.request

from scipy.stats import binomtest

from src.config import OLLAMA_BASE_URL
from src.dataset import load_flip
from src import paths

# Native Ollama endpoint (not the OpenAI-compat one): thinking-family models
# (gemma4, qwen3.5) burn the whole completion budget on a reasoning stream the
# OpenAI-compat API can't disable — content comes back EMPTY and the run scores
# a fake zero. The native API takes "think": false.
NATIVE_CHAT_URL = OLLAMA_BASE_URL.replace("/v1", "") + "/api/chat"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
N_ITEMS = 60
K_EXEMPLARS = 5
POLARITY_MATCH_RATE = 0.8  # corrected guardrail's cross-split range is 0.73-0.89

LABELS = ["ESCALATE", "FILE", "DEFER"]
# Fresh global rule (Rule-Shift's family): label depends ONLY on polarity.
CURRENT_RULE = {"request": "DEFER", "report": "FILE"}
# The stale rule for RC differs on the REQUEST row only (a partial shift).
STALE_RULE = {"request": "ESCALATE", "report": "FILE"}

RULE_TABLE_FMT = """Current routing rule (authoritative, as of today):
| kind of message | label |
|---|---|
| request / needs action | {request} |
| report / already done | {report} |"""

RULE_PROSE_FMT = ("Current routing rule (authoritative, as of today): messages that request "
                  "action or describe unresolved work are labeled {request}; messages that "
                  "report completed work or status are labeled {report}.")


def _fmt_rule(fmt):
    return fmt.format(**CURRENT_RULE)


def build_items(seed):
    """60 relabeled flip-generator messages + per-item exemplar sets."""
    rng = random.Random(f"rfmu-{seed}")
    pool = load_flip(split="test", num_samples=N_ITEMS * 4, seed=seed)
    for d in pool:
        d["label"] = CURRENT_RULE[d["meta"]["polarity"]]

    items, exemplar_pool = pool[:N_ITEMS], pool[N_ITEMS:]
    by_pol = {"request": [], "report": []}
    for d in exemplar_pool:
        by_pol[d["meta"]["polarity"]].append(d)

    for item in items:
        pol = item["meta"]["polarity"]
        other = "report" if pol == "request" else "request"
        exemplars = []
        for _ in range(K_EXEMPLARS):
            src = pol if rng.random() < POLARITY_MATCH_RATE else other
            exemplars.append(rng.choice(by_pol[src]))
        item["exemplars"] = exemplars
    return items


def _exemplar_block(exemplars, rule):
    return "\n".join(
        f"- {e['input']} -> {rule[e['meta']['polarity']]}" for e in exemplars
    )


def prompt_for(item, condition, rule_fmt=RULE_TABLE_FMT):
    label_str = ", ".join(LABELS)
    parts = []
    if condition in ("R", "RC"):
        parts.append(_fmt_rule(rule_fmt))
    if condition == "C":
        parts.append("Past labeled messages:\n" + _exemplar_block(item["exemplars"], CURRENT_RULE))
    if condition == "CS":
        # C-stale (2026-07-16 amendment): exemplars labeled under the OUTDATED
        # rule, with no hint they are stale — what post-shift retrieval actually
        # serves a copy arm. This, not C, is the realized copy ceiling.
        parts.append("Past labeled messages:\n" + _exemplar_block(item["exemplars"], STALE_RULE))
    if condition == "RC":
        parts.append("Past labeled messages (may be outdated):\n"
                     + _exemplar_block(item["exemplars"], STALE_RULE))
    parts.append(f"Message: {item['input']}")
    parts.append(f"Reply with exactly one word, one of: {label_str}.")
    return "\n\n".join(parts)


def _chat(prompt):
    """One native-API call, thinking disabled, deterministic."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 16},
    }
    req = urllib.request.Request(
        NATIVE_CHAT_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    body = json.loads(urllib.request.urlopen(req, timeout=300).read())
    if "error" in body and "think" in str(body["error"]).lower():
        # non-thinking model: retry without the think key
        del payload["think"]
        req = urllib.request.Request(
            NATIVE_CHAT_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return body.get("message", {}).get("content", "") or ""


def run_condition(items, condition, rule_fmt=RULE_TABLE_FMT, empties=None):
    correct = []
    empties = [] if empties is None else empties
    for item in items:
        raw = _chat(prompt_for(item, condition, rule_fmt)).strip().upper()
        pred = next((lab for lab in LABELS if lab in raw), "NONE")
        correct.append(pred == item["label"])
        if pred == "NONE":
            empties.append(raw[:40])
    return correct


def mcnemar_p(a, b):
    """Exact binomial test on discordant pairs, one-sided (a better than b)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return 1.0
    return float(binomtest(a_only, n, 0.5, alternative="greater").pvalue)


def main():
    items = build_items(SEED)
    print(f"RFμ: model={MODEL} seed={SEED} items={N_ITEMS} "
          f"(polarity-match rate {POLARITY_MATCH_RATE})", flush=True)

    empties = []
    r_table = run_condition(items, "R", RULE_TABLE_FMT, empties=empties)
    print(f"  R (table rule): {sum(r_table)/N_ITEMS:.3f}", flush=True)
    r_prose = run_condition(items, "R", RULE_PROSE_FMT, empties=empties)
    print(f"  R (prose rule): {sum(r_prose)/N_ITEMS:.3f}", flush=True)
    r_best, r_best_fmt = max(
        [(r_table, "table"), (r_prose, "prose")], key=lambda t: sum(t[0])
    )
    c = run_condition(items, "C", empties=empties)
    print(f"  C (copy only) : {sum(c)/N_ITEMS:.3f}", flush=True)
    cs = run_condition(items, "CS", empties=empties)
    print(f"  CS (stale copy): {sum(cs)/N_ITEMS:.3f}", flush=True)
    rc = run_condition(items, "RC",
                       RULE_TABLE_FMT if r_best_fmt == "table" else RULE_PROSE_FMT,
                       empties=empties)
    print(f"  RC (conflict) : {sum(rc)/N_ITEMS:.3f}", flush=True)

    # A no-label answer is a harness problem, not a model score — refuse to emit
    # a verdict built on unparsed replies (this exact failure produced a fake
    # RC=0.000 on the first run).
    unparsed_rate = len(empties) / (5 * N_ITEMS)
    print(f"  unparsed-answer rate: {unparsed_rate:.3f} ({len(empties)}/{4*N_ITEMS})")
    assert unparsed_rate <= 0.05, (
        f"unparsed rate {unparsed_rate:.2f} too high to score; sample: {empties[:5]}")

    acc = {"R_table": sum(r_table) / N_ITEMS, "R_prose": sum(r_prose) / N_ITEMS,
           "R": sum(r_best) / N_ITEMS, "C": sum(c) / N_ITEMS,
           "CS": sum(cs) / N_ITEMS, "RC": sum(rc) / N_ITEMS}

    # Original (2026-07-16 pre-registration): R vs C. Kept for the record, but
    # C saturates (a strong model INDUCES a global rule from 5 exemplars), so
    # this gate is nearly unpassable — see the amendment in NEXT_EXPERIMENT.md.
    p_c = mcnemar_p(r_best, c)
    strict = (acc["R"] - acc["C"] >= 0.10) and (acc["RC"] >= acc["R"] - 0.10) and p_c < 0.05

    # Amended Rule-Shift gate: the realized copy arm is CS (stale exemplars, no
    # staleness hint) — what post-shift retrieval actually serves.
    p_cs = mcnemar_p(r_best, cs)
    g_rule = acc["R"] >= 0.90
    g_conflict = acc["RC"] >= acc["R"] - 0.10
    g_sig = p_cs < 0.05
    amended = g_rule and g_conflict and g_sig

    print(f"\n  strict gate (R - C >= +0.10, RC, McNemar R>C): "
          f"{'PASS' if strict else 'FAIL'} (R-C = {acc['R'] - acc['C']:+.3f}, p = {p_c:.4f})")
    print(f"  amended gate — R >= 0.90: {acc['R']:.3f} -> {'ok' if g_rule else 'FAIL'}")
    print(f"  amended gate — RC >= R - 0.10: {acc['RC'] - acc['R']:+.3f} -> {'ok' if g_conflict else 'FAIL'}")
    print(f"  amended gate — McNemar R > CS: p = {p_cs:.4f} "
          f"(R {acc['R']:.3f} vs CS {acc['CS']:.3f}) -> {'ok' if g_sig else 'FAIL'}")
    print(f"\n  VERDICT: {MODEL} "
          f"{'QUALIFIES (amended Rule-Shift gate)' if amended else 'DOES NOT QUALIFY'}"
          f"{' and passes the strict gate too' if strict else ''}")

    safe = MODEL.replace(":", "_").replace("/", "_")
    with open(paths.calibration(f"rfmu.{safe}.json"), "w") as f:
        json.dump({"model": MODEL, "seed": SEED, "n_items": N_ITEMS,
                   "polarity_match_rate": POLARITY_MATCH_RATE, "acc": acc,
                   "r_best_format": r_best_fmt,
                   "mcnemar_p_R_gt_C": p_c, "mcnemar_p_R_gt_CS": p_cs,
                   "strict_gate": strict,
                   "amended_gates": {"rule": g_rule, "conflict": g_conflict,
                                     "significance": g_sig},
                   "qualified_amended": amended}, f, indent=2)
    print(f"  Wrote rfmu.{safe}.json")


if __name__ == "__main__":
    main()
