"""Crystallize from post-change evidence only, and G4-score what comes out.

§9b's failure was not detection strength -- it was that the crystallizer's
60-trial window straddles the shift. Pre-shift successes sit in the success
buffer and dilute the contrast at every check but the last, and pre-shift
failures let the model write an accurate statement of the OLD rule (seed 42's
batch-7 axiom). The outcome detector (outcome_detector.py) locates the change
at +0 to +5 trials, so the window can be truncated to post-change evidence only.

Three arms per seed, identical apart from which trials the kernel sees:

  full       every trial up to the end of the stream (what §9 ran)
  truncated  only trials at/after the DETECTED change point
  oracle     only trials at/after the TRUE shift (upper bound on truncation;
             the gap between this and `truncated` is what detection error costs)

All three use REAL per-trial correctness, the fixed _contrast_sets projection,
and forced crystallization with no gate -- the question here is rule QUALITY
given evidence, not when to fire.

Scoring is mechanical, not a human reading prose: the rule is applied by the
executor to a probe set drawn from trials the crystallizer never saw, and we
report accuracy per polarity. A rule that maps both polarities correctly scores
high on both; a half-stale rule (§9b) scores high on one and near zero on the
other, which is exactly the failure a human G4 read was catching by eye.

Usage: uv run python truncated_crystallization.py [seed ...]
Writes results/shift/truncated_crystallization.json
"""

import json
import sys

import numpy as np
from openai import OpenAI

from gate_roc import _NullConn
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from src.dataset import load_shift, shift_rules
from src.eigen_memory_agent.memory_kernel import EigenMemoryKernel
from src import paths

EXECUTOR = "gemma4:12b"
EXTRA_BODY = {"reasoning_effort": "none"}
KCFG = {"window": 60, "contrast_on": "embedding_mean",
        "consecutive_detections": 3, "stability_cos": 0.5}
SEEDS = [int(a) for a in sys.argv[1:]] or [2, 7, 18, 23, 42]
TRUE_SHIFT = 100
PROBE_N = 12          # held-out-ish probe per polarity, kept small: 24 calls/arm


class _AxiomConn(_NullConn):
    def __init__(self):
        super().__init__()
        self.rows = []

    def cursor(self):
        conn = self
        base = super().cursor()

        class _Cur(type(base)):
            def execute(self, sql, params=None):
                if "INSERT INTO semantic_core" in sql:
                    conn.inserts += 1
                    conn.rows.append(params)

        return _Cur()


def _crystallize_from(client, trials, embs, correct, seed, start):
    """Force one crystallization using only trials[start:]."""
    conn = _AxiomConn()
    kernel = EigenMemoryKernel(conn, client, model=EXECUTOR, extra_body=EXTRA_BODY,
                               rng_seed=seed, **KCFG)
    for i in range(start, len(trials)):
        if i == 0:
            continue
        # Nearest earlier neighbour must also come from the truncated region,
        # or the residual reintroduces the pre-shift evidence we just excluded.
        lo = max(start, 1)
        if i <= lo:
            continue
        nn = lo - 1 + int(np.argmax(embs[lo - 1:i] @ embs[i]))
        kernel.observe(embedding=embs[i], residual=embs[i] - embs[nn],
                       was_correct=bool(correct[i]), context_input=trials[i]["input"],
                       prediction=trials[nn]["label"], actual=trials[i]["label"])
    if (len(kernel.fail_records) < 5 or len(kernel.succ_records) < 3):
        return None, kernel
    F, S, basis = kernel._reduced_residuals()
    from src.eigen_memory_agent.memory_kernel import _mean_contrast, _unit
    lam1, v_red = _mean_contrast(F, S)
    kernel._crystallize(_unit(basis.T @ v_red), strength=1.0)
    rule = conn.rows[-1][0] if conn.inserts else None
    return rule, kernel


def _score(client, rule, probe, labels):
    """Apply the rule to probe items; return accuracy overall and per polarity."""
    if not rule:
        return None
    per = {}
    for pol, items in probe.items():
        hits = 0
        for t in items:
            prompt = (f"{rule}\n\nApply the rule above. Answer with exactly one "
                      f"of {', '.join(labels)} and nothing else.\n\n"
                      f"Input: {t['input']}\nAnswer:")
            try:
                r = client.chat.completions.create(
                    model=EXECUTOR, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, seed=0, max_tokens=8, extra_body=EXTRA_BODY)
                out = (r.choices[0].message.content or "").upper()
            except Exception:
                out = ""
            pred = next((l for l in labels if l.upper() in out), None)
            hits += (pred == t["label"])
        per[pol] = hits / max(len(items), 1)
    return per


def run_seed(client, seed, change_points):
    art = paths.shift(f"comparison_results.shift.{seed}.json")
    with open(art) as f:
        eigen = json.load(f)["arms"]["Treatment_Eigen"]
    correct = eigen.get("trial_correct")
    if not correct:
        print(f"seed {seed}: no trial_correct; skipping")
        return None

    trials, heldout = load_shift(seed=seed)
    n = min(len(correct), len(trials))
    trials, correct = trials[:n], correct[:n]
    pre, post = shift_rules(seed)
    labels = sorted({t["label"] for t in trials} | {t["label"] for t in heldout})

    embs = np.array([
        client.embeddings.create(input=t["input"], model=EMBEDDING_MODEL)
        .data[0].embedding for t in trials])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    # Probe from the HELD-OUT split, split by polarity, so scoring never touches
    # trials the crystallizer saw. Polarity = which planted rule row applies.
    probe = {"request": [], "report": []}
    for t in heldout:
        pol = "request" if t["label"] == post["request"] else "report"
        if len(probe[pol]) < PROBE_N:
            probe[pol].append(t)

    cp = change_points.get(seed, TRUE_SHIFT)
    arms = {"full": 0, "truncated": cp, "oracle": TRUE_SHIFT}
    print(f"\n===== seed {seed}  (detected change {cp}, true {TRUE_SHIFT}) =====")
    print(f"  planted post: request->{post['request']}  report->{post['report']}"
          f"   (stale pre: request->{pre['request']})")

    out = {"seed": seed, "change_point": cp, "pre_rule": pre, "post_rule": post,
           "arms": {}}
    for name, start in arms.items():
        rule, _ = _crystallize_from(client, trials, embs, correct, seed, start)
        scores = _score(client, rule, probe, labels)
        out["arms"][name] = {"start": start, "rule": rule, "scores": scores}
        if not rule:
            print(f"  {name:10s}: no axiom (insufficient evidence)")
            continue
        s = scores or {}
        both = all(v >= 0.5 for v in s.values()) if s else False
        print(f"  {name:10s}: request={s.get('request', float('nan')):.2f} "
              f"report={s.get('report', float('nan')):.2f} "
              f"{'BOTH-OK' if both else 'one-sided'}")
        print(f"              {rule[:96]}")
    return out


def main():
    with open(paths.shift("outcome_detection.json")) as f:
        det = json.load(f)
    cps = {r["seed"]: (r["per_label_fired"] or TRUE_SHIFT) for r in det["seeds"]}

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                    timeout=120.0, max_retries=3)
    rows = [r for r in (run_seed(client, s, cps) for s in SEEDS) if r]

    print(f"\n{'seed':>5} {'arm':>10} {'request':>8} {'report':>8}  both-ok")
    tally = {}
    for r in rows:
        for name, a in r["arms"].items():
            s = a.get("scores") or {}
            ok = bool(s) and all(v >= 0.5 for v in s.values())
            tally[name] = tally.get(name, 0) + ok
            print(f"{r['seed']:>5} {name:>10} "
                  f"{s.get('request', float('nan')):>8.2f} "
                  f"{s.get('report', float('nan')):>8.2f}  {'yes' if ok else 'no'}")
    print("\nrules correct on BOTH polarities:")
    for name in ("full", "truncated", "oracle"):
        if name in tally:
            print(f"  {name:10s} {tally[name]}/{len(rows)}")

    with open(paths.shift("truncated_crystallization.json"), "w") as f:
        json.dump({"probe_n": PROBE_N, "seeds": rows}, f, indent=2)
    print("\nwrote results/shift/truncated_crystallization.json")


if __name__ == "__main__":
    main()
