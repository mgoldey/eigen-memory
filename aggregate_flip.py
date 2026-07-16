"""Aggregate multi-seed flip results (comparison_results.flip.<seed>.json).

Reports per-arm mean +/- std across seeds, the paired H1 check
(Eigen > max(RAG, Baseline) per seed), and the C5 executor gate. C5 requires
Oracle > copy ceiling (the 3-class nearest-neighbor label-copy accuracy from
guardrail.flip.<seed>.json — NOT the 2-way polarity match m, which lives on a
different scale). Oracle below the ceiling means the executor applies a true
rule worse than blind copying scores, so rule-memory cannot win.

Reads guardrail.flip.<seed>.json (run guardrail_flip.py per seed first) —
guardrail numbers are consumed from artifacts, never hand-transcribed.

Usage: uv run python aggregate_flip.py <seed> [<seed> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

ARMS = ["Baseline", "Oracle_Rule", "Control_RAG", "Treatment_Eigen"]


def _load(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Missing {path} — run the corresponding script for that seed first.")
    return json.loads(p.read_text())


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [42, 2, 18, 23]
    runs = {s: _load(f"comparison_results.flip.{s}.json") for s in seeds}
    guards = {s: _load(f"guardrail.flip.{s}.json") for s in seeds}

    n_test = runs[seeds[0]]["n_test"]
    se_item = 0.5 / np.sqrt(n_test)  # worst-case SE of one proportion at n_test
    print(f"=== Multi-seed flip aggregate ({len(seeds)} seeds: {seeds}; "
          f"n_test={n_test}, per-seed SE up to ~{se_item:.3f}) ===\n")

    header = (f"{'seed':>6} | " + " | ".join(f"{a:>15}" for a in ARMS)
              + " | axioms | copy_ceil | Oracle>ceil?")
    print(header)
    print("-" * len(header))
    per_arm = {a: [] for a in ARMS}
    h1_wins, c5_passes = [], []
    for s in seeds:
        arms = runs[s]["arms"]
        for a in ARMS:
            per_arm[a].append(arms[a]["test_acc"])
        ceiling = guards[s]["copy_acc"]
        oracle = arms["Oracle_Rule"]["test_acc"]
        eig, rag, base = (arms["Treatment_Eigen"]["test_acc"],
                          arms["Control_RAG"]["test_acc"], arms["Baseline"]["test_acc"])
        h1_wins.append(eig > max(rag, base))
        c5_passes.append(oracle > ceiling)
        print(f"{s:>6} | " + " | ".join(f"{arms[a]['test_acc']:15.3f}" for a in ARMS)
              + f" | {arms['Treatment_Eigen']['n_axioms']:>6} | {ceiling:9.3f}"
              + f" | {'yes' if oracle > ceiling else 'NO'}")

    print("\n--- Across seeds (mean +/- std) ---")
    for a in ARMS:
        v = np.array(per_arm[a])
        print(f"  {a:16s}: {v.mean():.3f} +/- {v.std(ddof=1):.3f}")

    rag_v = np.array(per_arm["Control_RAG"])
    eig_v = np.array(per_arm["Treatment_Eigen"])
    orc_v = np.array(per_arm["Oracle_Rule"])
    ceil_v = np.array([guards[s]["copy_acc"] for s in seeds])

    n = len(seeds)
    sign_p = 0.5 ** n  # one-sided sign test if one direction wins every seed
    print(f"\nH1 (Eigen > max(RAG, Baseline)) per seed: {sum(h1_wins)}/{n} "
          f"-> {'SUPPORTED' if all(h1_wins) else 'NOT SUPPORTED'}")
    eigen_minus_rag = eig_v - rag_v
    print(f"  paired Eigen - RAG: {eigen_minus_rag.mean():+.3f} +/- {eigen_minus_rag.std(ddof=1):.3f} "
          f"(per seed: {[f'{x:+.3f}' for x in eigen_minus_rag]})")
    print(f"  (a unanimous {n}/{n} direction would be sign-test p = {sign_p:.3f}; "
          f"treat sub-unanimous counts as descriptive only)")

    print(f"\nC5 executor gate (requires Oracle > copy ceiling): "
          f"passes {sum(c5_passes)}/{n} seeds "
          f"-> {'GATE FAILS ON EVERY SEED — rule-memory could not win regardless of axiom quality' if not any(c5_passes) else 'gate passes on some seeds'}")
    oracle_minus_ceiling = orc_v - ceil_v
    print(f"  paired Oracle - ceiling: {oracle_minus_ceiling.mean():+.3f} "
          f"+/- {oracle_minus_ceiling.std(ddof=1):.3f} "
          f"(per seed: {[f'{x:+.3f}' for x in oracle_minus_ceiling]})")

    print("\n--- Health (from the run artifacts) ---")
    for s in seeds:
        arms = runs[s]["arms"]
        eig_arm = arms["Treatment_Eigen"]
        print(f"  seed {s}: buffer={eig_arm.get('buffer_size', '?')} "
              f"nll_missing_rate={eig_arm.get('nll_missing_rate', '?')} "
              f"parse_fallback_rate={eig_arm.get('parse_fallback_rate', '?')} "
              f"axioms={eig_arm['n_axioms']}")
        for ax in eig_arm.get("axioms", []):
            print(f"    (strength {ax['strength']:.2f}) {ax['rule'][:140]}")

    out = {
        "seeds": seeds,
        "n_test": n_test,
        "per_arm_mean": {a: float(np.mean(per_arm[a])) for a in ARMS},
        "per_arm_std": {a: float(np.std(per_arm[a], ddof=1)) for a in ARMS},
        "h1_wins": int(sum(h1_wins)),
        "c5_passes": int(sum(c5_passes)),
        "copy_ceilings": {s: guards[s]["copy_acc"] for s in seeds},
        "paired_eigen_minus_rag": [float(x) for x in eigen_minus_rag],
        "paired_oracle_minus_ceiling": [float(x) for x in oracle_minus_ceiling],
    }
    Path("comparison_results.flip.aggregate.json").write_text(json.dumps(out, indent=2))
    print("\nWrote comparison_results.flip.aggregate.json")


if __name__ == "__main__":
    main()
