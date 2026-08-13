#!/usr/bin/env bash
# Run the v4 pipeline on the four seeds it was NOT tuned on.
#
# v4's parameters (change-point truncation, readiness 40) were chosen after
# watching v3 and v3b write stale axioms on seed 42. That is selection pressure,
# so seed 42's 0.967 shows the mechanism CAN work by design -- not that it
# generally does. This script is the honest test: identical configuration, no
# tuning, on seeds 2, 7, 18 and 23.
#
# Per seed only Treatment_Eigen runs; the four control arms are seeded from the
# committed artifact via the driver's resume path. They construct no kernel and
# cannot be affected by any of this, so reusing them is correct and turns a 6 h
# run into ~1 h.
#
# Idempotent: a seed whose v4 artifact already holds Treatment_Eigen is skipped.
#
# Usage: scratch/run_v4_seeds.sh [seed ...]     (default: 2 7 18 23)

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-}")
[ -z "${SEEDS[0]:-}" ] && SEEDS=(2 7 18 23)
LOG=shift_v4_seeds.log

echo "== $(date '+%F %T') v4 sweep starting; seeds: ${SEEDS[*]} ==" | tee -a "$LOG"

docker compose up -d >>"$LOG" 2>&1
for i in $(seq 1 60); do
  docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 && break
  sleep 1
done
docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 \
  || { echo "FATAL: postgres not ready" | tee -a "$LOG"; exit 1; }

# The retirement columns are added by schema.sql, but a database created before
# that commit will not have them and every run dies on the first axiom SELECT.
docker exec memory-db psql -U postgres -d memory_agent -c \
  "ALTER TABLE semantic_core ADD COLUMN IF NOT EXISTS retired BOOLEAN DEFAULT FALSE, \
   ADD COLUMN IF NOT EXISTS retired_at TIMESTAMP;" >>"$LOG" 2>&1

curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null \
  || { echo "FATAL: ollama unreachable" | tee -a "$LOG"; exit 1; }

for SEED in "${SEEDS[@]}"; do
  ART="results/shift/comparison_results.shift.v4.${SEED}.json"
  SRC="results/shift/comparison_results.shift.${SEED}.json"

  if [ -f "$ART" ] && python3 -c "import json,sys; \
      sys.exit(0 if 'Treatment_Eigen' in json.load(open('$ART'))['arms'] else 1)"; then
    echo "== seed $SEED already has a v4 Treatment_Eigen; skipping ==" | tee -a "$LOG"
    continue
  fi

  # Seed the checkpoint with the committed control arms so only the eigen arm runs.
  python3 - "$SRC" "$ART" >>"$LOG" 2>&1 <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
keep = {k: v for k, v in d["arms"].items() if k != "Treatment_Eigen"}
json.dump({**d, "arms": keep}, open(dst, "w"), indent=2)
print("  checkpoint holds:", sorted(keep))
PY

  # Idle models can starve the 12B of memory; release anything not ours.
  for M in $(curl -s --max-time 10 http://localhost:11434/api/ps \
             | python3 -c "import json,sys; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[]) if 'gemma4:12b' not in m['name'] and 'embeddinggemma' not in m['name']))" 2>/dev/null); do
    curl -s --max-time 20 http://localhost:11434/api/generate \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$M\",\"keep_alive\":0}" >/dev/null 2>&1
    echo "  released idle model $M" | tee -a "$LOG"
  done

  echo "== $(date '+%F %T') seed $SEED: running v4 Treatment_Eigen ==" | tee -a "$LOG"
  uv run python run_shift_experiment.py "$SEED" --v4 >>"$LOG" 2>&1
  echo "== $(date '+%F %T') seed $SEED: done (exit $?) ==" | tee -a "$LOG"
done

echo "== $(date '+%F %T') sweep complete ==" | tee -a "$LOG"
echo "--- summary ---" | tee -a "$LOG"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
print(f'{"seed":>5} {"v4":>7} {"live":>5} {"written":>8} {"retired":>8}  {"request":>8} {"report":>8}')
for f in sorted(glob.glob("results/shift/comparison_results.shift.v4.*.json"),
                key=lambda p: int(p.split(".")[-2])):
    d = json.load(open(f))
    a = d["arms"].get("Treatment_Eigen")
    if not a:
        continue
    p = a.get("test_by_polarity") or {}
    print(f'{d["seed"]:>5} {a["test_acc"]:>7.3f} {a.get("n_axioms", 0):>5} '
          f'{a.get("n_axioms_written", 0):>8} {a.get("n_axioms_retired", 0):>8}  '
          f'{p.get("request", float("nan")):>8.3f} {p.get("report", float("nan")):>8.3f}')
print()
print("Axioms written (G4-score these against each seed's planted post-shift rule):")
for f in sorted(glob.glob("results/shift/comparison_results.shift.v4.*.json"),
                key=lambda p: int(p.split(".")[-2])):
    d = json.load(open(f))
    a = d["arms"].get("Treatment_Eigen") or {}
    post = d.get("post_rule", {})
    print(f'  seed {d["seed"]}: planted request->{post.get("request")} report->{post.get("report")}')
    for ax in a.get("axioms", []) or [{"rule": "(none live)"}]:
        print(f'      {ax["rule"][:150]}')
PY
