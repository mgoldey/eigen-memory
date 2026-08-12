#!/usr/bin/env bash
# Rerun ONLY seed 23's Treatment_Eigen arm, to capture `trial_correct` and
# feed gate_replay.py (docs/NEXT_EXPERIMENT.md §9a).
#
# The driver resumes: it skips arms already present in the artifact. So to
# rerun one arm we write a checkpoint holding the OTHER FOUR and let resume
# skip them. The four control arms are untouched by this question -- they
# don't feed a gate -- so reusing their committed results is correct, not a
# shortcut, and it turns a 6 h rerun into a 1 h one.
#
# Usage: scratch/replay_seed23.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ART=results/shift/comparison_results.shift.23.json
ORIG=results/shift/comparison_results.shift.23.pre-trialcorrect.json
LOG=shift_pilot_23_replay.log

if ! git diff --quiet HEAD -- "$ART" 2>/dev/null; then
  echo "FATAL: $ART differs from HEAD; resolve before rerunning." >&2
  exit 1
fi

# Keep the original as the record of the run that produced the §8 numbers.
cp "$ART" "$ORIG"
echo "original preserved -> $ORIG"

# Drop Treatment_Eigen so the driver reruns exactly that arm.
python3 - "$ART" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
removed = d["arms"].pop("Treatment_Eigen", None)
assert removed is not None, "Treatment_Eigen already absent"
json.dump(d, open(p, "w"), indent=2)
print("checkpoint now holds:", sorted(d["arms"]))
PY

docker compose up -d
for i in $(seq 1 60); do
  docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 && break
  sleep 1
done
docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 \
  || { echo "FATAL: postgres not ready" >&2; exit 1; }

curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null \
  || { echo "FATAL: ollama not reachable" >&2; exit 1; }
curl -s --max-time 10 http://localhost:11434/api/tags \
  | grep -q 'gemma4:12b' || { echo "FATAL: gemma4:12b not pulled" >&2; exit 1; }

echo "== $(date '+%F %T') rerunning seed 23 Treatment_Eigen; log: $LOG =="
uv run python run_shift_experiment.py 23 >>"$LOG" 2>&1
echo "== $(date '+%F %T') finished; exit $? =="
