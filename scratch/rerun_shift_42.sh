#!/usr/bin/env bash
# Relaunch the seed-42 Rule-Shift run on the fixed RULE:-extraction path.
#
# Why this script exists: run_shift_experiment.py RESUMES. It reads the
# committed artifact and skips arms already present, so pointing it at a
# complete results/shift/comparison_results.shift.42.json makes it exit
# having run nothing. The artifact must be moved aside first — and moved
# aside in a way that keeps it, because it is the committed record of what
# the buggy extractor produced.
#
# Usage: scratch/rerun_shift_42.sh
# Expect 6+ hours (measured 2026-08-10: ~9 s/executor call, ~2050 calls).

set -euo pipefail
cd "$(dirname "$0")/.."

ART=results/shift/comparison_results.shift.42.json
ARCHIVE=results/shift/comparison_results.shift.42.cotresidue.json
LOG=shift_pilot_42_rerun.log

# The pre-fix run must be preserved before anything else touches $ART.
# Once $ARCHIVE exists it IS that record, and $ART is a post-fix artifact --
# possibly a partial checkpoint we want to resume from, so do NOT compare it
# to HEAD. (HEAD still holds the buggy-extractor run; `git checkout` on this
# path would destroy a checkpoint, which is why that guard is gone.)
if [ ! -f "$ARCHIVE" ]; then
  # First run: $ART is still the committed pre-fix record. Refuse if it is
  # dirty, so an aborted run stays recoverable, then move it aside.
  if ! git diff --quiet HEAD -- "$ART" 2>/dev/null; then
    echo "FATAL: $ART differs from HEAD and $ARCHIVE does not exist." >&2
    echo "Cannot tell a checkpoint from a corrupted record. Resolve by hand." >&2
    exit 1
  fi
fi

echo "== $(date '+%F %T') starting seed-42 rerun =="

# Postgres
docker compose up -d
for i in $(seq 1 60); do
  docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 && break
  sleep 1
done
docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 \
  || { echo "FATAL: postgres not ready" >&2; exit 1; }

# Ollama + the executor this experiment requires
curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null \
  || { echo "FATAL: ollama not reachable on :11434" >&2; exit 1; }
curl -s --max-time 10 http://localhost:11434/api/tags \
  | grep -q 'gemma4:12b' || { echo "FATAL: gemma4:12b not pulled" >&2; exit 1; }

# Archive the pre-fix artifact so the rerun starts from zero arms -- but only
# once. If $ARCHIVE already exists, $ART is a post-fix checkpoint and the
# driver's resume logic should keep the arms it already has.
if [ ! -f "$ARCHIVE" ] && [ -f "$ART" ]; then
  mv "$ART" "$ARCHIVE"
  echo "archived pre-fix artifact -> $ARCHIVE"
elif [ -f "$ART" ]; then
  echo "resuming from checkpoint; arms already present:"
  python3 -c "import json;print('  '+', '.join(sorted(json.load(open('$ART'))['arms'])))"
fi

echo "== running remaining arms; log: $LOG =="
# Append, don't truncate: the crashed 2026-08-11 run's log is the only record
# of the three control arms' batch-by-batch behaviour.
uv run python run_shift_experiment.py 42 >>"$LOG" 2>&1
echo "== $(date '+%F %T') finished; exit $? =="
