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

# Refuse to run on a dirty tree: this script moves a committed artifact, and
# an aborted run must be recoverable with `git checkout`.
if ! git diff --quiet HEAD -- "$ART" 2>/dev/null; then
  echo "FATAL: $ART differs from HEAD. Resolve before rerunning." >&2
  exit 1
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

# Archive the pre-fix artifact so the rerun starts from zero arms.
if [ -f "$ART" ]; then
  mv "$ART" "$ARCHIVE"
  echo "archived pre-fix artifact -> $ARCHIVE"
fi

echo "== running all 5 arms; log: $LOG =="
uv run python run_shift_experiment.py 42 >"$LOG" 2>&1
echo "== $(date '+%F %T') finished; exit $? =="
