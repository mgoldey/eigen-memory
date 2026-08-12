#!/usr/bin/env bash
# Replay the remaining gate-shut seeds (2, 18, 7) to confirm the seed-23
# finding: given REAL per-trial correctness over unchanged featurization, the
# contrast statistic still sits below the noise edge -- i.e. the 1-of-5 fire
# rate is a featurization problem, not a label-noise one.
#
# Per seed: rerun ONLY Treatment_Eigen (to capture `trial_correct`), then
# gate_replay it. The four control arms are reused from the committed run via
# the driver's resume path -- they don't feed a gate, so reusing them is
# correct and turns a 6 h rerun into ~1 h.
#
# Idempotent: a seed whose artifact already carries trial_correct is skipped,
# so a re-invocation after a crash resumes rather than redoing work.
#
# Usage: scratch/replay_shut_seeds.sh [seed ...]   (default: 2 18 7)

set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-}")
[ -z "${SEEDS[0]:-}" ] && SEEDS=(2 18 7)
LOG=shift_replay_shut.log

echo "== $(date '+%F %T') replay run starting; seeds: ${SEEDS[*]} ==" | tee -a "$LOG"

# Services. memory-db carries restart:unless-stopped, so a reboot brings it
# back, but bring it up explicitly in case the box came up cold.
docker compose up -d >>"$LOG" 2>&1
for i in $(seq 1 60); do
  docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1 && break
  sleep 1
done
if ! docker exec memory-db pg_isready -U postgres -d memory_agent >/dev/null 2>&1; then
  echo "FATAL: postgres not ready" | tee -a "$LOG"; exit 1
fi
if ! curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null; then
  echo "FATAL: ollama not reachable" | tee -a "$LOG"; exit 1
fi
if ! curl -s --max-time 10 http://localhost:11434/api/tags | grep -q 'gemma4:12b'; then
  echo "FATAL: gemma4:12b not pulled" | tee -a "$LOG"; exit 1
fi

for SEED in "${SEEDS[@]}"; do
  ART="results/shift/comparison_results.shift.${SEED}.json"
  ORIG="results/shift/comparison_results.shift.${SEED}.pre-trialcorrect.json"

  if [ ! -f "$ART" ]; then
    echo "== seed $SEED: no artifact, skipping ==" | tee -a "$LOG"; continue
  fi

  if python3 -c "import json,sys; d=json.load(open('$ART')); \
      sys.exit(0 if d['arms'].get('Treatment_Eigen',{}).get('trial_correct') else 1)"; then
    echo "== seed $SEED: already has trial_correct, skipping rerun ==" | tee -a "$LOG"
  else
    # Preserve the committed run once, then drop Treatment_Eigen so the
    # driver's resume path reruns exactly that arm.
    [ -f "$ORIG" ] || { cp "$ART" "$ORIG"; echo "  preserved -> $ORIG" | tee -a "$LOG"; }
    python3 - "$ART" >>"$LOG" 2>&1 <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["arms"].pop("Treatment_Eigen", None)
json.dump(d, open(p, "w"), indent=2)
print("  checkpoint holds:", sorted(d["arms"]))
PY
    echo "== $(date '+%F %T') seed $SEED: rerunning Treatment_Eigen ==" | tee -a "$LOG"
    uv run python run_shift_experiment.py "$SEED" >>"$LOG" 2>&1
    echo "== $(date '+%F %T') seed $SEED: arm done (exit $?) ==" | tee -a "$LOG"
  fi

  echo "== $(date '+%F %T') seed $SEED: gate replay ==" | tee -a "$LOG"
  uv run python gate_replay.py "$SEED" >>"$LOG" 2>&1
done

echo "== $(date '+%F %T') all seeds done ==" | tee -a "$LOG"
echo "--- summary ---" | tee -a "$LOG"
uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json
rows = []
for f in sorted(glob.glob("results/shift/gate_replay.*.json")):
    d = json.load(open(f))
    rows.append((d["seed"], d["proxy"]["ratio"], d["live"]["ratio"],
                 (d.get("live_run_last_check") or [None, None])))
print(f'{"seed":>5} {"proxy":>7} {"live":>7} {"run":>7}')
for s, p, l, lr in sorted(rows):
    run = f"{lr[0]/lr[1]:.2f}" if lr and lr[0] else "--"
    print(f"{s:>5} {p:>7.2f} {l:>7.2f} {run:>7}")
print("\nlive tracking run (not proxy) on every seed => featurization is the bottleneck.")
PY
