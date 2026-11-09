#!/usr/bin/env bash
# 2 Nov 2026 "go": redeploy, resume full_seed0 to 62,000 steps, spawn the reference eval, watch step 2,100.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
( cd modal_app && modal deploy train_app.py 2>&1 | grep -E "deployed|Error|error" ) || { echo "DEPLOY_FAILED"; exit 1; }
echo "DEPLOYED $(date -u +%H:%M:%SZ)"
res=$("$PY" launch/launch.py resume full_seed0 | tail -1) || { echo "RESUME_FAILED $res"; exit 1; }
echo "RESUMED $res"
cid=$(echo "$res" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['function_call_id'])")
ref=$("$PY" - <<'PYEOF'
import json, modal, time
call = modal.Function.from_name("orb-retrain", "eval_reference").spawn("full_seed0")
rec = {"function_call_id": call.object_id, "spawned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
open("runs_local/full_seed0/launches.jsonl", "a").write(json.dumps({"ref_eval": rec}) + "\n")
print(json.dumps(rec))
PYEOF
) || { echo "REF_SPAWN_FAILED"; }
echo "REF_SPAWNED $ref"
rid=$(echo "$ref" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['function_call_id'])")
# watch for the step-2100 log line (poll the Volume every 20 s, up to 20 min)
for i in $(seq 1 60); do
  modal volume get --force orb-retrain runs/full_seed0/train_log.csv runs_local/full_seed0/train_log.csv >/dev/null 2>&1
  line=$(grep -E "^2100," runs_local/full_seed0/train_log.csv 2>/dev/null | head -1)
  if [ -n "$line" ]; then echo "STEP2100 $line"; break; fi
  if modal volume get --force orb-retrain runs/full_seed0/status.json runs_local/full_seed0/status.json >/dev/null 2>&1; then
    st=$(grep -o '"state": "[a-z_]*"' runs_local/full_seed0/status.json | head -1)
    case "$st" in *stopped*|*done*) echo "STATE_CHANGED $st"; break;; esac  # "paused" is the previous chunk's state
  fi
  sleep 20
done
[ -z "${line:-}" ] && echo "STEP2100_NOT_SEEN_IN_20MIN"
"$PY" launch/wait_call.py "$rid" 2400 | sed 's/^CALL_RESULT/REF_RESULT/; s/^CALL_ERROR/REF_ERROR/'
echo "GO_SCRIPT_END train_call=$cid"
