#!/usr/bin/env bash
# Smoke test: start a smoke run paused at step 200, wait, resume, wait. Prints every result line.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
start=$("$PY" launch/launch.py start smoke 0 --stop-at 200 | tail -1) || { echo "START_FAILED $start"; exit 1; }
echo "STARTED $start"
rid=$(echo "$start" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
cid=$(echo "$start" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['function_call_id'])")
"$PY" launch/wait_call.py "$cid" 3600 | tee /tmp/smoke_c0.txt
grep -q '"state": "paused"' /tmp/smoke_c0.txt || { echo "CHUNK0_NOT_PAUSED"; exit 1; }
res=$("$PY" launch/launch.py resume "$rid" | tail -1) || { echo "RESUME_FAILED $res"; exit 1; }
echo "RESUMED $res"
cid2=$(echo "$res" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['function_call_id'])")
"$PY" launch/wait_call.py "$cid2" 3600
echo "SMOKE_DONE $rid"
