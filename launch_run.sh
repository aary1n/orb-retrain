#!/usr/bin/env bash
# Start a run in the deployed Modal app `orb-retrain` (it only spawns; the ledger inside the app holds the cap).
#   ./launch_run.sh <condition> <seed> [--stop-at N]
# Resume a paused run:  python launch/launch.py resume <run_id> [--stop-at N]
# Needs the Modal client logged in (on Aaryan's laptop: python -m modal ...). Only condition "full" is built.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
"$PY" launch/launch.py start "$@"
