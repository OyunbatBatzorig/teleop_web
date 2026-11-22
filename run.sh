#!/usr/bin/env bash
# Start the Quest 3 web teleop server.
# Usage: bash ~/VLA/teleop_web/run.sh [server.py flags]
#   replay (default):  bash ~/VLA/teleop_web/run.sh [--episodes-dir DIR] [--cloud-file NPZ]
#   live sim:          bash ~/VLA/teleop_web/run.sh --source sim      (start the sim first, see PLAN_live_sim.md)
# Uses the conda env "tv" when it exists. All output also goes to logs/server_<date_time>.log.
set -e
cd "$(dirname "$0")"
PY=python
if [ -x "$HOME/miniconda3/envs/tv/bin/python" ]; then PY="$HOME/miniconda3/envs/tv/bin/python"; fi
if ! "$PY" -c "import aiohttp" 2>/dev/null; then
  echo "aiohttp is missing in this Python ($PY). Install it once with:  pip install aiohttp"
  exit 1
fi
mkdir -p logs
LOG="logs/server_$(date +%Y%m%d_%H%M%S).log"
IP=$(hostname -I | tr ' ' '\n' | grep -E '^(192\.168|10\.|172\.)' | head -1)
echo "Quest 3 browser address:  https://${IP:-<PC-IP>}:8443    (log: $PWD/$LOG)"
"$PY" -u server.py "$@" 2>&1 | tee -a "$LOG"
