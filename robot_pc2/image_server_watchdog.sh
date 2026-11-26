#!/bin/bash
# Watchdog for ~/image_server/image_server.py on PC2 (2026-11-26).
# Restarts it if the process dies (e.g. the hconcat crash seen when a camera read times out)
# or if it is stuck spamming "frame read is error" without a single success in a while.
# Run this INSTEAD of starting image_server.py by hand:
#   nohup bash ~/image_server_watchdog.sh > ~/image_server_watchdog.log 2>&1 &
set -u
cd ~/image_server || exit 1
LOG=~/image_server.log
STUCK_ERROR_LINES=20     # consecutive ERROR lines at the tail = stuck, not just one bad frame

start() {
    echo "$(date '+%F %T') watchdog: starting image_server.py" >> "$LOG"
    nohup python image_server.py >> "$LOG" 2>&1 &
    SERVER_PID=$!
}

start
while true; do
    sleep 10
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "$(date '+%F %T') watchdog: process died, restarting" >> "$LOG"
        start
        continue
    fi
    tail_lines=$(tail -n "$STUCK_ERROR_LINES" "$LOG" 2>/dev/null)
    error_count=$(echo "$tail_lines" | grep -c "ERROR")
    if [ "$error_count" -ge "$STUCK_ERROR_LINES" ]; then
        echo "$(date '+%F %T') watchdog: stuck in an error loop ($error_count/$STUCK_ERROR_LINES), restarting" >> "$LOG"
        kill "$SERVER_PID" 2>/dev/null
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null
        start
    fi
done
