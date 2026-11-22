#!/usr/bin/env bash
# Start the head D435i point-cloud publisher on PC2 in the background (log: ~/d435_pub.log).
#   bash run_d435_pub.sh            start (static waist)
#   bash run_d435_pub.sh stop       stop
# Extra args go to d435_depth_pub.py, e.g.  bash run_d435_pub.sh --waist lowstate --network-interface eth0
cd "$(dirname "$0")"
if [ "$1" = "stop" ]; then pkill -f d435_depth_pub.py && echo stopped || echo "not running"; exit 0; fi
if pgrep -f d435_depth_pub.py >/dev/null; then echo "already running (bash run_d435_pub.sh stop)"; exit 1; fi
nohup python3 -u d435_depth_pub.py "$@" >> ~/d435_pub.log 2>&1 &
sleep 3; tail -5 ~/d435_pub.log
