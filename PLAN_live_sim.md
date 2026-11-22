# Live sim control (implemented 2026-11-22)

Status: implemented 2026-11-22 (sim_source.py, server.py --source sim). Parked 2026-11-21, plan unchanged.
Run: T1 sim (conda activate unitree_sim_env; cd ~/VLA/unitree_sim_isaaclab; G1_REAL_SELF_COLLISION=0 python sim_main.py --device cpu --enable_cameras --task Isaac-Chess-Pieces-G129-Dex1-Joint-Factory (old id Isaac-Stack-RgyBlock-Real-G129-Dex1-Joint still works) --enable_dex1_dds --robot_type g129), T2 bash ~/VLA/teleop_web/run.sh --source sim. Offline test: ~/miniconda3/envs/tv/bin/python tools/test_live_offline.py

## Data flow (existing, reuse)
- DDS domain 1 (sim and xr_teleoperate --sim); the real robot is domain 0.
- Arms: rt/lowcmd (hg LowCmd) -> sim, rt/lowstate (hg LowState) <- sim; motor order legs 0-11, waist 12-14, arms 15-28.
- Dex1: rt/dex1/{left,right}/cmd|state (MotorCmds_/MotorStates_); the sim uses only cmds[0].q.
- Reset: rt/reset_pose/cmd (String "1"). Images: teleimager zmq 55555 head, 55556 / 55557 wrists, 55558 head depth.

## Server (server.py --source sim), conda env "tv", cwd ~/xr_teleoperate/teleop for the IK cache
- IK: G1_29_ArmIK.solve_ik(left 4x4, right 4x4, q, dq). The page's xr message "ee" (column-major, pelvis frame, Unitree
  L_ee/R_ee convention) is the IK target directly (transpose to row-major).
- Arms: G1_29_ArmController(simulation_mode=True); JointLimiter per arm, track 300 deg/s / 3000 deg/s^2; release = brake and
  hold; start = home_q (median of the real G1_Stack_Block first frames), 60 deg/s.
- Gripper: own publisher on rt/dex1/*/cmd; q = trigger closes from the width setting: q_max = width_mm / 90 * 5.4 rad.
  Force: no effect in the sim (kp ignored); log only.

## Decisions (user, 2026-11-21)
- NO speed ramp on clutch (no 0.5 s ramp-in); the limiter alone smooths.
- Clutch is rejected only if the ghost is more than 25 cm from the current end effector (toast on the page).
- Watchdog: no xr message for 0.3 s -> hold the arms where they are.
- SimSource only on DDS domain 1; a real-robot source needs its own explicit flag.
- Refuse to start if another rt/lowcmd publisher (xr_teleoperate --sim) is running.
