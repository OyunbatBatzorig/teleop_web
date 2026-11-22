# First real-robot test of the Quest web teleop (--source real)

Written 2026-11-22. **Never run on the robot yet.** Go step by step and stop at anything unexpected.

## What `--source real` does
- Uses DDS domain 0 on the interface you name. It never falls back to another interface or domain.
  `--source sim` always stays on domain 1.
- Arm mode, which you must choose explicitly:
  - `--arm-mode arm_sdk`: `rt/arm_sdk`, like xr_teleoperate `--motion`. The robot's own controller keeps the legs.
    Use it with the robot standing in its normal (locomotion) mode.
  - `--arm-mode lowcmd`: `rt/lowcmd`, like xr_teleoperate without `--motion`. This LOCKS ALL 29 JOINTS at their
    current angles with xr_teleoperate's gains. Use it only with the robot hanging in its frame, in debug mode.
- Startup:
  - It needs `--i-understand-real-robot`, then you type `REAL ROBOT`.
  - It starts by holding the current arm pose, so there is no jump.
- Arm motion:
  - An arm moves only while you hold A (right) / X (left).
  - Engaging is refused if the ghost is more than 25 cm from the hand.
  - No page data for 0.3 s: the arm brakes and holds.
  - Speed limit 90 deg/s and 600 deg/s², plus xr_teleoperate's own 90 deg/s clip.
- Gripper (Dex1):
  - Trigger with the width slider, q 0..5.4. The sim's 0..5.6 is not used.
  - kp 5 / kd 0.05.
  - At most 0.18 rad from the measured gripper position per step.
  - The force slider is only logged, not sent.
- No scene reset or homing: next / previous episode only show a notice.
- Quitting (Ctrl+C):
  - The arms hold their current pose.
  - In `arm_sdk` mode the arm weight ramps 1 -> 0 over 1 s, so the robot controller takes the arms back where they
    are. xr_teleoperate moves them to 0 first; this does not.

## Before
1. E-stop / remote in hand, a second person watching the robot. Nobody within arm reach.
2. Robot:
   - `lowcmd`: hanging in its frame, feet off the ground.
   - `arm_sdk`: standing in a clear area.
3. PC on the robot network (`192.168.123.x`). Find the interface name with `ip -br addr`, e.g. `enp3s0`.
4. Stop anything else that commands the arms: xr_teleoperate, other scripts. The server refuses to start if
   xr_teleoperate is running.
5. PC2:
   - teleimager running (head + wrist images).
   - Optional: `robot_pc2/d435_depth_pub.py` for the point cloud. See `robot_pc2/README_PC2.md`.

## Step 1: dry run (nothing is sent)
```bash
bash ~/VLA/teleop_web/run.sh --source real --dry-run --i-understand-real-robot --arm-mode arm_sdk --network-interface enp3s0
```
- The log shows `DRY RUN: reading the robot only, publishing NOTHING` and `[real] ready (DRY RUN)`.
- On the Quest, the digital robot shows the real arm and gripper angles, and the cameras and point cloud appear.
- Hold A / X. The HUD shows follow / refused (> 25 cm), and the log shows `[sim] ... clutch follow`. The real arm
  must NOT move. Move the controllers around and watch `[hap]` / `[grip]`: they show what would be sent.

## Step 2: one arm, slowly
1. Start without `--dry-run`. Type `REAL ROBOT`.
2. Right arm only:
   1. Put the ghost on the gripper (HUD distance < 5 cm).
   2. Hold A.
   3. Move a few cm, slowly. Release A: the arm must stop and hold.
3. Gripper: press the trigger slowly. It must close; release and it opens. Test the width slider at 0 / 50 / 100 %.
4. Watchdog: while holding A, take the headset off or close the page. The arm must brake and hold within ~0.3 s.
5. Then the left arm (X), then both.
6. Ctrl+C: the arms stay where they are. In `arm_sdk` mode the robot controller takes them over smoothly.

## Things not done yet / known
- The haptic contact thresholds are tuned on the sim only.
- Real `tau_est` will be noisier, so the contact buzz may need `--haptic-torque` raised.
- The Dex1 force field is not sent (the message has no documented force).
- The point cloud camera pose is the nominal URDF mount: expect ~1-2 cm error.
