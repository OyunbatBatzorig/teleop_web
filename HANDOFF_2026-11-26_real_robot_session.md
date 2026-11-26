# HANDOFF 2026-11-26 (evening): first real-robot session - what was done, what is next

Continues from `HANDOFF_2026-11-26.md` (that file is the plan written *before* today's real-robot
session; this file is what actually happened). Read both, then `FIRST_ROBOT_TEST.md` and `README.md`.

## Working rules (unchanged, keep them)
Same as `HANDOFF_2026-11-26.md`: reply to the user in Mongolian, explain new terms, the user runs the
sim/robot themselves (only offline/non-actuating tests and code edits from this side), back up before
every edit (`file.bak_<YYYYMMDD><letter>`), one step at a time, read logs yourself, never commit/push
unless asked, never skip the real-robot safety gates.

## Topology (learned today - write this down, it is not obvious)
- **PC1** (this machine, RTX 3090): runs `teleop_web`'s `server.py` + sim/IK, the Quest connects to it.
- **PC2** (Jetson-class computer *inside* the G1): runs the camera/depth services, reachable at
  `192.168.123.164` (`eth0` there; PC1's robot-network interface is `enp5s0`). SSH as `unitree`.
- PC2 has **three** cameras, not the one "head D435i" the code originally assumed:
  - Head: a **plain 2D USB "tripod webcam"** (no depth), physical USB port `...3.2.3:1.0`. This is what
    `~/image_server/image_server.py`'s stock config calls `head_camera`.
  - Left/right wrist: **2x Intel RealSense D405** (depth-capable, but used here only for their color
    stream), ports `...3.1:1.0` and `...3.3:1.0`.
  - A **separate Intel RealSense D435i** (serial `243322072402`) exists but was **unplugged**; found and
    physically reconnected into the USB hub at `usb2/2-2/2-2.1` (the hub `cam_config_server.yaml` already
    had a now-correct, previously-commented-out path for it). This is the one that can give a *head*
    point cloud: nominal URDF mount, `d435_depth_pub.py` already assumes this exact position.
  - `rs-enumerate-devices` is the tool to list RealSense serials; USB device numbers (`/dev/videoN`)
    **renumber** on every replug/power event - never hardcode them, always use `--serial` with the stable
    RealSense serial, or the `by-path` symlinks (which encode the physical USB port, not the number).

## Camera pipeline: two mismatched protocols (this was the main time sink)
`teleimager.image_client.ImageClient` (what `sim_source.py` originally used) expects: a config query on
port 60000 (`CamConfigResponder`), then one zmq PUB per camera (3 ports, default 55555/56/57 from
`cam_config_server.yaml`). The actual script PC2 was running by hand, `~/image_server/image_server.py`
(Unitree's older/simpler reference `image_server.py`, not the teleimager package), does **not** implement
that: it is a **single** zmq PUB (default port 5555), **one combined JPEG per message** = head|left_wrist|
right_wrist images `cv2.hconcat`'d side by side at 640 px each (480 px tall). No config responder, so the
real client's query to port 60000 always timed out and silently fell back to a **stale, sim-era cached**
`~/xr_teleoperate/teleop/teleimager/cam_config_client.yaml` (`type: isaacsim`, wrong assumptions) -
pointing at zmq ports that never matched what was actually running. Result: dry run looked perfect
(robot state, clutch gate, haptics all correct) but **no camera image and no point cloud ever arrived**,
silently (the bug in `sim_source.py`'s `images()` swallows all exceptions).

Also found (and still running, found by accident): PC2 has an **official Unitree teleimager-server**
(`/home/unitree/.cache/g1_nocode_xr/teleimager-server`, launched by their own `g1studio` app/service,
*not* by us) that **does** implement the full protocol correctly (60000 + 55555-57 confirmed via
`ss -tlnp`). It is not under our control, though: it stopped on its own partway through the session for
an unknown reason (possibly tied to `g1studio`/FSM state), so we went back to our own, PC2-managed setup
rather than depend on it.

**Fix applied**: `real_source.py` now has its own `SimpleImageClient` class (new, ~50 lines) that speaks
the *actual* single-port combined-JPEG protocol `~/image_server/image_server.py` uses, splits the combined
frame into 3 per-camera JPEGs, and exposes the same `get_head_frame()` / `get_left_wrist_frame()` /
`get_right_wrist_frame()` interface `ImageClient` has, so nothing else in `sim_source.py` needed to
change. `RealSource.__init__` now does `self.img = SimpleImageClient(img_host)` right after
`super().__init__(...)` (which still creates the stock `ImageClient`, now unused for `--source real`;
`--source sim` is untouched and still uses the stock `ImageClient`). Backup: `real_source.py.bak_20261126a`.

**`image_server.py` keeps crashing** (a camera read timeout -> `cv2.hconcat` shape-mismatch exception ->
unhandled -> process exits). Wrote `robot_pc2/image_server_watchdog.sh` (new file, copied to PC2's home
dir as `~/image_server_watchdog.sh`): restarts the process if it died, and also force-restarts it if the
log's last 20 lines are all `ERROR` (seen once: it was alive but stuck spamming read errors without
producing a frame). Run it **instead of** starting `image_server.py` by hand:
```bash
conda activate tv
nohup bash ~/image_server_watchdog.sh > ~/image_server_watchdog_wrapper.log 2>&1 &
```
**Both PC2 services (this watchdog and `d435_depth_pub.py`) die whenever the robot's power/FSM state
changes** (seen after a Debug-Mode -> Normal-Mode transition) - after any such transition, check
`ps aux | grep -i "image_server\|d435_depth_pub"` on PC2 and restart both if needed:
```bash
conda activate tv
nohup bash ~/image_server_watchdog.sh > ~/image_server_watchdog_wrapper.log 2>&1 &
nohup python3 d435_depth_pub.py --serial 243322072402 --waist lowstate --network-interface eth0 \
    > ~/d435_depth_pub.log 2>&1 &
```
If `d435_depth_pub.py` fails with `Device or resource busy` right after a kill/crash, it is usually a
stuck librealsense/USB lock from the previous run - **unplug and replug the D435i's USB cable**, wait a
few seconds, replug, then retry (confirm with `rs-enumerate-devices | grep -A2 243322072402` first).

## Point cloud: `--waist static` vs `--waist lowstate`
`d435_depth_pub.py --serial 243322072402` needed `unitree_sdk2py` in PC2's `tv` conda env for
`--waist lowstate` (reads the real waist yaw/roll/pitch from `rt/lowstate`, read-only, to correctly
place the camera-in-pelvis transform). It was missing; installed via the same editable checkout the
`g1` env already used: `pip install -e ~/Teleoperacao/unitree_sdk2_python` (while `tv` was active).
Without `--waist lowstate` (i.e. the default `--waist static`, waist assumed at 0), the 3D robot model
and the point cloud visibly disagreed (seen clearly: gripper touching a box in reality, the rendered
robot's gripper clearly short of it) wherever the real waist was not exactly level - switching to
`--waist lowstate --network-interface eth0` fixed the gross misalignment. A smaller, edge-level mismatch
remained near the gripper's metal fingers in the point cloud (likely stereo-depth noise on a reflective
edge, not a calibration bug - not pursued further, not urgent).

## `web/app.js` changes today
- Wrist camera panel 25 % smaller (`0.32x0.24` -> `0.24x0.18` m), same top edge as before (offset
  recomputed: `0.08 + (0.24-0.18)` bottom offset, `+ 0.09` half-height). Backup: `app.js.bak_20261126a`.
- TCP dots: robot's dot is now gray (`COLOR_ROBOT`, matches the robot body), ghost's dot is yellow while
  idle and turns green (`COLOR_FOLLOW`) together with the ghost body while the arm is following (was:
  robot red, ghost magenta, both fixed colors).

## First real-robot arm motion test: robot FSM gotcha
`--arm-mode arm_sdk` publishes to `rt/arm_sdk` (`kTopicLowCommand_Motion`), which the robot's own
controller only honours in its **Normal Mode -> Main Operation Control** FSM state (robot standing,
balancing itself). The robot had been left in **Debug Mode** (just powered on and left there) - in that
state the gripper moved (separate DDS channel, `rt/dex1/*`, unaffected by FSM) but the **arm never moved
at all**, which looked like a bug but was the robot simply not listening to `rt/arm_sdk` yet. Fix: on the
Unitree remote, `L2+B` (Debug Mode -> Normal Mode/Damping) -> `L2+UP` (Damping -> Get Ready) -> `R1+X`
(Get Ready -> Main Operation Control). After that the arm did move (slowly - by design, see below).

## Open issue: clutch flaps follow/off even while A/X is held continuously
Once in Main Operation Control, the arm moved but felt very hard to control and the gripper's ghost
didn't track well. The server log shows `R clutch follow` / `R clutch off` toggling rapidly in isolated
single-sample blips while the user reports holding A continuously. In code (`sim_source.py`, `ClutchGate.
update`, ~line 104): `state` only goes to `"off"` when `pressed` is false, and
`pressed = bool(fresh and d and d.get("clutch"))` with `fresh = age < WATCHDOG_S` (0.3 s). So every blip
means either the page's A-state briefly read as unpressed, or - more likely, since it happens in short
isolated blips inside long "follow" runs rather than long released-look stretches - the websocket data
briefly arrived >0.3 s late. Every "off" blip resets the arm's target to "hold current pose"
(`targets[s] = ee_now[s]`), which is very plausibly *why* the motion felt jerky/unresponsive. **Confirmed
not caused by our point-cloud/camera traffic**: the user retested with `d435_depth_pub.py` killed and the
flapping was unchanged.

**Leading theory (not yet proven, no profiling done)**: `teleop_web`'s control loop (IK solve, 30 Hz) and
the websocket I/O that updates `xr_t`/`fresh` run as **threads inside one Python process**, sharing the
GIL; a CPU-heavy tick (IK solve, or - on `--source real`, less likely since cameras are on PC2 - any
synchronous image/cloud processing on PC1) could stall websocket processing long enough to cross the
0.3 s watchdog, even though new Quest data was actually waiting in the OS socket buffer. Found a strong
supporting data point: `~/Teleoperacao/xr_teleoperate/teleop/teleop_peter.py` (Unitree's own reference
teleop script, which the user says tracks fast and well on this same robot) uses **real `multiprocessing`**
(`shared_memory`, `Value`, `Array`, `Lock` - separate OS processes, no GIL contention) instead of threads.
This is the single biggest architectural difference found between the two and the most likely explanation.

**Not fixed yet** - the user wants to test `teleop_peter.py` directly first (to confirm it really does
track well on the real robot, as a point of comparison / to get a first real-arm-motion data point this
way while `teleop_web`'s threading issue is still open), and revisit `teleop_web`'s GIL/threading
architecture in a later session (a real refactor - moving the control loop to a separate process - not a
quick fix; `teleop_web` is a side tool for data collection/demos, not the thesis subject itself, so this
is lower priority than the thesis's actual orchestration work).

## Next steps (suggested order)
1. Run `teleop_peter.py` directly (see its `--help` / argparse for exact flags; it uses `TeleVuer`/`vuer`
   for the VR side, not our `app.js`/WebXR page - different headset connection, same `G1_29_ArmController`
   / `G1_29_ArmIK` / DDS underneath). Confirm whether it really tracks well on this robot with no flapping.
2. If (1) confirms the multiprocessing theory: plan a `teleop_web` refactor (control loop in its own
   process, shared memory or a pipe to the asyncio/websocket side) - a real piece of work, not a quick
   patch. Until then, real-robot arm teleop through `teleop_web` should be expected to feel jerky.
3. Once camera/point-cloud is confirmed stable again after any FSM-mode change, continue
   `FIRST_ROBOT_TEST.md` Step 2 (left arm, then both arms, then the watchdog/release test) when ready.
4. Minor/cosmetic, not urgent: the small point-cloud edge artifact near the gripper fingers; whether to
   raise the real-robot speed limit (currently 90 deg/s, intentionally conservative for a first test) once
   confidence is higher - ask the user first, it is a safety parameter.

## Files touched today (backups made before every edit, per the house rule)
- `real_source.py` (`.bak_20261126a`): added `SimpleImageClient`, wired it into `RealSource.__init__`.
- `web/app.js` (`.bak_20261126a`): wrist panel size/position, TCP dot colors.
- New: `robot_pc2/image_server_watchdog.sh` (local here, copied to PC2 as `~/image_server_watchdog.sh`).
- PC2 (not version-controlled here): `~/image_server/image_server.py` restarted multiple times by hand /
  via the watchdog; `pip install -e ~/Teleoperacao/unitree_sdk2_python` into the `tv` env (adds
  `unitree_sdk2py`, `cyclonedds`, `rich-click`, `click`, `typing-extensions`).

## When you finish
Same as before: write a short Mongolian summary for the user of what changed and what was tested, so they
can pass it back to the project thread.
