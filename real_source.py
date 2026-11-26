"""Real-robot live control for the Quest teleop page (2026-11-22, NOT tested on the robot yet - see FIRST_ROBOT_TEST.md).

RealSource = SimSource (sim_source.py) with the real robot's settings. Same page, same messages, same gates:
  * DDS domain 0 on an explicitly named network interface (no default, never a silent fallback).
  * Needs --i-understand-real-robot AND a typed confirmation at startup (skipped only with --dry-run).
  * --arm-mode is required: "arm_sdk" (rt/arm_sdk, xr_teleoperate --motion: the robot's own controller keeps the legs,
    robot standing / walking mode) or "lowcmd" (rt/lowcmd, xr_teleoperate without --motion: debug mode, this program
    LOCKS ALL 29 joints at their current angles with xr_teleoperate's gains - the robot must hang in its frame).
  * Arms move only while A / X is held; engage refused if the ghost is > 25 cm from the hand; no page data for 0.3 s
    -> brake and hold; the controller starts by holding the CURRENT pose (no jump to 0).
  * Slower than the sim: 90 deg/s, 600 deg/s^2 (JointLimiter), and xr_teleoperate's own 250 Hz clip set to 90 deg/s.
  * Dex1: rt/dex1/{left,right}/cmd, q = (1 - trigger) * width (0 closed .. 5.4 open), kp 5 / kd 0.05 like
    xr_teleoperate, and at most 0.18 rad from the measured gripper per step (xr_teleoperate DELTA_GRIPPER_CMD).
    The force slider is logged only: the Dex1 command has no documented force field.
  * No scene reset / homing (next / prev episode only show a notice).
  * Cameras: teleimager on PC2 (--img-server-ip, default 192.168.123.164). Point cloud: robot_pc2/d435_depth_pub.py
    on PC2, read with --cloud-live tcp://<PC2>:55558 (server.py sets that by default for --source real).
  * --dry-run: everything runs (robot state, IK, gates, haptics, logs) but no DDS writer is created, nothing is sent.
"""
import sys
import threading
import time

import numpy as np

from sim_source import SimSource

REAL_TRACK_VEL, REAL_TRACK_ACC = np.deg2rad(90.0), np.deg2rad(600.0)
REAL_ARM_CLIP = np.deg2rad(90.0)          # G1_29_ArmController.set_arm_velocity_limit (rad/s)
REAL_GRIP_STEP = 0.18                     # rad per control step, as xr_teleoperate
CONFIRM_TEXT = "REAL ROBOT"
ARM_MODES = {"arm_sdk": True, "lowcmd": False}   # -> G1_29_ArmController(motion_mode=...)


class _Jpg:
    __slots__ = ("jpg",)
    def __init__(self, jpg):
        self.jpg = jpg


class SimpleImageClient:
    """Client for ~/image_server/image_server.py on PC2 (2026-11-22 version, confirmed running as-is: no
    CamConfigResponder, single zmq PUB on `port`, one JPEG per message = head|left_wrist|right_wrist images
    hconcat'd at `cam_w` px each, `cam_h` px tall - NOT the 3-port + cam-config-request protocol that
    teleimager.image_client.ImageClient expects. PC2's official teleimager-server (g1_nocode_xr) does speak
    that protocol, but it is not ours to manage (it stopped on its own 2026-11-26, cause unknown - possibly
    tied to g1studio/FSM state) - this script's own camera service is the one we can keep running ourselves.
    Same get_head_frame/get_left_wrist_frame/get_right_wrist_frame interface as ImageClient (an object with
    .jpg, or None)."""
    def __init__(self, host, port=5555, cam_w=640, cam_h=480):
        import zmq
        self._cam_w, self._cam_h = cam_w, cam_h
        self._lock = threading.Lock()
        self._frames = {"head": None, "left": None, "right": None}
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.connect(f"tcp://{host}:{port}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import zmq
        import cv2
        import numpy as np
        w = self._cam_w
        while True:
            try:
                data = self._sock.recv()
            except zmq.Again:
                continue
            except Exception:
                time.sleep(0.2)
                continue
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[1] < 3 * w:
                continue
            out = {}
            for name, crop in (("head", img[:, 0:w]), ("left", img[:, w:2 * w]), ("right", img[:, 2 * w:3 * w])):
                ok, buf = cv2.imencode(".jpg", crop)
                if ok:
                    out[name] = _Jpg(buf.tobytes())
            with self._lock:
                self._frames.update(out)

    def get_head_frame(self):
        with self._lock:
            return self._frames["head"]

    def get_left_wrist_frame(self):
        with self._lock:
            return self._frames["left"]

    def get_right_wrist_frame(self):
        with self._lock:
            return self._frames["right"]


def real_live_cfg(arm_mode, img_host, dry_run):
    if arm_mode not in ARM_MODES:
        raise SystemExit(f"--arm-mode must be one of {sorted(ARM_MODES)} (no default on the real robot)")
    return {"domain": 0, "label": "real", "simulation_mode": False, "motion_mode": ARM_MODES[arm_mode],
            "track_vel": REAL_TRACK_VEL, "track_acc": REAL_TRACK_ACC, "img_host": img_host, "dry_run": bool(dry_run),
            "allow_reset": False, "grip_step": REAL_GRIP_STEP, "arm_vel_limit": REAL_ARM_CLIP}


def check_start(network_interface, understood, arm_mode, dry_run, ask=input):
    """All startup gates. Returns normally only when it is OK to start; raises SystemExit otherwise."""
    if not understood:
        raise SystemExit("--source real needs --i-understand-real-robot (read FIRST_ROBOT_TEST.md first).")
    if not network_interface:
        raise SystemExit("--source real needs --network-interface (the PC's interface on the robot network, e.g. enp3s0).")
    if arm_mode not in ARM_MODES:
        raise SystemExit(f"--source real needs --arm-mode {'|'.join(sorted(ARM_MODES))} (see FIRST_ROBOT_TEST.md).")
    print("=" * 78, flush=True)
    print(f"  REAL ROBOT: DDS domain 0 on {network_interface}, arm mode {arm_mode}"
          f"{'  (rt/lowcmd: ALL joints locked, robot must HANG)' if arm_mode == 'lowcmd' else '  (rt/arm_sdk)'}", flush=True)
    print(f"  {'DRY RUN - nothing will be sent' if dry_run else 'Commands WILL be sent to the robot.'}", flush=True)
    print("=" * 78, flush=True)
    if dry_run:
        return
    try:
        typed = ask(f'Type "{CONFIRM_TEXT}" to start (anything else quits): ')
    except EOFError:
        typed = ""
    if typed.strip() != CONFIRM_TEXT:
        raise SystemExit("Not confirmed - nothing was sent.")


class RealSource(SimSource):
    def __init__(self, network_interface, arm_mode, img_host="192.168.123.164", dry_run=False, haptics=True,
                 haptic_cfg=None):
        super().__init__(network_interface, haptics=haptics, haptic_cfg=haptic_cfg,
                         live=real_live_cfg(arm_mode, img_host, dry_run), _real_ok=True)
        self.img = SimpleImageClient(img_host)      # our own PC2 server (single-port protocol), see SimpleImageClient
        print(f"[real] ready{' (DRY RUN)' if dry_run else ''}: hold A / X to let that arm follow its ghost", flush=True)
        import atexit
        atexit.register(self.shutdown)

    def shutdown(self):
        """On exit: hold the CURRENT arm pose and, in arm_sdk mode, hand the arms back to the robot's controller by
        ramping the arm_sdk weight 1 -> 0 over ~1 s (xr_teleoperate does the same, after moving the arms to 0 first;
        here they are not moved). In lowcmd mode the joints simply stay locked until the robot is put in damping."""
        import time
        if getattr(self, "_shut", False):
            return
        self._shut = True
        try:
            from teleop.robot_control.robot_arm import G1_29_JointIndex
            q = self.arm.get_current_dual_arm_q()
            self.arm.ctrl_dual_arm(q, np.zeros(14))
            self.xr = {}                                   # the control loop now only holds
            if self.live["motion_mode"]:
                for w in np.linspace(1.0, 0.0, 51):
                    self.arm.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = float(w)
                    time.sleep(0.02)
                print("[real] arm_sdk weight ramped to 0: arms handed back to the robot controller", flush=True)
        except Exception as e:
            print(f"[real] shutdown: {e}", flush=True)

    def _step(self):
        super()._step()
        for s in ("left", "right"):              # force slider: logged only (no force field in the Dex1 command)
            f = (self.xr.get(s) or {}).get("force_phys")
            if f is not None and f != getattr(self, "_flog", {}).get(s):
                self.__dict__.setdefault("_flog", {})[s] = f
                print(f"[real] {s} force slider {f:.0f} N (not sent: Dex1 command has no force field)", flush=True)


if __name__ == "__main__":
    sys.exit("Start it through server.py --source real (see FIRST_ROBOT_TEST.md).")
