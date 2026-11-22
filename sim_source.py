"""Live sim control for the Quest teleop page (2026-11-22, see PLAN_live_sim.md).

SimSource talks to unitree_sim_isaaclab on DDS domain 1 (never the real robot, domain 0):
  state  : rt/lowstate (arms 15-28, legs 0-11, waist 12-14) and rt/dex1/{left,right}/state
  arms   : the page's ghost pose (xr message "ee", L_ee / R_ee in the pelvis frame) -> G1_29_ArmIK (xr_teleoperate)
           -> JointLimiter (300 deg/s, 3000 deg/s^2) -> G1_29_ArmController (rt/lowcmd, arm motors only).
           Only while A / X (clutch) is held. Clutch engage is refused if the ghost is more than CLUTCH_MAX_M from the
           current end effector. No speed ramp. No xr message for WATCHDOG_S -> hold. Release -> brake and hold.
  gripper: trigger -> rt/dex1/{side}/cmd q = (1 - trigger) * q_max, q_max = width_mm / 90 * 5.4 rad (width slider),
           whenever that controller is live (2026-11-22: no longer needs the clutch); force is logged only (the sim
           ignores kp). Rubber hand: none.
  haptics: per-arm level 0..1 in the state message (page pulses the controller), only while that arm follows:
           joint limit: a joint within LIMIT_MARGIN of its limit -> up to 0.6 at the limit;
           contact: tracking error |q_cmd(prev) - q| above TRACK_ERR_LO .. TRACK_ERR_HI rad, or gravity-compensated
           |tau_est - tau_ff| above TAU_LO .. TAU_HI N.m -> up to 1.0. One-shot 0.5 / 80 ms pulse when the clutch is
           refused. Thresholds are first guesses (not calibrated on the sim yet).
  images : teleimager zmq 55555 head / 55556 left wrist / 55557 right wrist (JPEG, passed through as tags 0 / 1 / 2).
  episode: next / prev episode -> rt/reset_pose/cmd "1" and the arms go home slowly; start / end only times the HUD.
Run in the conda env "tv" (pinocchio, casadi, unitree_sdk2py, teleimager).
"""
import os
import sys
import threading
import time

import numpy as np

XR_TELEOP = os.path.expanduser(os.environ.get("XR_TELEOP_DIR", "~/xr_teleoperate"))   # xr_teleoperate checkout
CLUTCH_MAX_M = 0.25
WATCHDOG_S = 0.3
CTRL_HZ = 30.0
TRACK_VEL, TRACK_ACC = np.deg2rad(300.0), np.deg2rad(3000.0)
HOME_VEL, HOME_ACC = np.deg2rad(60.0), np.deg2rad(300.0)
# median first frame of the real G1_Stack_Block episodes (xr_teleoperate --start-pose real)
HOME_Q = np.array([-0.245, 0.278, 0.312, 0.059, -0.170, 0.042, -0.091,
                   -0.688, -0.663, -0.437, 1.179, 0.286, -0.295, 0.591])
# haptics (2026-11-22 v2, softer after the first test: it buzzed in free motion). All can be changed with server flags.
HAPTIC = {"gain": 1.0,                 # contact (tracking error / torque) strength; 2026-11-22 v3: was 0.5 for all
          "limit_gain": 0.5,           # joint-limit strength
          "impulse": 0.5,              # contact level jump within one step that fires a 1.0 / 80 ms pulse
          "limit": (0.08, 0.4),        # joint within 0.08 rad of a limit -> up to 0.4 at the limit
          "track": (0.15, 0.40),       # |q_cmd(prev) - q| rad: onset .. full
          "torque": (6.0, 15.0)}       # |tau_est - tau_gravity - baseline| N.m: onset .. full
TAU_BASELINE_N = 90                    # running median over the last ~3 s (always updated; v2 only updated at 0 and stuck)
LOWSTATE_WAIT_S = 60.0
EE_I_GAIN, EE_I_MAX = 0.02, 0.04       # slow position correction of the IK target (per 30 Hz step, max 4 cm), 2026-11-22 v2:
EE_I_DEADBAND, EE_I_STILL = 0.003, 0.15  # only beyond 3 mm and while the arm is nearly still (|dq| < 0.15 rad/s):
                                         # v1 (0.15 / step, always on) wound up during moves and made the arm swing
                                       # removes the IK's ~1-2 cm softness and steady sag while an arm follows
SIDES = ("left", "right")
SL = {"left": slice(0, 7), "right": slice(7, 14)}
GRIP_FULL_RAD, GRIP_FULL_MM = 5.4, 90.0


def grip_q(trigger, width_mm):
    """Dex1 command q (0 closed .. 5.4 open) from the trigger (0..1) and the width slider (mm).
    width_mm None / missing = full opening. 2026-11-22f: was `width_mm or 90`, so width 0 mm (falsy) became FULLY OPEN."""
    w = GRIP_FULL_MM if width_mm is None else min(max(float(width_mm), 0.0), GRIP_FULL_MM)
    t = min(max(float(trigger or 0.0), 0.0), 1.0)
    return (1.0 - t) * w / GRIP_FULL_MM * GRIP_FULL_RAD


def haptic_parts(q, q_cmd_prev, tau_res, lo, hi, cfg=HAPTIC):
    """(limit, track, torque) each 0..1 for one arm (7 joints)."""
    q = np.asarray(q, float)
    m_lim, lvl_lim = cfg["limit"]
    d = np.minimum(q - lo, hi - q)
    limit = lvl_lim * float(np.clip((m_lim - d) / m_lim, 0, 1).max())
    t_lo, t_hi = cfg["track"]
    err = 0.0 if q_cmd_prev is None else float(np.abs(np.asarray(q_cmd_prev) - q).max())
    track = float(np.clip((err - t_lo) / (t_hi - t_lo), 0, 1))
    k_lo, k_hi = cfg["torque"]
    torque = 0.0 if tau_res is None else float(np.clip((float(np.max(np.abs(tau_res))) - k_lo) / (k_hi - k_lo), 0, 1))
    return limit, track, torque


def haptic_level(q, q_cmd_prev, tau_res, lo, hi, following, cfg=HAPTIC):
    """Haptic level 0..1 for one arm: gain * max(joint limit, tracking error, torque), 0 when not following."""
    if not following:
        return 0.0
    lim, trk, tq = haptic_parts(q, q_cmd_prev, tau_res, lo, hi, cfg)
    return float(min(1.0, max(cfg.get("limit_gain", 0.5) * lim, cfg["gain"] * max(trk, tq))))


class ClutchGate:
    """Per-side clutch: engage only if the ghost is close to the current end effector; hold until released."""

    def __init__(self, max_dist=CLUTCH_MAX_M):
        self.max_dist = max_dist
        self.state = "off"          # off | follow | rejected
        self.dist = None

    def update(self, pressed, ghost_pos, ee_pos):
        self.dist = None if ghost_pos is None or ee_pos is None else float(np.linalg.norm(ghost_pos - ee_pos))
        if not pressed:
            self.state = "off"
        elif self.state == "off":
            self.state = "follow" if (self.dist is not None and self.dist <= self.max_dist) else "rejected"
        return self.state


def xr_import():
    """Make xr_teleoperate's packages importable (package root = ~/xr_teleoperate, IK cache in teleop/)."""
    for p in (XR_TELEOP, os.path.join(XR_TELEOP, "teleop")):
        if p not in sys.path:
            sys.path.insert(0, p)


def make_ik():
    xr_import()
    cwd = os.getcwd()
    os.chdir(os.path.join(XR_TELEOP, "teleop"))      # G1_29_ArmIK uses relative URDF / cache paths
    try:
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
        return G1_29_ArmIK()
    finally:
        os.chdir(cwd)


def torso_correction(ik, waist_q):
    """C = T_torso(waist_q) * T_torso(0)^-1 (pelvis frame). The IK model is built with the waist locked at 0, but the
    sim robot's torso is turned by its waist joints (Real task: pitch 7.4 deg), which moves the hands a few cm.
    Targets go into the IK frame with C^-1, IK end effectors come back with C (2026-11-22)."""
    import pinocchio as pin
    m = ik.robot.model; d = m.createData()
    def torso(wq):
        q = pin.neutral(m)
        for name, v in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), wq):
            if m.existJointName(name):
                q[m.joints[m.getJointId(name)].idx_q] = v
        pin.framesForwardKinematics(m, d, q)
        return d.oMf[m.getFrameId("torso_link")].homogeneous.copy()
    return torso(waist_q) @ np.linalg.inv(torso((0.0, 0.0, 0.0)))


def ee_positions(ik, q14):
    """Current L_ee / R_ee positions (pelvis frame) from the IK model."""
    import pinocchio as pin
    m, d = ik.reduced_robot.model, ik.reduced_robot.data
    pin.framesForwardKinematics(m, d, np.asarray(q14, dtype=float))
    return {s: d.oMf[m.getFrameId(f)].homogeneous.copy() for s, f in (("left", "L_ee"), ("right", "R_ee"))}


def other_teleop_running():
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "teleop_hand_and_arm.py" in cmd:
                return cmd.strip()
    return None


class _NullPublisher:
    """--dry-run: stands in for unitree_sdk2py's ChannelPublisher; creates no DDS writer and sends nothing."""
    sent = 0

    def __init__(self, *a, **k):
        pass

    def Init(self, *a, **k):
        return True

    def Write(self, *a, **k):
        _NullPublisher.sent += 1
        return True


# The sim settings. RealSource (real_source.py) is the ONLY place that may pass other values (domain 0).
SIM_LIVE = {"domain": 1, "label": "sim", "simulation_mode": True, "motion_mode": False,
            "track_vel": TRACK_VEL, "track_acc": TRACK_ACC, "img_host": "127.0.0.1", "dry_run": False,
            "allow_reset": True, "grip_step": None, "arm_vel_limit": None}


class SimSource:
    def __init__(self, network_interface=None, haptics=True, haptic_cfg=None, live=None, _real_ok=False):
        self.live = dict(SIM_LIVE, **(live or {}))
        if self.live["domain"] != 1 and not _real_ok:
            raise SystemExit("SimSource runs on DDS domain 1 only (the sim). The real robot is --source real (real_source.py).")
        L = self.live
        other = other_teleop_running()
        if other:
            raise SystemExit(f"xr_teleoperate is running ({other}); it would also publish rt/lowcmd. Stop it first.")
        xr_import()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from teleop.robot_control.robot_arm import G1_29_ArmController
        from teleop.utils.joint_limiter import JointLimiter
        from teleimager.image_client import ImageClient

        if L["dry_run"]:                                       # nothing is ever published (no DDS writers at all)
            ChannelPublisher = _NullPublisher
            import teleop.robot_control.robot_arm as _ra
            _ra.ChannelPublisher = _NullPublisher              # the arm controller's rt/lowcmd / rt/arm_sdk writer too
            print(f"[{L['label']}] DRY RUN: reading the robot only, publishing NOTHING", flush=True)
        ChannelFactoryInitialize(L["domain"], networkInterface=network_interface)   # sim: 1; 0 only via RealSource
        self._wait_for_sim(ChannelSubscriber)

        class HoldingArm(G1_29_ArmController):
            """The base class starts publishing with q_target = 0; start from the current arm pose instead."""
            def _ctrl_motor_state(self):
                with self.ctrl_lock:
                    self.q_target = self.get_current_dual_arm_q()
                super()._ctrl_motor_state()
        self._arm_cls = HoldingArm

        print("[sim] IK model ...", flush=True)
        self.ik = make_ik()
        print("[sim] waiting for rt/lowstate from the sim (domain 1) ...", flush=True)
        self.arm = HoldingArm(motion_mode=L["motion_mode"], simulation_mode=L["simulation_mode"])
        if L["arm_vel_limit"] is not None:                    # real: the controller's own 250 Hz clip as well
            self.arm.set_arm_velocity_limit(L["arm_vel_limit"])
        q0 = self.arm.get_current_dual_arm_q()
        self.limiter = {s: JointLimiter(q0[SL[s]], dt=1.0 / CTRL_HZ) for s in SIDES}
        self.hold_q = q0.copy()
        self.homing = {s: False for s in SIDES}
        self.gate = {s: ClutchGate() for s in SIDES}
        self.was_follow = {s: False for s in SIDES}
        self.ee_bias = {s: np.zeros(3) for s in SIDES}

        self.grip_state = {"left": 0.0, "right": 0.0}
        self.grip_pub, self.grip_msg = {}, {}
        for s in SIDES:
            pub = ChannelPublisher(f"rt/dex1/{s}/cmd", MotorCmds_); pub.Init(); self.grip_pub[s] = pub
            msg = MotorCmds_(); msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            msg.cmds[0].dq, msg.cmds[0].tau, msg.cmds[0].kp, msg.cmds[0].kd = 0.0, 0.0, 5.0, 0.05
            self.grip_msg[s] = msg
            sub = ChannelSubscriber(f"rt/dex1/{s}/state", MotorStates_)
            sub.Init(lambda m, side=s: self._on_grip(side, m), 1)
        self.grip_cmd = {s: 83.0 / GRIP_FULL_MM * GRIP_FULL_RAD for s in SIDES}   # open; both sides always sent
        # (the sim applies the gripper command only when it has BOTH sides)
        self.reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_); self.reset_pub.Init()
        self.haptics = haptics
        self.hcfg = dict(HAPTIC, **(haptic_cfg or {}))
        self.tau_hist = {sd: [] for sd in SIDES}
        self.hap_parts = {sd: (0.0, 0.0, 0.0) for sd in SIDES}
        self.hap = {s: 0.0 for s in SIDES}
        self.pulse = {s: None for s in SIDES}
        self.pulse_seq = 0
        self.last_cmd = None
        mdl = self.ik.reduced_robot.model
        self.q_lo, self.q_hi = np.array(mdl.lowerPositionLimit[:14]), np.array(mdl.upperPositionLimit[:14])
        self.String_ = String_

        self.img = ImageClient(host=L["img_host"], request_bgr=False)
        self.xr, self.xr_t = {}, 0.0
        self.lock = threading.Lock()
        self.running, self.t_start = False, None
        self.notice, self.notice_t = "", 0.0
        threading.Thread(target=self._loop, daemon=True).start()
        print("[sim] live control ready: hold A / X to let that arm follow its ghost", flush=True)

    def _wait_for_sim(self, ChannelSubscriber):
        """Wait (up to LOWSTATE_WAIT_S) for the sim's rt/lowstate before the arm controller starts."""
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        got = threading.Event()
        self.raw_low = None

        def on_low(m):
            self.raw_low = m
            got.set()
        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)     # kept: raw state incl. tau_est for haptics
        self._low_sub.Init(on_low, 1)
        t0 = time.time()
        while not got.wait(5.0):
            waited = time.time() - t0
            if waited >= LOWSTATE_WAIT_S:
                if self.live["label"] != "sim":
                    raise SystemExit(f"No rt/lowstate from the robot on DDS domain {self.live['domain']} after "
                                     f"{LOWSTATE_WAIT_S:.0f} s. Robot on? Network interface right? (see FIRST_ROBOT_TEST.md)")
                raise SystemExit("No rt/lowstate from the sim on DDS domain 1 after "
                                 f"{LOWSTATE_WAIT_S:.0f} s. Start the sim first (terminal 1, see PLAN_live_sim.md) and "
                                 "wait until it shows the robot, then start this server again.")
            print(f"[sim] still waiting for the sim ({waited:.0f} s) - is sim_main.py running and fully loaded?", flush=True)
        print("[sim] sim found (rt/lowstate)", flush=True)

    # ---- inputs from the page
    def on_xr(self, m):
        with self.lock:
            self.xr, self.xr_t = m, time.time()

    def command(self, cmd):
        if cmd == "toggle_episode":
            self.running = not self.running
            self.t_start = time.time() if self.running else None
        elif cmd in ("next_episode", "prev_episode") and not self.live["allow_reset"]:
            self._note("Real robot: no scene reset / homing")
        elif cmd in ("next_episode", "prev_episode"):
            self.reset_pub.Write(self.String_(data="1"))
            self.homing = {s: True for s in SIDES}
            self._note("Scene reset, arms going home")
        print(f"[cmd] {cmd} running={self.running}", flush=True)

    def _on_grip(self, side, msg):
        try:
            self.grip_state[side] = float(msg.states[0].q)
        except Exception:
            pass

    def _note(self, text):
        self.notice, self.notice_t = text, time.time()

    # ---- control loop (own thread, 30 Hz)
    def _loop(self):
        dt = 1.0 / CTRL_HZ
        while True:
            t0 = time.time()
            try:
                self._step()
            except Exception as e:                       # keep holding even if one step fails
                print(f"[sim] control step failed: {e}", flush=True)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _step(self):
        import pinocchio as pin
        with self.lock:
            xr, age = self.xr, time.time() - self.xr_t
        fresh = age < WATCHDOG_S
        q = self.arm.get_current_dual_arm_q(); dq = self.arm.get_current_dual_arm_dq()
        allq = self.arm.get_current_motor_q()
        C = torso_correction(self.ik, allq[12:15])                     # measured waist yaw / roll / pitch
        Ci = np.linalg.inv(C)
        ee_now = {sd: C @ T for sd, T in ee_positions(self.ik, q).items()}   # pelvis frame (real torso pose)
        targets, follow = {}, {}
        for s in SIDES:
            d = xr.get(s) if fresh else None
            ghost = np.array(d["ee"], dtype=float).reshape(4, 4).T if d and "ee" in d else None   # column-major
            pressed = bool(fresh and d and d.get("clutch"))
            prev = self.gate[s].state
            st = self.gate[s].update(pressed, None if ghost is None else ghost[:3, 3], ee_now[s][:3, 3])
            if st == "rejected" and prev != "rejected":
                self.pulse_seq += 1
                self.pulse[s] = {"seq": self.pulse_seq, "level": 0.5, "ms": 80}
                self._note(f"{s} arm: ghost {self.gate[s].dist * 100:.0f} cm from the hand (max {CLUTCH_MAX_M * 100:.0f})"
                           " - bring it closer")
            follow[s] = st == "follow" and ghost is not None
            if follow[s]:
                err = ghost[:3, 3] - ee_now[s][:3, 3]
                still = float(np.abs(dq[SL[s]]).max()) < EE_I_STILL
                if still and EE_I_DEADBAND < np.linalg.norm(err) < 0.05:
                    self.ee_bias[s] = np.clip(self.ee_bias[s] + EE_I_GAIN * err, -EE_I_MAX, EE_I_MAX)
                tgt = ghost.copy(); tgt[:3, 3] += self.ee_bias[s]
            else:
                self.ee_bias[s] = np.zeros(3)
            targets[s] = tgt if follow[s] else ee_now[s]
            if follow[s]:
                self.homing[s] = False
            # gripper: trigger closes from the width setting whenever this controller is live
            if d and d.get("gripper", "dex1") == "dex1":
                self.grip_cmd[s] = grip_q(d.get("trigger", 0.0), d.get("width_phys"))
        if any(follow.values()):
            seed = self.last_cmd if self.last_cmd is not None else q              # commanded, not the lagging measured q
            sol_q, _ = self.ik.solve_ik(Ci @ targets["left"], Ci @ targets["right"], seed, np.zeros(14))   # IK frame
        else:
            sol_q = q
        cmd = np.zeros(14)
        for s in SIDES:
            lim = self.limiter[s]
            if follow[s]:
                cmd[SL[s]] = lim.step(np.asarray(sol_q)[SL[s]], self.live["track_vel"], self.live["track_acc"])
            elif self.homing[s]:
                cmd[SL[s]] = lim.step(HOME_Q[SL[s]], HOME_VEL, HOME_ACC)
                self.hold_q[SL[s]] = HOME_Q[SL[s]]
            else:
                if self.was_follow[s]:
                    self.hold_q[SL[s]] = lim.stopping_point(self.live["track_acc"])
                cmd[SL[s]] = lim.step(self.hold_q[SL[s]], self.live["track_vel"], self.live["track_acc"])
            self.was_follow[s] = follow[s]
        m, dd = self.ik.reduced_robot.model, self.ik.reduced_robot.data
        tau = pin.rnea(m, dd, cmd, np.zeros(m.nv), np.zeros(m.nv))         # gravity feed-forward
        self.arm.ctrl_dual_arm(cmd, tau)
        for s in SIDES:                                                     # gripper before anything that may fail
            if self.grip_cmd[s] is not None:
                gq = float(self.grip_cmd[s])
                if self.live["grip_step"] is not None:          # real Dex1: at most this far from the measured state
                    gs = self.live["grip_step"]                  # per step (xr_teleoperate DELTA_GRIPPER_CMD 0.18 rad)
                    gq = min(max(gq, self.grip_state[s] - gs), self.grip_state[s] + gs)
                self.grip_msg[s].cmds[0].q = gq
                self.grip_pub[s].Write(self.grip_msg[s])
        if self.haptics:
            try:
                tau_res = None
                raw = self.raw_low
                if raw is not None:
                    tau_est = np.array([raw.motor_state[j].tau_est for j in range(15, 29)])
                    tau_g = pin.rnea(m, dd, np.asarray(q, float), np.zeros(m.nv), np.zeros(m.nv))
                    tau_res = tau_est - tau_g
                for sd in SIDES:
                    res = None
                    if tau_res is not None:
                        h = self.tau_hist[sd]                       # per-joint baseline = running median (no contact)
                        base = np.median(np.array(h), axis=0) if h else np.zeros(7)
                        res = tau_res[SL[sd]] - base
                    parts = haptic_parts(q[SL[sd]], None if self.last_cmd is None else self.last_cmd[SL[sd]], res,
                                         self.q_lo[SL[sd]], self.q_hi[SL[sd]], self.hcfg)
                    if tau_res is not None:
                        h.append(tau_res[SL[sd]].copy()); del h[:-TAU_BASELINE_N]
                    contact = max(parts[1], parts[2])
                    prev_contact = max(self.hap_parts[sd][1], self.hap_parts[sd][2])
                    if follow[sd] and contact - prev_contact >= self.hcfg.get("impulse", 0.5):   # sudden hit: kick
                        self.pulse_seq += 1
                        self.pulse[sd] = {"seq": self.pulse_seq, "level": 1.0, "ms": 80}
                    self.hap_parts[sd] = parts
                    self.hap[sd] = (float(min(1.0, max(self.hcfg.get("limit_gain", 0.5) * parts[0], self.hcfg["gain"] * contact)))
                                    if follow[sd] else 0.0)
                if any(follow.values()) and time.time() - getattr(self, "_hlog_t", 0.0) > 0.5:   # 2 Hz while following
                    self._hlog_t = time.time()
                    print("[hap] " + " | ".join(f"{sd[0].upper()} lim {self.hap_parts[sd][0]:.2f} trk {self.hap_parts[sd][1]:.2f}"
                                               f" tq {self.hap_parts[sd][2]:.2f} -> {self.hap[sd]:.2f}" for sd in SIDES), flush=True)
            except Exception as e:
                if time.time() - getattr(self, "_hap_err_t", 0.0) > 5.0:
                    self._hap_err_t = time.time()
                    print(f"[sim] haptics skipped: {e}", flush=True)
        self.last_cmd = cmd.copy()
        for sd in SIDES:
            key = (None if self.grip_cmd[sd] is None else round(self.grip_cmd[sd], 1), round(self.grip_state[sd], 1))
            if key != getattr(self, "_glast", {}).get(sd) and time.time() - getattr(self, "_glog_t", 0.0) > 0.1:
                self._glog_t = time.time(); self.__dict__.setdefault("_glast", {})[sd] = key
                print(f"[grip] {sd} cmd {key[0]} state {key[1]}", flush=True)
        if time.time() - getattr(self, "_log_t", 0.0) > 1.0:     # once a second: what we send (diagnosis)
            self._log_t = time.time()
            print("[sim] " + " | ".join(f"{s[0].upper()} clutch {self.gate[s].state:8s} grip_q "
                                       f"{'-' if self.grip_cmd[s] is None else f'{self.grip_cmd[s]:.2f}'}"
                                       f" (state {self.grip_state[s]:.2f})" for s in SIDES)
                  + ("" if fresh else "  [no controller data from the page]"), flush=True)

    # ---- outputs to the page
    def step(self):
        pass

    def state(self):
        allq = self.arm.get_current_motor_q()
        live = {s: {"clutch": self.gate[s].state, "dist_cm": None if self.gate[s].dist is None
                    else round(self.gate[s].dist * 100, 1)} for s in SIDES}
        return {"t": "state", "source": self.live["label"], "arm": self.arm.get_current_dual_arm_q().tolist(),
                "legs": allq[0:12].tolist(), "waist": allq[12:15].tolist(),
                "grip": [self.grip_state["left"], self.grip_state["right"]],
                "episode": f"live {self.live['label']}" + (" (dry run)" if self.live["dry_run"] else ""), "episode_idx": 0, "episode_count": 1, "frame": 0, "frames": 0,
                "running": self.running, "elapsed": (time.time() - self.t_start) if self.running else 0.0,
                "live": live, "haptic": {"left": round(self.hap["left"], 3), "right": round(self.hap["right"], 3),
                                          "pulse": self.pulse} if self.haptics else None,
                "notice": self.notice if time.time() - self.notice_t < 3.0 else ""}

    def images(self):
        out = []
        for tag, get in ((0, self.img.get_head_frame), (1, self.img.get_left_wrist_frame),
                         (2, self.img.get_right_wrist_frame)):
            try:
                f = get()
                if f is not None and f.jpg:
                    out.append(bytes([tag]) + f.jpg)
            except Exception:
                pass
        return out
