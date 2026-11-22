import numpy as np


class JointLimiter:
    """Moves a commanded joint vector toward a target with per-joint velocity and acceleration limits (2026-10-27).

    Each joint brakes in time to stop at the target (v <= sqrt(2 * a_max * distance)), so a step in the target becomes
    a smooth trapezoid-like move instead of a jump. Used for teleop so recorded actions stay within human-like
    speeds (real G1_Stack_Block data: ~110-337 deg/s, ~1700-4900 deg/s^2 peak per joint).
    """

    def __init__(self, q0, dt):
        self.q = np.array(q0, dtype=float)
        self.v = np.zeros_like(self.q)
        self.dt = float(dt)

    def reset(self, q):
        self.q = np.array(q, dtype=float)
        self.v = np.zeros_like(self.q)

    def stopping_point(self, a_max):
        """Where each joint comes to rest if it brakes at a_max from now."""
        return self.q + self.v * np.abs(self.v) / (2.0 * a_max)

    def step(self, target, v_max, a_max):
        """Advance one control step toward target (rad); v_max in rad/s, a_max in rad/s^2. Returns the new command."""
        e = np.asarray(target, dtype=float) - self.q
        # fastest speed from which a_max braking in whole dt steps still stops exactly at the target
        # (discrete form of sqrt(2 * a_max * |e|); the continuous one overshoots a_max ~2x at 30 Hz)
        adt = a_max * self.dt
        v_brake = adt * (np.sqrt(2.0 * np.abs(e) / (adt * self.dt) + 0.25) - 0.5)
        v_des = np.sign(e) * np.minimum(v_max, v_brake)
        self.v += np.clip(v_des - self.v, -a_max * self.dt, a_max * self.dt)
        dq = self.v * self.dt
        # landing on the target this step: snap to it and keep only the speed needed to get there
        arrive = (np.abs(dq) >= np.abs(e)) & (np.sign(dq) == np.sign(e))
        self.q = np.where(arrive, self.q + e, self.q + dq)
        self.v = np.where(arrive, e / self.dt, self.v)
        return self.q.copy()
