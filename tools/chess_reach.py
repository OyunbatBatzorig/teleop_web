"""Chess board reachability for the G1 + Dex1 in the Chess (Real) task (2026-11-22, offline, no sim).

Same arm model as the live teleop (xr_teleoperate G1_29_ArmIK reduced model, waist locked) with the Chess task's waist
(yaw -0.0175, roll 0, pitch 0.129 rad) through sim_source.torso_correction. Per arm, a damped least squares IK with
the joints kept >= MARGIN_Q from their limits; a target counts when the fingertip pad (GRASP_OFF in front of the IK
end-effector frame) is within 5 mm and the gripper axis within 5 deg of the wanted direction, for at least one of the
grasp directions (straight down, or tilted down 70/55/40 deg, turned 0 / +-25 / +-50 deg). A crude torso check rejects
solutions with the elbow / wrist inside the torso.

Square "ok with margin": reachable at grasp height, +5 cm and +10 cm (approach), and with the grasp point moved
+-4 cm in x and in y. Writes chess_reach.json and chess_reach.png (per arm) next to --out.
"""
import argparse, json, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

WAIST = np.array([-0.0175, 0.0, 0.129])
GRASP_OFF = 0.11            # m, IK ee frame -> fingertip pad centre along the gripper axis (Dex1 base +0.0415, pads ~0.12)
MARGIN_Q = 0.15
POS_TOL, ROT_TOL = 0.005, np.deg2rad(5.0)
SL = {"left": slice(0, 7), "right": slice(7, 14)}
FRAME = {"left": "L_ee", "right": "R_ee"}


def directions():
    out = [(90.0, 0.0)]
    for tilt in (70.0, 55.0, 40.0):
        for head in (0.0, 25.0, -25.0, 50.0, -50.0):
            out.append((tilt, head))
    return out


def ee_rot(tilt_deg, head_deg):
    t, h = np.deg2rad(tilt_deg), np.deg2rad(head_deg)
    a = np.array([np.cos(t) * np.cos(h), np.cos(t) * np.sin(h), -np.sin(t)])       # gripper axis (ee x)
    y = np.array([-np.sin(h), np.cos(h), 0.0])                                      # jaw axis, horizontal
    y -= a * (a @ y); y /= np.linalg.norm(y)
    return np.column_stack([a, y, np.cross(a, y)])


class Reach:
    def __init__(self):
        import pinocchio as pin
        from sim_source import make_ik, torso_correction, HOME_Q
        self.pin = pin
        self.ik = make_ik()
        self.m, self.d = self.ik.reduced_robot.model, self.ik.reduced_robot.data
        self.C = torso_correction(self.ik, WAIST)                  # IK frame -> pelvis frame
        self.Ci = np.linalg.inv(self.C)
        lo, hi = np.array(self.m.lowerPositionLimit[:14]), np.array(self.m.upperPositionLimit[:14])
        self.lo, self.hi = lo + MARGIN_Q, hi - MARGIN_Q
        self.fid = {s: self.m.getFrameId(FRAME[s]) for s in SL}
        self.home = np.clip(np.asarray(HOME_Q, float), self.lo, self.hi)
        self.dirs = [(t, h, ee_rot(t, h)) for t, h in directions()]
        names = [self.m.frames[i].name for i in range(len(self.m.frames))]
        self.elbow = {s: self.m.getFrameId(f"{s}_elbow_link") if f"{s}_elbow_link" in names else None for s in SL}
        self.wrist = {s: self.m.getFrameId(f"{s}_wrist_roll_link") if f"{s}_wrist_roll_link" in names else None for s in SL}

    def solve(self, side, T_pelvis, seed):
        pin = self.pin
        T = self.Ci @ T_pelvis
        tgt = pin.SE3(T[:3, :3], T[:3, 3])
        q = seed.copy(); sl = SL[side]
        for _ in range(120):
            pin.framesForwardKinematics(self.m, self.d, q)
            cur = self.d.oMf[self.fid[side]]
            err = pin.log6(cur.actInv(tgt)).vector
            if np.linalg.norm(err[:3]) < POS_TOL * 0.5 and np.linalg.norm(err[3:]) < ROT_TOL * 0.5:
                break
            J = pin.computeFrameJacobian(self.m, self.d, q, self.fid[side], pin.LOCAL)[:, sl]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            q[sl] = np.clip(q[sl] + np.clip(dq, -0.3, 0.3), self.lo[sl], self.hi[sl])
        pin.framesForwardKinematics(self.m, self.d, q)
        cur = self.d.oMf[self.fid[side]]
        e = pin.log6(cur.actInv(tgt)).vector
        ok = np.linalg.norm(cur.translation - tgt.translation) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL
        if ok:                                                        # crude torso collision: elbow / wrist in the torso
            for f in (self.elbow[side], self.wrist[side]):
                if f is None:
                    continue
                p = (self.C @ np.r_[self.d.oMf[f].translation, 1.0])[:3]
                if p[0] < 0.10 and abs(p[1]) < 0.11 and -0.05 < p[2] < 0.45:
                    ok = False
        return ok, q

    def reachable(self, side, g, seed=None):
        """Fingertip pad at g (pelvis frame): (ok, direction, q)."""
        for t, h, R in self.dirs:
            if side == "right":
                R = np.diag([1, -1, 1]) @ R @ np.diag([1, -1, 1])       # mirror the heading for the right arm
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = g - R[:, 0] * GRASP_OFF
            for s0 in ([seed] if seed is not None else []) + [self.home]:
                ok, q = self.solve(side, T, s0)
                if ok:
                    return True, (t, h), q
        return False, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=0.0565, help="grasp height in the pelvis frame (board top 0.0265 + 3 cm)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "chessreach"))
    ap.add_argument("--step", type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = Reach()
    xs = np.round(np.arange(0.14, 0.70 + 1e-9, args.step), 3)
    ys = np.round(np.arange(-0.50, 0.50 + 1e-9, args.step), 3)
    zs = [args.z, args.z + 0.05, args.z + 0.10]
    grid = {}
    t0 = time.time()
    for side in ("left", "right"):
        G = np.zeros((len(zs), len(xs), len(ys)), bool)
        for k, z in enumerate(zs):
            for i, x in enumerate(xs):
                seed = None
                for j, y in enumerate(ys):
                    ok, _, q = R.reachable(side, np.array([x, y, z]), seed)
                    G[k, i, j] = ok
                    seed = q if ok else None
        grid[side] = G
        print(f"[reach] {side}: {G.sum(axis=(1, 2)).tolist()} reachable cells per height ({time.time() - t0:.0f} s)", flush=True)
    np.savez(os.path.join(args.out, "reach_grid.npz"), xs=xs, ys=ys, zs=zs, left=grid["left"], right=grid["right"])
    print("saved", os.path.join(args.out, "reach_grid.npz"))


if __name__ == "__main__":
    main()
