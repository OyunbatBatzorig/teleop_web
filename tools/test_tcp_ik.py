"""Offline test of the fingertip TCP IK (no sim, no robot). Run in env tv from ~/VLA/teleop_web:
    ~/miniconda3/envs/tv/bin/python tools/test_tcp_ik.py
Moves the ghost a few cm around the home pose, holds it still for 1 s (30 IK steps), and compares the FINGERTIP error of
  old: xr_teleoperate's IK at L_ee (wrist + 5 cm), regularisation 0.02
  new: IK at the fingertip TCP (sim_source.rebuild_ik_at_tcp)."""
import os, sys, copy, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("TELEOP_TCP_M", "0.1348")
import sim_source as S
import pinocchio as pin

def tip_pose(ik, q, frames):
    m, d = ik.reduced_robot.model, ik.reduced_robot.data
    pin.framesForwardKinematics(m, d, q)
    return [d.oMf[m.getFrameId(f)].homogeneous.copy() for f in frames]

new = S.make_ik()                                   # TCP IK
S_TCP = S.TCP_M
S.TCP_M = 0.0; old = S.make_ik(); S.TCP_M = S_TCP    # plain xr_teleoperate IK (wrist point)
T = S.T_TCP
q0 = S.HOME_Q.copy()
rng = np.random.default_rng(0)
res = {"old": [], "new": []}
times = {"old": [], "new": []}
for k in range(20):
    dq = rng.normal(0, 0.15, 14)
    mdl = new.reduced_robot.model
    qg = np.clip(q0 + dq, mdl.lowerPositionLimit[:14] + 0.05, mdl.upperPositionLimit[:14] - 0.05)  # reachable
    goal = tip_pose(new, qg, ("L_tcp", "R_tcp"))           # a reachable fingertip pose = the ghost tip
    # old IK gets the ghost L_ee pose (tip pose with the TCP offset removed)
    # hold the ghost still for 1 s (30 control steps, seed = previous command), like the server does
    sol_old, sol_new = q0.copy(), q0.copy()
    for i in range(30):
        t0 = time.perf_counter()
        sol_old = np.array(old.solve_ik(goal[0] @ np.linalg.inv(T), goal[1] @ np.linalg.inv(T), sol_old, np.zeros(14))[0])
        t1 = time.perf_counter()
        sol_new = np.array(new.solve_ik(goal[0], goal[1], sol_new, np.zeros(14))[0])
        times["old"].append(t1 - t0); times["new"].append(time.perf_counter() - t1)
    for name, sol in (("old", sol_old), ("new", sol_new)):
        tips = tip_pose(new, np.asarray(sol, float), ("L_tcp", "R_tcp"))
        res[name].append(max(np.linalg.norm(tips[i][:3, 3] - goal[i][:3, 3]) * 1000 for i in (0, 1)))
    old.smooth_filter._data_queue.clear(); new.smooth_filter._data_queue.clear()
for name in ("old", "new"):
    a = np.array(res[name]); tt = np.array(times[name]) * 1000
    print(f"{name}: fingertip error mm  median {np.median(a):.2f}  max {a.max():.2f} | IK time ms median {np.median(tt):.1f} max {tt.max():.1f}")
assert np.median(res["new"]) < 1.0, "the TCP IK should put the fingertip within 1 mm"
assert np.median(res["new"]) < np.median(res["old"])
assert np.percentile(np.array(times["new"]) * 1000, 95) < 25, "IK must stay fast enough for the 30 Hz loop"
print("PASS: the fingertip follows the ghost tip more precisely with the TCP IK")
