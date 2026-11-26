"""Diagnostics for the ~3.4 mm fingertip error of the TCP IK (2026-11-25). Read-only: changes no project file.
Run in env tv from ~/VLA/teleop_web:
    ~/miniconda3/envs/tv/bin/python tools/test_tcp_ik_diag.py
Same 20 reachable goals as test_tcp_ik.py. Each variant changes ONE thing of the IK cost/loop, so the variant that
removes the error shows the cause:
  A now        : 50*trans + 1*rot + 0.005*reg + 0.1*smooth, xr filter, 30 steps (= sim_source now)
  B 150 steps  : like A, but the ghost is held 5 s     -> if better: slow convergence
  C no reg     : reg 0                                 -> if better: regularisation pulls the arm away
  D no filter  : no 4-sample moving filter             -> if better: the filter
  E raw solve  : one solve from the home pose, reg 0, smooth 0, no filter (pure IK accuracy)
  F trans 1000 : translation weight 1000 instead of 50 (fingertip position first)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("TELEOP_TCP_M", "0.1348")
import sim_source as S
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin

ik = S.make_ik()                       # builds the model with the L_tcp / R_tcp frames
m = ik.reduced_robot.model
d = m.createData()
FL, FR = m.getFrameId("L_tcp"), m.getFrameId("R_tcp")
EL, ER = m.getFrameId("L_ee"), m.getFrameId("R_ee")


def fk(q):
    pin.framesForwardKinematics(m, d, np.asarray(q, float))
    return d.oMf[FL].homogeneous.copy(), d.oMf[FR].homogeneous.copy()


# 0) frame sanity: the TCP frame must be exactly L_ee shifted by TCP_M along the wrist x axis
pin.framesForwardKinematics(m, d, S.HOME_Q)
gap = np.linalg.norm((d.oMf[EL].homogeneous @ S.T_TCP)[:3, 3] - d.oMf[FL].translation) * 1000
print(f"0) frame check: |L_ee*T_TCP - L_tcp| = {gap:.3f} mm (must be 0)")


def build(w_trans=50.0, w_reg=S.IK_W_REG, w_smooth=0.1):
    cm = cpin.Model(m); cd = cm.createData()
    cq = casadi.SX.sym("q", m.nq, 1); tl = casadi.SX.sym("tl", 4, 4); tr = casadi.SX.sym("tr", 4, 4)
    cpin.framesForwardKinematics(cm, cd, cq)
    tr_f = casadi.Function("t", [cq, tl, tr], [casadi.vertcat(cd.oMf[FL].translation - tl[:3, 3],
                                                              cd.oMf[FR].translation - tr[:3, 3])])
    ro_f = casadi.Function("r", [cq, tl, tr], [casadi.vertcat(cpin.log3(cd.oMf[FL].rotation @ tl[:3, :3].T),
                                                              cpin.log3(cd.oMf[FR].rotation @ tr[:3, :3].T))])
    o = casadi.Opti()
    q = o.variable(m.nq); ql = o.parameter(m.nq); pl = o.parameter(4, 4); pr = o.parameter(4, 4)
    o.subject_to(o.bounded(m.lowerPositionLimit, q, m.upperPositionLimit))
    o.minimize(w_trans * casadi.sumsqr(tr_f(q, pl, pr)) + casadi.sumsqr(ro_f(q, pl, pr))
               + w_reg * casadi.sumsqr(q) + w_smooth * casadi.sumsqr(q - ql))
    o.solver("ipopt", {"expand": True, "detect_simple_bounds": True, "calc_lam_p": False, "print_time": False,
                       "ipopt.sb": "yes", "ipopt.print_level": 0, "ipopt.max_iter": 200, "ipopt.tol": 1e-10,
                       "ipopt.acceptable_tol": 1e-9, "ipopt.warm_start_init_point": "yes"})
    return o, q, ql, pl, pr


W = np.array([0.4, 0.3, 0.2, 0.1])


def run(prob, goal, q0, steps, use_filter):
    o, q, ql, pl, pr = prob
    o.set_value(pl, goal[0]); o.set_value(pr, goal[1])
    cur, queue, stats = q0.copy(), [], []
    for _ in range(steps):
        o.set_initial(q, cur); o.set_value(ql, cur)
        try:
            o.solve(); s = np.array(o.value(q)).ravel(); stats.append(o.stats()["return_status"])
        except Exception:
            s = np.array(o.debug.value(q)).ravel(); stats.append("FAILED")
        if use_filter:
            if not queue or not np.array_equal(s, queue[-1]):
                queue = (queue + [s])[-4:]
            cur = queue[-1] if len(queue) < 4 else np.sum(np.array(queue) * W[::-1, None], axis=0)
        else:
            cur = s
    return cur, stats


def errors(q, goal):
    tips = fk(q)
    pos = max(np.linalg.norm(tips[i][:3, 3] - goal[i][:3, 3]) for i in (0, 1)) * 1000
    rot = max(np.degrees(np.linalg.norm(pin.log3(tips[i][:3, :3] @ goal[i][:3, :3].T))) for i in (0, 1))
    return pos, rot


def min_sv(q):
    """Smallest singular value of the fingertip POSITION Jacobian (m/rad): small = near a singular pose."""
    pin.computeJointJacobians(m, d, np.asarray(q, float)); pin.framesForwardKinematics(m, d, np.asarray(q, float))
    sv = []
    for f in (FL, FR):
        J = pin.getFrameJacobian(m, d, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        sv.append(np.linalg.svd(J[:, np.abs(J).sum(0) > 1e-9], compute_uv=False).min())
    return min(sv)


variants = [("A now", dict(), 30, True), ("B 150 steps", dict(), 150, True), ("C no reg", dict(w_reg=0.0), 30, True),
            ("D no filter", dict(), 30, False), ("E raw solve", dict(w_reg=0.0, w_smooth=0.0), 1, False),
            ("F trans 1000", dict(w_trans=1000.0), 30, True)]
probs = {name: build(**kw) for name, kw, _, _ in variants}

rng = np.random.default_rng(0)                      # same goals as test_tcp_ik.py
q0 = S.HOME_Q.copy()
goals, svs = [], []
for k in range(20):
    dq = rng.normal(0, 0.15, 14)
    qg = np.clip(q0 + dq, m.lowerPositionLimit[:14] + 0.05, m.upperPositionLimit[:14] - 0.05)
    goals.append(fk(qg)); svs.append(min_sv(qg))

print(f"   goal poses: smallest fingertip-Jacobian singular value median {np.median(svs):.3f} min {min(svs):.3f} m/rad")
res = {}
for name, kw, steps, filt in variants:
    pe, re, st, tt = [], [], [], []
    for g in goals:
        t0 = time.perf_counter()
        q, stats = run(probs[name], g, q0, steps, filt)
        tt.append((time.perf_counter() - t0) / steps * 1000)
        p, r = errors(q, g); pe.append(p); re.append(r); st += stats
    res[name] = pe
    bad = sum(s not in ("Solve_Succeeded", "Solved_To_Acceptable_Level") for s in st)
    print(f"{name:13s}: tip err mm median {np.median(pe):6.2f} max {max(pe):6.2f} | rot err deg median {np.median(re):5.2f}"
          f" | solver not converged {bad}/{len(st)} | ms/solve {np.median(tt):.1f}")
worst = int(np.argmax(res["A now"]))
print(f"   worst goal in A: #{worst} err {res['A now'][worst]:.2f} mm, Jacobian sv {svs[worst]:.3f} m/rad;"
      f" correlation(err, 1/sv) = {np.corrcoef(res['A now'], 1 / np.array(svs))[0, 1]:.2f}")
print("DONE - paste all lines above into the chat")
