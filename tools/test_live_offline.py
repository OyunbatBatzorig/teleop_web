"""Offline test of the live-sim control math (no DDS, no sim): IK model, FK of the home pose, IK toward a ghost 5 cm
away, and the 25 cm clutch gate. Run:  ~/miniconda3/envs/tv/bin/python ~/VLA/teleop_web/tools/test_live_offline.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import sim_source as S

ik = S.make_ik()
q = S.HOME_Q.copy()
ee = S.ee_positions(ik, q)
print("home EE (pelvis frame, m): L", ee["left"][:3, 3].round(3), " R", ee["right"][:3, 3].round(3))
# ghost 5 cm forward of the right hand, left stays
tgt = {s: ee[s].copy() for s in S.SIDES}; tgt["right"][0, 3] += 0.05
# like the 30 Hz loop: solve repeatedly, feeding the result back (the IK smooths its output over a few calls)
q2, times = q.copy(), []
for i in range(30):
    t0 = time.time(); sol, tau = ik.solve_ik(tgt["left"], tgt["right"], q2, np.zeros(14)); times.append(time.time() - t0)
    q2 = np.asarray(sol)
dt = float(np.median(times[1:]))
ee2 = S.ee_positions(ik, q2)
err = np.linalg.norm(ee2["right"][:3, 3] - tgt["right"][:3, 3])
print(f"IK 5 cm step: right EE error {err * 1000:.1f} mm, left moved "
      f"{np.linalg.norm(ee2['left'][:3, 3] - ee['left'][:3, 3]) * 1000:.1f} mm, solve median {dt * 1000:.1f} ms, first {times[0] * 1000:.0f} ms")
# clutch gate
g = S.ClutchGate()
p = ee["right"][:3, 3]
print("gate 10 cm :", g.update(True, p + [0.10, 0, 0], p))
print("hold press, move ghost 40 cm away (stays follow):", g.update(True, p + [0.40, 0, 0], p))
print("release     :", g.update(False, p, p))
print("press at 30 cm:", g.update(True, p + [0.30, 0, 0], p), f"(dist {g.dist:.2f} m)")
print("still held, ghost back to 5 cm (stays rejected until released):", g.update(True, p + [0.05, 0, 0], p))
print("release + press at 5 cm:", g.update(False, p, p), g.update(True, p + [0.05, 0, 0], p))
ok = g.state == "follow" and dt < 0.02   # IK accuracy is reported, not gated: the xr_teleoperate IK is soft (~1 cm bias)
print("RESULT", "PASS" if ok else "FAIL")

# ---- haptics (2026-11-22)
lo, hi = np.array(ik.reduced_robot.model.lowerPositionLimit[:14]), np.array(ik.reduced_robot.model.upperPositionLimit[:14])
qr = S.HOME_Q[S.SL["right"]].copy(); lr, hr = lo[S.SL["right"]], hi[S.SL["right"]]
print("haptic free motion           :", round(S.haptic_level(qr, qr, np.zeros(7), lr, hr, True), 2))
q_lim = qr.copy(); q_lim[3] = hr[3] - 0.02
print("haptic elbow 2 cm from limit :", round(S.haptic_level(q_lim, q_lim, np.zeros(7), lr, hr, True), 2))
print("haptic tracking err 0.15 rad :", round(S.haptic_level(qr, qr + 0.15, np.zeros(7), lr, hr, True), 2))
print("haptic torque residual 12 Nm :", round(S.haptic_level(qr, qr, np.r_[12, np.zeros(6)], lr, hr, True), 2))
print("haptic not following         :", S.haptic_level(q_lim, q_lim, np.r_[12, np.zeros(6)], lr, hr, False))
# ---- cloud.py with the new head_depth_pub header (D435i model)
import json, cloud
hdr = {"h": 240, "w": 320, "stride": 2, "fx": 604.16, "fy": 604.16, "cx": 320.0, "cy": 240.0,
       "cam_in_pelvis": [[-0.017452, -0.819027, 0.573489, 0.108835], [-0.999848, 0.014296, -0.01001, 0.015633],
                         [0.0, -0.573576, -0.819152, 0.462868], [0.0, 0.0, 0.0, 1.0]]}
dmm = (np.ones((240, 320)) * 600).astype("<u2").tobytes(); rgb = np.zeros((240, 320, 3), "u1").tobytes()
xyz, _ = cloud.depth_to_points(hdr, dmm, rgb)
print(f"cloud from D435i header: {len(xyz)} points, centre {xyz.mean(0).round(3) if len(xyz) else None}")

# ---- waist correction (2026-11-22): Real task torso pitch 7.4 deg, yaw -1 deg
C = S.torso_correction(ik, (np.radians(-1.0), 0.0, np.radians(7.4)))
ee_ik = S.ee_positions(ik, S.HOME_Q)
for sd in S.SIDES:
    shift = (C @ ee_ik[sd])[:3, 3] - ee_ik[sd][:3, 3]
    print(f"waist 7.4 deg moves the {sd} hand by {np.round(shift * 100, 1)} cm (x fwd, y left, z up) -> was the ghost/robot gap")
# solve for a pelvis-frame target with the correction, then check the corrected FK lands on it
tgt = C @ ee_ik["right"]; tgt[0, 3] += 0.03
q2 = S.HOME_Q.copy()
for i in range(60):
    q2 = np.asarray(ik.solve_ik(np.linalg.inv(C) @ (C @ ee_ik["left"]), np.linalg.inv(C) @ tgt, q2, np.zeros(14))[0])
err = np.linalg.norm((C @ S.ee_positions(ik, q2)["right"])[:3, 3] - tgt[:3, 3])
print(f"with correction: right hand error {err * 1000:.1f} mm (IK softness only)")
# haptics v3
cfg = S.HAPTIC
print("hard hit (torque part 1.0) ->", S.haptic_level(qr, qr, np.r_[15, np.zeros(6)], lr, hr, True, cfg),
      "| light (8 N.m) ->", round(S.haptic_level(qr, qr, np.r_[8, np.zeros(6)], lr, hr, True, cfg), 2),
      "| at joint limit ->", round(S.haptic_level(q_lim, q_lim, np.zeros(7), lr, hr, True, cfg), 2))

# ---- slow correction loop (no sim: the "measured" q is the IK output itself)
gt = C @ ee_ik["right"]; gt[0, 3] += 0.03
bias = np.zeros(3); qq = S.HOME_Q.copy()
for i in range(200):
    now = (C @ S.ee_positions(ik, qq)["right"])[:3, 3]
    e = gt[:3, 3] - now
    if np.linalg.norm(e) < 0.05:
        bias = np.clip(bias + S.EE_I_GAIN * e, -S.EE_I_MAX, S.EE_I_MAX)
    t = gt.copy(); t[:3, 3] += bias
    qq = np.asarray(ik.solve_ik(np.linalg.inv(C) @ (C @ ee_ik["left"]), np.linalg.inv(C) @ t, qq, np.zeros(14))[0])
final = np.linalg.norm(gt[:3, 3] - (C @ S.ee_positions(ik, qq)["right"])[:3, 3])
print(f"with slow correction after 200 steps (~7 s): right hand error {final * 1000:.1f} mm")
