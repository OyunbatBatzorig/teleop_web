"""Offline test of SimSource._ik_guard (no sim, no robot). Run in env tv from ~/VLA/teleop_web:
    ~/miniconda3/envs/tv/bin/python tools/test_ik_guard.py
Shows the twitch xr_teleoperate's IK filter causes after one failed solve, and that the guard removes it."""
import os, sys, types
import numpy as np

XR = os.environ.get("XR_TELEOP_DIR", os.path.expanduser("~/xr_teleoperate"))
sys.path.insert(0, XR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

try:
    from sim_source import SimSource
    guard = SimSource._ik_guard
except Exception as e:  # heavy imports missing: fall back to the function body via source parsing
    print(f"(could not import sim_source: {e}); aborting"); sys.exit(1)


class FakeIK:
    """Mimics G1_29_ArmIK.solve_ik's filter / return behaviour."""
    def __init__(self):
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.init_data = np.zeros(14)

    def solve_ik(self, target_q, seed, fail=False, fail_value=0.0):
        sol = np.full(14, fail_value) if fail else target_q.copy()
        self.smooth_filter.add_data(sol)
        self.init_data = self.smooth_filter.filtered_data
        return (seed, None) if fail else (self.smooth_filter.filtered_data, None)


def run(use_guard, fail_value):
    ik = FakeIK()
    me = types.SimpleNamespace(ik=ik, ik_fail_count=0)
    out, cmd = [], np.full(14, 1.0)
    for k in range(10):                       # the hand moves slowly: target 1.00, 1.01, 1.02 ... rad
        target = np.full(14, 1.0 + 0.01 * k)
        sol, _ = ik.solve_ik(target, cmd, fail=(k == 5), fail_value=fail_value)
        if use_guard:
            sol = guard(me, sol, cmd)
        cmd = np.array(sol, dtype=float)
        out.append(float(cmd[0]))
    return out

for fv in (0.0, float("nan")):
    raw, fixed = run(False, fv), run(True, fv)
    print("  (diffs with guard:", np.round(np.diff(fixed), 4), ")")
    print(f"failed solution = {fv}:  without guard {np.round(raw, 2)}  with guard {np.round(fixed, 2)}")
    # with the guard the command never drops: it holds at the failed step, then follows again
    assert all(np.isfinite(fixed)) and min(np.diff(fixed)) > -0.005, "guard must not let the command jump back"
    assert abs(fixed[-1] - 1.09) < 0.02
print("PASS: after a failed IK solve the command holds, then follows the target again (no twitch toward 0, no NaN)")
