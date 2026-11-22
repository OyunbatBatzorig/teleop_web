"""grip_q: width slider 0 / 1 / 50 / 100 % and trigger -> Dex1 command (2026-11-22f, width 0 used to give fully open)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim_source import grip_q, GRIP_FULL_RAD

W = 83.0                                                      # Dex1-1 profile: slider 0..83 mm
cases = [  # (trigger, width_mm, expected q)
    (0.0, 0.0, 0.0), (0.0, 0.01 * W, 0.01 * W / 90 * 5.4), (0.0, 0.5 * W, 0.5 * W / 90 * 5.4), (0.0, W, W / 90 * 5.4),
    (1.0, W, 0.0), (0.5, W, 0.5 * W / 90 * 5.4), (1.0, 0.0, 0.0), (0.0, None, GRIP_FULL_RAD), (0.0, 500.0, GRIP_FULL_RAD),
    (0.0, -3.0, 0.0), (None, 40.0, 40 / 90 * 5.4)]
bad = [(t, w, e, grip_q(t, w)) for t, w, e in cases if abs(grip_q(t, w) - e) > 1e-9]
for b in bad:
    print("FAIL trigger %s width %s: expected %.4f got %.4f" % b)
print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
