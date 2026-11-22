"""Pick the chess board placement from chessreach/reach_grid.npz (tools/chess_reach.py) and draw the reach map.
Pelvis frame: x forward, y left. Robot = White: rank 1 nearest the robot, a-file on the robot's left.
A square is "ok" for an arm when the grid says reachable at grasp height, +5 cm and +10 cm, and at +-4 cm in x and y."""
import json, os, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chessreach")
S = float(sys.argv[1]) if len(sys.argv) > 1 else 0.035
TABLE_NEAR_X = 0.22 + 0.01          # table near edge (TABLE_GAP 0.22) + 1 cm
g = np.load(os.path.join(OUT, "reach_grid.npz"))
xs, ys = g["xs"], g["ys"]
step = float(xs[1] - xs[0])


def cell(G, k, x, y):
    i, j = int(round((x - xs[0]) / step)), int(round((y - ys[0]) / step))
    if i < 0 or j < 0 or i >= len(xs) or j >= len(ys):
        return False
    return bool(G[k, i, j])


def margin_ok(G, x, y):
    if not all(cell(G, k, x, y) for k in range(3)):
        return 0
    shifts = [(0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04)]
    return 2 if all(cell(G, 0, x + dx, y + dy) for dx, dy in shifts) else 1      # 2 = with margin, 1 = bare


def squares(bx, by):
    out = {}
    for fi, f in enumerate("abcdefgh"):
        for ri, r in enumerate("12345678"):
            out[f + r] = (bx + (ri - 3.5) * S, by - (fi - 3.5) * S)   # rank 1 nearest, a-file on the robot's left
    return out


def evaluate(bx, by):
    res = {}
    for name, (x, y) in squares(bx, by).items():
        L, R = margin_ok(g["left"], x, y), margin_ok(g["right"], x, y)
        near = "left" if y > 0 else "right"
        res[name] = {"x": round(x, 4), "y": round(y, 4), "left": L, "right": R, "near": near,
                     "near_ok": (L if near == "left" else R)}
    return res


best = None
for bx in np.arange(TABLE_NEAR_X + 4 * S, 0.60, 0.01):
    for by in np.arange(-0.08, 0.081, 0.01):
        r = evaluate(bx, by)
        n2 = sum(v["near_ok"] == 2 for v in r.values()); n1 = sum(v["near_ok"] >= 1 for v in r.values())
        key = (n2, n1, -abs(by))
        if best is None or key > best[0]:
            best = (key, round(float(bx), 3), round(float(by), 3), r)
cands = []                                   # all placements as good as the best: take the middle of their x range
for bx in np.arange(TABLE_NEAR_X + 4 * S, 0.60, 0.01):
    r = evaluate(bx, best[2])
    n2 = sum(v["near_ok"] == 2 for v in r.values()); n1 = sum(v["near_ok"] >= 1 for v in r.values())
    if (n2, n1) == best[0][:2]:
        cands.append(round(float(bx), 3))
bx = cands[len(cands) // 2] if cands else best[1]
by = best[2]
res = evaluate(bx, by)
n2 = sum(v["near_ok"] == 2 for v in res.values()); n1 = sum(v["near_ok"] >= 1 for v in res.values())
both = sum(v["left"] == 2 and v["right"] == 2 for v in res.values())
summary = {"square_m": S, "board_center_pelvis": [bx, by], "x_range_equal": [cands[0], cands[-1]] if cands else None,
           "ok_with_margin_nearer_arm": n2, "reachable_nearer_arm": n1, "ok_both_arms": both,
           "table_near_edge_limit_x": TABLE_NEAR_X, "squares": res}
json.dump(summary, open(os.path.join(OUT, f"chess_place_{int(S * 1000)}mm.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "squares"}))
bad = sorted(k for k, v in res.items() if v["near_ok"] < 2)
print("squares without full margin (nearer arm):", bad)

# reach map image: one panel per arm, squares coloured 2 green / 1 yellow / 0 red, grid reach shown underneath
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
col = {2: "#2e9d4f", 1: "#e3b505", 0: "#c8312f"}
for ax, side in zip(axs[:2], ("left", "right")):
    G = g[side]
    ax.imshow((G[0] & G[1] & G[2]).T, origin="lower", extent=[xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2],
              cmap="Greys", alpha=0.35, aspect="equal")
    for n, v in res.items():
        ax.add_patch(Rectangle((v["x"] - S / 2, v["y"] - S / 2), S, S, color=col[v[side]], ec="k", lw=0.4))
        ax.text(v["x"], v["y"], n, ha="center", va="center", fontsize=5)
    ax.plot(0, 0, "k^", ms=10); ax.axvline(TABLE_NEAR_X - 0.01, color="saddlebrown", ls="--", lw=1)
    ax.set_title(f"{side} arm (grey: reachable at 3 heights)"); ax.set_xlabel("pelvis x forward (m)"); ax.set_ylabel("pelvis y left (m)")
    ax.set_xlim(0.0, 0.7); ax.set_ylim(-0.45, 0.45)
ax = axs[2]
for n, v in res.items():
    ax.add_patch(Rectangle((v["x"] - S / 2, v["y"] - S / 2), S, S, color=col[v["near_ok"]], ec="k", lw=0.4))
    ax.text(v["x"], v["y"], n, ha="center", va="center", fontsize=5)
for p, sq in (("K", "e1"), ("Q", "d1"), ("B", "c1")):
    ax.text(res[sq]["x"], res[sq]["y"] + 0.009, p, ha="center", fontsize=10, weight="bold", color="w")
ax.plot(0, 0, "k^", ms=10); ax.axvline(TABLE_NEAR_X - 0.01, color="saddlebrown", ls="--", lw=1, label="table near edge")
ax.set_xlim(0.0, 0.7); ax.set_ylim(-0.45, 0.45); ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7)
ax.set_title(f"nearer arm: {n2}/64 with margin, {n1}/64 reachable"); ax.set_xlabel("pelvis x forward (m)")
fig.suptitle(f"Chess board reach, squares {S * 100:.1f} cm, centre {bx:.2f} m ahead, {by:+.2f} m left "
             "(green: margin  yellow: reachable, no margin  red: not)")
fig.tight_layout(); p = os.path.join(OUT, f"chess_reach_{int(S * 1000)}mm.png"); fig.savefig(p, dpi=130); print("saved", p)
