"""Synthetic point cloud for testing the Quest page without the sim (2026-11-21).

Table top at the Real task's height (0.794 m world, pelvis at 0.775 m -> z = +0.019 in the pelvis frame), near edge
0.22 m in front of the pelvis, 0.76 m deep; three 4 cm blocks (red, green, yellow) like G1_Stack_Block, seen from the
head camera only (points facing away from the camera are dropped so it looks like a depth camera's view).
    python ~/VLA/teleop_web/tools/make_test_cloud.py ~/VLA/teleop_web/clouds/test_table_blocks.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cloud  # noqa: E402

TABLE_Z, NEAR, DEPTH, HALF_W = 0.019, 0.22, 0.76, 0.6
BLOCKS = [((0.40, 0.12), (220, 40, 50)), ((0.45, -0.05), (40, 170, 70)), ((0.36, -0.20), (235, 200, 40))]
S = 0.04


def main(out):
    rng = np.random.default_rng(0)
    step = 0.004
    xs, ys = np.meshgrid(np.arange(NEAR, NEAR + DEPTH, step), np.arange(-HALF_W, HALF_W, step), indexing="ij")
    table = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, TABLE_Z)], 1)
    wood = np.array([181, 140, 96]) + rng.integers(-12, 12, (len(table), 1))
    pts, cols = [table], [np.clip(wood, 0, 255)]
    cam = cloud.CAM_IN_PELVIS[:3, 3]
    g = np.arange(-S / 2, S / 2 + 1e-9, 0.004)
    for (bx, by), c in BLOCKS:
        centre = np.array([bx, by, TABLE_Z + S / 2])
        for axis in range(3):
            for sign in (-1, 1):
                normal = np.zeros(3); normal[axis] = sign
                if np.dot(cam - (centre + normal * S / 2), normal) <= 0:
                    continue                                   # face points away from the head camera
                a, b = [i for i in range(3) if i != axis]
                u, v = np.meshgrid(g, g, indexing="ij")
                f = np.zeros((u.size, 3)); f[:, a] = u.ravel(); f[:, b] = v.ravel(); f[:, axis] = sign * S / 2
                pts.append(centre + f)
                shade = 1.0 if axis == 2 else 0.8
                cols.append(np.tile(np.array(c) * shade, (len(f), 1)))
    xyz = np.concatenate(pts).astype(np.float32)
    rgb = np.clip(np.concatenate(cols), 0, 255).astype(np.uint8)
    xyz, rgb = cloud.crop(xyz, rgb)
    meta = {"source": "synthetic (make_test_cloud.py)", "frame": "robot pelvis (x forward, y left, z up), metres"}
    np.savez_compressed(out, xyz=xyz, rgb=rgb, meta=json.dumps(meta))
    print(f"saved {len(xyz)} points to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clouds", "test_table_blocks.npz"))
