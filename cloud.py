"""Head-camera point cloud for the Quest teleop page (2026-11-21).

Depth + RGB frames come from the sim's head_depth_pub (unitree_sim_isaaclab/tools/head_depth_pub.py, enabled with
G1_REAL_HEAD_DEPTH=1 in the Real task with G1_REAL_HEAD_CAM=fitted): zmq PUB tcp://<sim>:55558, multipart
[json {"h", "w", "stride", "t"}, depth uint16 mm (h, w), rgb uint8 (h, w, 3)], every stride-th pixel of 640 x 480 (G1_REAL_HEAD_DEPTH_STRIDE=2 for 320 x 240, default 4: 160 x 120).

Points are expressed in the robot PELVIS frame (x forward, y left, z up), the frame the page draws the robot in, so the
cloud lands in front of the digital robot on its own. Head camera = the fitted sim head camera (298 px focal at
640 x 480, 8.6 deg down and 8.7 deg left of the URDF D435, torso pitched 7.4 deg / yawed -1 deg like the Real task),
computed with ~/xr_teleoperate/teleop/utils/ghost_overlay.py and frozen here so the server needs no pinocchio.

Binary message to the browser, tag 4: [uint8 4][uint32 n][int16 xyz mm (n*3)][uint8 rgb (n*3)], little endian.
"""
import json
import struct

import numpy as np

# camera optical frame (x right, y down, z forward) -> pelvis frame
CAM_IN_PELVIS = np.array([[0.049994, -0.895575, 0.442092, 0.108835],
                          [-0.989517, 0.015632, 0.143567, 0.015633],
                          [-0.135486, -0.444635, -0.885406, 0.462868],
                          [0.0, 0.0, 0.0, 1.0]])
FOCAL_PX, CX, CY = 298.0, 320.0, 240.0          # full 640 x 480 head image (sim: the Real task's FITTED head camera,
# the only one that renders depth). A frame header may carry its own "fx", "fy", "cx", "cy" (full-resolution pixels,
# e.g. a real head D435i: fx 604.20, fy 603.95, cx 322.06, cy 248.55) and "cam_in_pelvis" (4x4); those win.

# keep the work area only (pelvis frame, metres): the table and what is on it, not the room
CROP = {"x": (0.10, 1.00), "y": (-0.60, 0.60), "z": (-0.10, 0.40), "range": (0.15, 1.50)}
VOXEL_M = 0.007                                  # 2026-11-21: was 0.01 (user: twice as dense)
DENSITY = {"normal": 25000, "dense": 50000}      # max points per frame (9 bytes each); was 12k / 25k
TAG_CLOUD_I16 = 4


def depth_to_points(header, depth_mm, rgb):
    """One head_depth_pub frame -> pelvis-frame points (N, 3) float32 m and colours (N, 3) uint8 (cropped)."""
    h, w, s = int(header["h"]), int(header["w"]), int(header["stride"])
    fx, fy = float(header.get("fx", FOCAL_PX)), float(header.get("fy", header.get("fx", FOCAL_PX)))
    cx, cy = float(header.get("cx", CX)), float(header.get("cy", CY))
    cam = np.array(header["cam_in_pelvis"], dtype=np.float64) if "cam_in_pelvis" in header else CAM_IN_PELVIS
    z = np.frombuffer(depth_mm, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    col = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
    v, u = np.mgrid[0:h, 0:w]
    ok = (z > CROP["range"][0]) & (z < CROP["range"][1])
    zc = z[ok]
    p_cam = np.stack([(u[ok] * s - cx) / fx * zc, (v[ok] * s - cy) / fy * zc, zc], axis=1)
    p = p_cam @ cam[:3, :3].T + cam[:3, 3]
    return crop(p.astype(np.float32), col[ok])


def crop(xyz, rgb):
    m = np.ones(len(xyz), dtype=bool)
    for i, k in enumerate("xyz"):
        m &= (xyz[:, i] >= CROP[k][0]) & (xyz[:, i] <= CROP[k][1])
    return xyz[m], rgb[m]


def voxel_down(xyz, rgb, voxel=VOXEL_M, max_points=DENSITY["normal"]):
    """One point per voxel (the first one), then evenly thinned to max_points."""
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    xyz, rgb = xyz[idx], rgb[idx]
    if len(xyz) > max_points:
        keep = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def pack_tag4(xyz, rgb):
    mm = np.clip(np.round(np.asarray(xyz, dtype=np.float64) * 1000.0), -32768, 32767).astype("<i2")
    return (struct.pack("<BI", TAG_CLOUD_I16, len(mm)) + mm.tobytes()
            + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def frame_from_zmq(parts):
    header = json.loads(parts[0].decode())
    return depth_to_points(header, parts[1], parts[2])


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["xyz"].astype(np.float32), d["rgb"].astype(np.uint8)
