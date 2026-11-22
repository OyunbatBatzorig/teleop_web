#!/usr/bin/env python3
"""Head D435i depth + colour publisher for the Quest teleop point cloud - runs ON THE ROBOT'S PC2 (2026-11-22).

Sends exactly what the sim's tools/head_depth_pub.py sends, so ~/VLA/teleop_web/cloud.py needs no change:
  zmq PUB tcp://*:<port> (default 55558), multipart
    [json header, depth uint16 millimetres (h, w), rgb uint8 (h, w, 3)]
  header: {"h", "w", "stride", "t", "fx", "fy", "cx", "cy", "camera", "cam_in_pelvis", "waist_source", "serial",
           "depth_scale"}; h / w = the strided image, fx .. cy in FULL-resolution pixels (cloud.py multiplies by stride).
Depth is aligned to the colour camera (rs.align), intrinsics come from the running colour stream (factory calibration).
cam_in_pelvis: colour optical frame (x right, y down, z forward) -> pelvis, from the URDF chain pelvis -> waist yaw ->
waist roll -> waist pitch -> torso -> d435_joint, with the waist angles from rt/lowstate (DDS domain 0, READ ONLY:
this program never publishes on DDS) when --waist lowstate, else waist = 0 ("waist_source": "static").
NOTE: the URDF d435 pose is the nominal mount; the colour imager sits ~15 mm off the d435_link origin (not modelled).

Only one process can open the D435i. If teleimager on PC2 uses this D435i as its head camera (cam_config_server.yaml
head_camera type "realsense" with this serial), stop that or point teleimager at the other head camera - see README_PC2.md.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

# ---- URDF g1_29dof_mode_15_with_dex1_1: pelvis -> d435_link
WAIST_ROLL_XYZ = (-0.0039635, 0.0, 0.044)           # waist_roll_joint origin (in waist_yaw_link)
D435_XYZ, D435_PITCH = (0.0576235, 0.01753, 0.42987), 0.8307767239493009   # d435_joint (in torso_link)
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],          # optical (x right, y down, z fwd) -> link (x fwd, y left, z up)
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])
WAIST_IDX = (12, 13, 14)                              # rt/lowstate motor order: waist yaw, roll, pitch


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def cam_in_pelvis(yaw=0.0, roll=0.0, pitch=0.0):
    """4x4: D435 colour optical frame -> pelvis frame for the given waist angles (rad)."""
    return (_T(_rot("z", yaw)) @ _T(_rot("x", roll), WAIST_ROLL_XYZ) @ _T(_rot("y", pitch))
            @ _T(_rot("y", D435_PITCH), D435_XYZ) @ _T(R_LINK_OPTICAL))


class WaistReader:
    """Latest waist angles from rt/lowstate (DDS domain 0, subscriber only)."""
    def __init__(self, network_interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, networkInterface=network_interface)
        self.q, self.t = (0.0, 0.0, 0.0), 0.0
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on, 1)

    def _on(self, m):
        self.q, self.t = tuple(float(m.motor_state[i].q) for i in WAIST_IDX), time.time()

    def get(self):
        return self.q if time.time() - self.t < 0.5 else None


def import_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        for p in ("~/librealsense/build/wrappers/python", "~/librealsense/build/Release", "~/librealsense/build"):
            p = os.path.expanduser(p)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        import pyrealsense2 as rs          # still missing -> see README_PC2.md
        return rs


def build_header(h, w, stride, intr, cam, waist_source, serial, depth_scale):
    return {"h": int(h), "w": int(w), "stride": int(stride), "t": time.time(),
            "fx": float(intr["fx"]), "fy": float(intr["fy"]), "cx": float(intr["cx"]), "cy": float(intr["cy"]),
            "camera": "D435i", "cam_in_pelvis": np.round(cam, 6).tolist(), "waist_source": waist_source,
            "serial": serial, "depth_scale": float(depth_scale)}


def to_mm_strided(depth_raw, depth_scale, stride):
    """uint16 sensor units -> uint16 millimetres, every stride-th pixel."""
    d = depth_raw[::stride, ::stride].astype(np.float32) * (depth_scale * 1000.0)
    return np.clip(np.round(d), 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=55558)
    ap.add_argument("--hz", type=float, default=15.0, help="frames sent per second (max)")
    ap.add_argument("--stride", type=int, default=2, help="every n-th pixel (2: 320 x 240 from 640 x 480)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, choices=(15, 30), help="camera frame rate")
    ap.add_argument("--serial", default=None, help="D435i serial (default: the first D435 found)")
    ap.add_argument("--waist", choices=("static", "lowstate"), default="static",
                    help="waist angles for cam_in_pelvis: static = 0, lowstate = read rt/lowstate (DDS domain 0, read only)")
    ap.add_argument("--network-interface", default=None, help="--waist lowstate: DDS interface, e.g. eth0")
    ap.add_argument("--list", action="store_true", help="list RealSense devices and exit")
    args = ap.parse_args()

    rs = import_realsense()
    ctx = rs.context()
    devs = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in ctx.query_devices()]
    print("[d435_pub] RealSense devices:", devs or "none", flush=True)
    if args.list:
        return
    serial = args.serial or next((s for n, s in devs if "D435" in n), None)
    if serial is None:
        sys.exit("[d435_pub] no D435 found (is teleimager or another program holding it? see README_PC2.md)")

    import zmq
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{args.port}")

    waist = WaistReader(args.network_interface) if args.waist == "lowstate" else None

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = {"fx": ci.fx, "fy": ci.fy, "cx": ci.ppx, "cy": ci.ppy}
    align = rs.align(rs.stream.color)
    print(f"[d435_pub] {serial}: {args.width}x{args.height}@{args.fps}, colour fx {ci.fx:.1f} fy {ci.fy:.1f} "
          f"cx {ci.ppx:.1f} cy {ci.ppy:.1f}, depth scale {depth_scale}, publishing tcp://*:{args.port} "
          f"(stride {args.stride}, <= {args.hz} Hz, waist {args.waist})", flush=True)
    last, n, t_log = 0.0, 0, time.time()
    try:
        while True:
            frames = align.process(pipe.wait_for_frames(5000))
            d, c = frames.get_depth_frame(), frames.get_color_frame()
            if not d or not c or time.time() - last < 1.0 / args.hz:
                continue
            last = time.time()
            depth = to_mm_strided(np.asanyarray(d.get_data()), depth_scale, args.stride)
            rgb = np.ascontiguousarray(np.asanyarray(c.get_data())[::args.stride, ::args.stride])
            wq = waist.get() if waist else None
            src = "lowstate" if wq is not None else ("static" if waist is None else "static (no lowstate)")
            hdr = build_header(depth.shape[0], depth.shape[1], args.stride, intr, cam_in_pelvis(*(wq or (0, 0, 0))),
                               src, serial, depth_scale)
            sock.send_multipart([json.dumps(hdr).encode(), depth.tobytes(), rgb.tobytes()])
            n += 1
            if time.time() - t_log > 10:
                print(f"[d435_pub] {n / (time.time() - t_log):.1f} frames/s sent, waist {src}", flush=True)
                n, t_log = 0, time.time()
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
