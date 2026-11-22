"""Save one head-camera point cloud frame from the running sim to an .npz (2026-11-21).

The sim must run the Real task with the fitted head camera and depth on, e.g.:
    cd ~/VLA/unitree_sim_isaaclab && G1_REAL_HEAD_CAM=fitted G1_REAL_HEAD_DEPTH=1 G1_REAL_HEAD_DEPTH_STRIDE=2 python sim_main.py --device cpu \
        --enable_cameras --task Isaac-Stack-RgyBlock-Real-G129-Dex1-Joint --enable_dex1_dds --robot_type g129
Then (conda env "tv"):
    python ~/VLA/teleop_web/tools/save_cloud_frame.py ~/VLA/teleop_web/clouds/real_task_frame.npz
and replay with it:  bash ~/VLA/teleop_web/run.sh --cloud-file ~/VLA/teleop_web/clouds/real_task_frame.npz
The npz holds xyz (N, 3) float32 metres in the robot pelvis frame, rgb (N, 3) uint8 and meta (json string).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cloud  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--addr", default="tcp://127.0.0.1:55558", help="sim head depth publisher")
    ap.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for a frame")
    ap.add_argument("--voxel", type=float, default=0.005, help="voxel size (m) for the saved frame")
    args = ap.parse_args()

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    sock.connect(args.addr)
    print(f"waiting up to {args.timeout:.0f} s for a frame on {args.addr} ...")
    try:
        parts = sock.recv_multipart()
    except zmq.Again:
        sys.exit("No frame. Is the sim running the Real task with G1_REAL_HEAD_CAM=fitted G1_REAL_HEAD_DEPTH=1?")
    header = json.loads(parts[0].decode())
    xyz, rgb = cloud.frame_from_zmq(parts)
    xyz, rgb = cloud.voxel_down(xyz, rgb, voxel=args.voxel, max_points=200000)
    meta = {"source": args.addr, "header": header, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame": "robot pelvis (x forward, y left, z up), metres", "voxel_m": args.voxel, "crop": cloud.CROP}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, xyz=xyz.astype(np.float32), rgb=rgb.astype(np.uint8), meta=json.dumps(meta))
    print(f"saved {len(xyz)} points to {args.out}")


if __name__ == "__main__":
    main()
