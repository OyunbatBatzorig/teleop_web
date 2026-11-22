"""Head-camera depth + RGB publisher for the operator's point cloud (2026-10-27, sim teleop UX, item 6).

Enabled with G1_REAL_HEAD_DEPTH=1 (the Real task then also renders "distance_to_image_plane" for the head camera).
Publishes on zmq PUB tcp://*:G1_REAL_HEAD_DEPTH_PORT (default 55558), at most G1_REAL_HEAD_DEPTH_HZ (default 15),
every G1_REAL_HEAD_DEPTH_STRIDE-th pixel (default 4: 120 x 160). Message (multipart):
    [json header {"h", "w", "stride", "t"}, depth uint16 millimetres (h, w), rgb uint8 (h, w, 3)]
Sending runs in a background thread with a 1-slot queue, so the sim never waits on it.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import zmq

ENABLED = os.environ.get("G1_REAL_HEAD_DEPTH") == "1"
PORT = int(os.environ.get("G1_REAL_HEAD_DEPTH_PORT", "55558"))
HZ = float(os.environ.get("G1_REAL_HEAD_DEPTH_HZ", "15"))
STRIDE = int(os.environ.get("G1_REAL_HEAD_DEPTH_STRIDE", "4"))
# Camera model sent in the header (2026-11-22) so the receiver (~/VLA/teleop_web/cloud.py) needs no guessing.
# Default head camera = the real D435i (camera_configs.g1_front_camera: focal 18.88 / aperture 20 -> fx 604.16 px at
# 640 x 480, centred principal point), URDF d435 pose at the Real task's waist (yaw -1, pitch 7.4 deg) in the pelvis
# frame. With G1_REAL_HEAD_CAM=fitted no model is sent and the receiver uses its fitted-camera default (298 px).
if os.environ.get("G1_REAL_HEAD_CAM") == "fitted":
    CAM_MODEL = {}
else:
    CAM_MODEL = {"fx": 604.16, "fy": 604.16, "cx": 320.0, "cy": 240.0, "camera": "D435i",
                 "cam_in_pelvis": [[-0.017452, -0.819027, 0.573489, 0.108835],
                                   [-0.999848, 0.014296, -0.01001, 0.015633],
                                   [0.0, -0.573576, -0.819152, 0.462868],
                                   [0.0, 0.0, 0.0, 1.0]]}

_queue = queue.Queue(maxsize=1)
_last = 0.0
_thread = None


def _sender():
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{PORT}")
    print(f"[head_depth_pub] publishing head depth + rgb on tcp://*:{PORT} (stride {STRIDE}, <= {HZ} Hz)")
    while True:
        depth, rgb = _queue.get()
        header = {"h": depth.shape[0], "w": depth.shape[1], "stride": STRIDE, "t": time.time(), **CAM_MODEL}
        sock.send_multipart([json.dumps(header).encode(), depth.tobytes(), rgb.tobytes()])


def publish(depth_m, rgb):
    """depth_m: (H, W) or (H, W, 1) metres; rgb: (H, W, 3) uint8. Cheap no-op when rate-limited or disabled."""
    global _last, _thread
    if not ENABLED:
        return
    now = time.time()
    if now - _last < 1.0 / HZ:
        return
    _last = now
    if _thread is None:
        _thread = threading.Thread(target=_sender, daemon=True)
        _thread.start()
    d = np.asarray(depth_m).reshape(depth_m.shape[0], depth_m.shape[1])[::STRIDE, ::STRIDE]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    c = np.ascontiguousarray(np.asarray(rgb)[::STRIDE, ::STRIDE, :3]).astype(np.uint8)
    try:
        _queue.put_nowait((np.ascontiguousarray(d_mm), c))
    except queue.Full:
        pass
