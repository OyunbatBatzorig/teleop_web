# Head D435i point cloud publisher (robot PC2)

`d435_depth_pub.py` runs on the G1's PC2 (Jetson, `192.168.123.164`). It sends the head D435i depth and colour
to the teleop server in the same format as the sim (`tools/head_depth_pub.py`): zmq PUB on port 55558. The server
turns it into the point cloud on the Quest.

**Not tested on the robot yet (2026-11-22).**

## Copy to PC2 (from the teleop PC)
```bash
ssh unitree@192.168.123.164 'mkdir -p ~/teleop_pc2'
scp ~/VLA/teleop_web/robot_pc2/{d435_depth_pub.py,run_d435_pub.sh,d435-depth-pub.service,README_PC2.md} unitree@192.168.123.164:~/teleop_pc2/
```

## Dependencies on PC2
- `pyrealsense2`. PC2 has a source build: the script also looks in `~/librealsense/build/wrappers/python`
  (and `~/librealsense/build/Release`). Check it with `python3 -c "import pyrealsense2"`. If that fails, find the
  `pyrealsense2*.so` with `find ~/librealsense -name "pyrealsense2*.so"` and add its folder to `PYTHONPATH`.
- `pyzmq`, `numpy`: `pip3 install --user pyzmq numpy`
- Optional, only for `--waist lowstate`: `unitree_sdk2py`. This reads `rt/lowstate` and never publishes.

## Run
```bash
cd ~/teleop_pc2
python3 d435_depth_pub.py --list               # which RealSense cameras are there (serials)
bash run_d435_pub.sh                           # start in the background, log ~/d435_pub.log
bash run_d435_pub.sh stop
```
Options: `--stride 2` (320 x 240 points, default), `--hz 15`, `--fps 30`, `--serial <D435i serial>`,
`--waist lowstate --network-interface eth0`. The last one makes the cloud follow the waist; without it, waist = 0.

Autostart (optional): see the comments in `d435-depth-pub.service`.

On the teleop PC:
```bash
bash ~/VLA/teleop_web/run.sh --source real ... --cloud-live tcp://192.168.123.164:55558
```
For `--source real`, `--cloud-live tcp://<img-server-ip>:55558` is already the default.

## Camera conflict with teleimager
Only one program can open the D435i at a time.
- teleimager's `cam_config_server.yaml` on PC2: if `head_camera` is `type: uvc` (the binocular head camera, the
  factory default), there is no conflict.
- If `head_camera` is `type: realsense` with the D435i serial, teleimager and this publisher fight over the camera.
  Either give teleimager the other head camera, or stop teleimager's head camera while using the point cloud.
- The wrist cameras are not affected.

## Camera pose
`cam_in_pelvis` comes from the URDF (`d435_joint` on `torso_link`, waist chain) and is sent in every frame header.
- The colour imager sits ~1.5 cm beside the `d435_link` origin. That offset is not modelled yet, so expect the cloud
  to be off by about that much.
- Intrinsics come from the running colour stream (factory calibration). Depth is aligned to colour.
