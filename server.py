"""Quest 3 web teleop: episode interface server (v1, 2026-11-09; v1.2 point cloud 2026-11-21; v1.3 live sim 2026-11-22: --source sim, see sim_source.py).

Serves the WebXR page (web/) over HTTPS, the G1 + Dex1 URDF and meshes, and one WebSocket (/ws) that streams
robot joint states and camera JPEGs to the headset and receives the controllers' state from it.

v1 source = "replay": plays back recorded xr_teleoperate episodes (data.json + colors/*.jpg), so the layout can be
checked in the headset without the sim or the robot. Live sim / real robot comes in the next step (same messages).

Run (on the PC, in the conda env that runs xr_teleoperate, needs aiohttp):
    python ~/VLA/teleop_web/server.py
then open https://<PC-IP>:8443 in the Quest 3 browser (accept the self-signed certificate once).

Messages to the browser:
  text   {"t":"state", "arm":[14], "waist":[3], "legs":[12], "grip":[2] (0 closed .. 5.4 open), "episode":"episode_0014",
          "episode_idx":i, "episode_count":n, "frame":k, "frames":N, "running":bool, "elapsed":s, "source":"replay"}
  binary 1 byte tag + payload: 0 head JPEG, 1 left wrist JPEG, 2 right wrist JPEG,
         3 point cloud (uint32 n, float32 xyz[n*3] in the robot pelvis frame, uint8 rgb[n*3])  (legacy, not sent)
         4 point cloud (uint32 n, int16 xyz mm[n*3] in the robot pelvis frame, uint8 rgb[n*3])  (v1.2, see cloud.py)
  Point cloud sources (v1.2, 2026-11-21): --cloud-file <npz> (resent ~5 Hz, works with replay, no sim) or
  --cloud-live [tcp://127.0.0.1:55558] (sim head depth, Real task with G1_REAL_HEAD_CAM=fitted G1_REAL_HEAD_DEPTH=1).
Messages from the browser:
  text   {"t":"xr", ...controller state...}  (logged once a second in v1; used for control in step 2)
         {"t":"grip_settings", "side":"left"|"right", "gripper":..., "force":0..1, "width":0..1, "force_n":N, "width_mm":mm}
         (stored per side, printed when it changes; v1.1 2026-11-10)
  state also carries "gripper": "dex1" | "rubber_hand" (--gripper; the page's ?gripper= URL param wins over it)
         {"t":"cmd", "cmd": "toggle_episode" | "back" | "next_episode" | "prev_episode"}
"""
import argparse
import asyncio
import json
import os
import ssl
import threading
import time
from pathlib import Path

import cloud

from aiohttp import web, WSMsgType

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TAG_HEAD, TAG_LWRIST, TAG_RWRIST, TAG_CLOUD = 0, 1, 2, 3
# colors/<frame>_color_<i>.jpg written by xr_teleoperate's EpisodeWriter: 0 head, 1 left wrist, 2 right wrist
COLOR_TAGS = (("color_0", TAG_HEAD), ("color_1", TAG_LWRIST), ("color_2", TAG_RWRIST))


def find_cert(args):
    cands = [(args.cert, args.key),
             (os.getenv("XR_TELEOP_CERT"), os.getenv("XR_TELEOP_KEY")),
             (HOME / ".config/xr_teleoperate/cert.pem", HOME / ".config/xr_teleoperate/key.pem"),
             (HOME / "xr_teleoperate/teleop/televuer/cert.pem", HOME / "xr_teleoperate/teleop/televuer/key.pem")]
    for c, k in cands:
        if c and k and Path(c).exists() and Path(k).exists():
            return str(c), str(k)
    raise SystemExit("No HTTPS certificate found. Pass --cert and --key (the Quest browser needs HTTPS for WebXR).")


class ReplaySource:
    """Plays recorded episodes. 'toggle_episode' starts playback from frame 0 / pauses it."""

    def __init__(self, episodes_dir):
        self.dir = Path(episodes_dir).expanduser()
        self.episodes = sorted(p for p in self.dir.glob("episode_*") if (p / "data.json").exists())
        if not self.episodes:
            raise SystemExit(f"No episode_*/data.json in {self.dir}")
        self.idx = len(self.episodes) - 1
        self.running = False
        self.frame = 0
        self.t_start = None
        self.load()

    def load(self):
        ep = self.episodes[self.idx]
        t = time.time()
        with open(ep / "data.json", encoding="utf-8") as f:
            self.data = json.load(f)["data"]
        self.frame = 0
        self.running = False
        self.t_start = None
        print(f"[replay] {ep.name}: {len(self.data)} frames ({time.time() - t:.1f} s to load)", flush=True)

    def command(self, cmd):
        if cmd == "toggle_episode":
            self.running = not self.running
            if self.running:
                self.frame, self.t_start = 0, time.time()
        elif cmd in ("next_episode", "prev_episode"):
            step = 1 if cmd == "next_episode" else -1
            self.idx = (self.idx + step) % len(self.episodes)
            self.load()
        print(f"[cmd] {cmd} -> {self.episodes[self.idx].name} running={self.running}", flush=True)

    def step(self):
        if self.running:
            self.frame += 1
            if self.frame >= len(self.data):        # end of the recording: stop like a finished episode
                self.frame, self.running = len(self.data) - 1, False

    def state(self):
        it = self.data[self.frame]["states"]
        body = it["body"]["qpos"]
        return {"t": "state", "source": "replay",
                "arm": it["left_arm"]["qpos"] + it["right_arm"]["qpos"],
                "waist": body[12:15] if len(body) >= 15 else [0.0, 0.0, 0.0],
                "legs": body[0:12] if len(body) >= 12 else [0.0] * 12,
                "grip": [it["left_ee"]["qpos"][0], it["right_ee"]["qpos"][0]],
                "episode": self.episodes[self.idx].name, "episode_idx": self.idx,
                "episode_count": len(self.episodes), "frame": self.frame, "frames": len(self.data),
                "running": self.running, "elapsed": (time.time() - self.t_start) if self.running else 0.0}

    def images(self):
        ep = self.episodes[self.idx]
        colors = self.data[self.frame].get("colors") or {}
        out = []
        for key, tag in COLOR_TAGS:
            rel = colors.get(key)
            if rel and (ep / rel).exists():
                out.append(bytes([tag]) + (ep / rel).read_bytes())
        return out


class CloudFeed:
    """Packs the point cloud (tag 4) in a background thread; the 30 Hz broadcast loop only picks up the latest packet."""

    def __init__(self, args):
        self.max_points = cloud.DENSITY[args.cloud_density]
        self.lock = threading.Lock()
        self.packet, self.fresh = None, False
        self.sent = 0
        if args.cloud_file:
            xyz, rgb = cloud.load_npz(os.path.expanduser(args.cloud_file))
            xyz, rgb = cloud.crop(xyz, rgb)
            xyz, rgb = cloud.voxel_down(xyz, rgb, max_points=self.max_points)
            self.static = cloud.pack_tag4(xyz, rgb)
            print(f"[cloud] {args.cloud_file}: {len(xyz)} points, {len(self.static) / 1024:.0f} KB per send, ~5 Hz",
                  flush=True)
            threading.Thread(target=self._file_loop, daemon=True).start()
        else:
            threading.Thread(target=self._live_loop, args=(args.cloud_live,), daemon=True).start()

    def _put(self, pkt):
        with self.lock:
            self.packet, self.fresh = pkt, True

    def take(self):
        with self.lock:
            if not self.fresh:
                return None
            self.fresh = False
            self.sent += 1
            return self.packet

    def _file_loop(self):
        while True:
            self._put(self.static)
            time.sleep(0.2)

    def _live_loop(self, addr):
        import zmq
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.connect(addr)
        print(f"[cloud] live from {addr}, <= 6 Hz", flush=True)
        last = 0.0
        while True:
            parts = sock.recv_multipart()
            while True:                                    # skip to the newest frame
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
            if time.time() - last < 1.0 / 6.0:
                continue
            last = time.time()
            try:
                xyz, rgb = cloud.frame_from_zmq(parts)
                xyz, rgb = cloud.voxel_down(xyz, rgb, max_points=self.max_points)
                self._put(cloud.pack_tag4(xyz, rgb))
            except Exception as e:                         # a bad frame must not stop the feed
                print(f"[cloud] frame skipped: {e}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--source", choices=("replay", "sim", "real"), default="replay",
                    help="replay: recorded episodes; sim: live control of unitree_sim_isaaclab (DDS domain 1, env 'tv'); "
                         "real: the REAL robot (DDS domain 0, see FIRST_ROBOT_TEST.md)")
    ap.add_argument("--i-understand-real-robot", action="store_true", help="--source real: required")
    ap.add_argument("--arm-mode", choices=("arm_sdk", "lowcmd"), default=None,
                    help="--source real (required): arm_sdk = rt/arm_sdk, robot's controller keeps the legs; "
                         "lowcmd = rt/lowcmd debug mode, ALL joints locked, robot must hang")
    ap.add_argument("--img-server-ip", default="192.168.123.164", help="--source real: teleimager on PC2")
    ap.add_argument("--dry-run", action="store_true", help="--source sim/real: read state, run IK and gates, publish nothing")
    ap.add_argument("--no-haptics", action="store_true", help="--source sim: no controller vibration")
    ap.add_argument("--haptic-gain", type=float, default=1.0, help="contact vibration strength 0..1 (default 1.0)")
    ap.add_argument("--haptic-limit-gain", type=float, default=0.5, help="joint-limit vibration strength (default 0.5)")
    ap.add_argument("--haptic-limit", type=float, nargs=2, default=(0.08, 0.4), metavar=("MARGIN_RAD", "MAX"),
                    help="joint-limit buzz: starts MARGIN_RAD from a limit, reaches MAX at the limit")
    ap.add_argument("--haptic-track", type=float, nargs=2, default=(0.15, 0.40), metavar=("ON", "FULL"),
                    help="tracking-error buzz (rad): starts at ON, full at FULL")
    ap.add_argument("--haptic-torque", type=float, nargs=2, default=(6.0, 15.0), metavar=("ON", "FULL"),
                    help="contact-torque buzz (N.m above the running baseline): starts at ON, full at FULL")
    ap.add_argument("--no-cloud", action="store_true", help="--source sim: do not read the sim head depth (point cloud)")
    ap.add_argument("--network-interface", default=None, help="DDS network interface for --source sim (default: auto)")
    ap.add_argument("--episodes-dir", default=os.environ.get("TELEOP_EPISODES_DIR", str(HOME / "VLA/data/teleop_sim/G1_Stack_Block_teleop")))
    ap.add_argument("--robot-dir", default=os.environ.get("TELEOP_ROBOT_DIR", str(HOME / "projects/unitree_ros/robots/g1_description")),
                    help="folder with g1_29dof_mode_15_with_dex1_1.urdf and meshes/")
    ap.add_argument("--gripper", choices=("dex1", "rubber_hand"), default="dex1",
                    help="end effector mounted on the robot (shown on the page; ?gripper= in the URL overrides it)")
    ap.add_argument("--image-hz", type=float, default=15.0, help="camera images sent per second")
    ap.add_argument("--cloud-file", help="point cloud .npz (tools/save_cloud_frame.py or tools/make_test_cloud.py), "
                    "resent about 5 times a second")
    ap.add_argument("--cloud-live", nargs="?", const="tcp://127.0.0.1:55558",
                    help="live point cloud from the sim head depth publisher (default tcp://127.0.0.1:55558)")
    ap.add_argument("--cloud-density", choices=tuple(cloud.DENSITY), default="normal",
                    help="max points per frame: normal 25k, dense 50k")
    ap.add_argument("--cert")
    ap.add_argument("--key")
    args = ap.parse_args()

    hcfg = {"gain": args.haptic_gain, "limit_gain": args.haptic_limit_gain, "limit": tuple(args.haptic_limit),
            "track": tuple(args.haptic_track), "torque": tuple(args.haptic_torque)}
    if args.source == "real":
        from real_source import RealSource, check_start
        check_start(args.network_interface, args.i_understand_real_robot, args.arm_mode, args.dry_run)
        source = RealSource(args.network_interface, args.arm_mode, img_host=args.img_server_ip, dry_run=args.dry_run,
                            haptics=not args.no_haptics, haptic_cfg=hcfg)
        if not args.cloud_file and not args.cloud_live and not args.no_cloud:
            args.cloud_live = f"tcp://{args.img_server_ip}:55558"     # robot_pc2/d435_depth_pub.py on PC2
    elif args.source == "sim":
        from sim_source import SimSource
        source = SimSource(args.network_interface, haptics=not args.no_haptics, live={"dry_run": args.dry_run},
                           haptic_cfg={"gain": args.haptic_gain, "limit_gain": args.haptic_limit_gain, "limit": tuple(args.haptic_limit),
                                       "track": tuple(args.haptic_track), "torque": tuple(args.haptic_torque)})
        if not args.cloud_file and not args.cloud_live and not args.no_cloud:
            args.cloud_live = "tcp://127.0.0.1:55558"           # live sim: point cloud on by default
    else:
        source = ReplaySource(args.episodes_dir)
    cloud_feed = CloudFeed(args) if (args.cloud_file or args.cloud_live) else None
    clients = set()
    last_xr_log = [0.0]
    grip_settings = {"left": None, "right": None}       # latest force / width per side from the headset

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0, compress=False)
        await ws.prepare(request)
        clients.add(ws)
        print(f"[ws] headset connected from {request.remote} ({len(clients)} open)", flush=True)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                if m.get("t") == "cmd":
                    source.command(m.get("cmd"))
                elif m.get("t") == "grip_settings" and m.get("side") in grip_settings:
                    new = {k: m.get(k) for k in ("gripper", "force", "width", "force_n", "width_mm")}
                    if new != grip_settings[m["side"]]:
                        grip_settings[m["side"]] = new
                        fn, wm = new.get("force_n"), new.get("width_mm")
                        print(f"[grip] {m['side']} {new.get('gripper')}: force {fn if fn is None else round(fn, 1)} N,"
                              f" width {wm if wm is None else round(wm, 1)} mm", flush=True)
                elif m.get("t") == "log":           # page-side errors / controller source changes (app.js pageLog)
                    print(f"[page] {str(m.get('msg', ''))[:500]}", flush=True)
                elif m.get("t") == "xr":
                    if hasattr(source, "on_xr"):
                        source.on_xr(m)                    # live sim: latest controller state for the control loop
                    if time.time() - last_xr_log[0] <= 1.0:
                        continue
                    last_xr_log[0] = time.time()
                    r, l = m.get("right", {}), m.get("left", {})
                    print(f"[xr] R trig {r.get('trigger', 0):.2f} grip {int(r.get('squeeze', 0))} A {int(r.get('clutch', 0))}"
                          f" | L trig {l.get('trigger', 0):.2f} grip {int(l.get('squeeze', 0))} X {int(l.get('clutch', 0))}",
                          flush=True)
        finally:
            clients.discard(ws)
            print(f"[ws] headset disconnected ({len(clients)} open)", flush=True)
        return ws

    async def index(_):
        return web.FileResponse(HERE / "web/index.html")

    robot_dir = Path(args.robot_dir).expanduser()

    async def robot_file(request):     # only the URDF and its meshes, nothing else from that folder
        name = request.match_info["name"]
        p = (robot_dir / name).resolve()
        if robot_dir.resolve() not in p.parents or p.suffix.lower() not in (".urdf", ".stl", ".dae") or not p.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    app = web.Application()

    async def no_cache(request, response):          # the Quest browser must never run an old cached app.js
        response.headers["Cache-Control"] = "no-store"
    app.on_response_prepare.append(no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/robot/{name:.+}", robot_file)
    app.router.add_static("/static/", HERE / "web")

    async def broadcast():
        period, img_every = 1.0 / 30.0, max(1, round(30.0 / args.image_hz))
        tick = 0
        while True:
            t0 = time.time()
            source.step()
            if clients:
                st = source.state()
                st.setdefault("source", args.source)
                st["cloud_expected"] = bool(args.cloud_live)
                st["gripper"] = args.gripper
                st = json.dumps(st)
                imgs = source.images() if tick % img_every == 0 else []
                pkt = cloud_feed.take() if cloud_feed else None      # packed in its own thread, never blocks here
                if pkt:
                    imgs.append(pkt)
                for ws in list(clients):
                    try:
                        await ws.send_str(st)
                        for b in imgs:
                            await ws.send_bytes(b)
                    except Exception:
                        clients.discard(ws)
            tick += 1
            await asyncio.sleep(max(0.0, period - (time.time() - t0)))

    cert, key = find_cert(args)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(cert, key)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", args.port, ssl_context=ctx).start()
    print(f"[server] open https://<this PC's IP>:{args.port} in the Quest 3 browser (cert {cert})", flush=True)
    await broadcast()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
