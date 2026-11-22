"""Offline tests for --source real (2026-11-22): startup gates, real settings, SimSource refusing domain 0, and that
--dry-run publishes NOTHING (checked on the SIM domain 1 with a fake rt/lowstate on loopback; domain 0 is never touched).
Run: ~/miniconda3/envs/tv/bin/python tools/test_real_gates.py   (stop the sim first: it would also publish rt/lowstate on domain 1)"""
import os, sys, time, pathlib, threading
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from real_source import check_start, real_live_cfg, CONFIRM_TEXT
import sim_source
from sim_source import SimSource

fails = []
def expect_exit(name, fn):
    try:
        fn(); fails.append(f"{name}: did not refuse")
    except SystemExit:
        pass
def ok(name, cond):
    if not cond: fails.append(name)

# 1) startup gates
no_ask = lambda *_: (_ for _ in ()).throw(AssertionError("asked"))
expect_exit("no --i-understand", lambda: check_start("eth0", False, "arm_sdk", False, ask=no_ask))
expect_exit("no interface", lambda: check_start(None, True, "arm_sdk", False, ask=no_ask))
expect_exit("no arm mode", lambda: check_start("eth0", True, None, False, ask=no_ask))
expect_exit("bad arm mode", lambda: check_start("eth0", True, "walk", False, ask=no_ask))
expect_exit("wrong confirmation", lambda: check_start("eth0", True, "arm_sdk", False, ask=lambda *_: "yes"))
expect_exit("empty confirmation", lambda: check_start("eth0", True, "arm_sdk", False, ask=lambda *_: ""))
check_start("eth0", True, "arm_sdk", False, ask=lambda *_: CONFIRM_TEXT)       # passes
check_start("eth0", True, "lowcmd", True, ask=no_ask)                           # dry run: no question

# 2) real settings
c = real_live_cfg("arm_sdk", "192.168.123.164", False)
ok("domain 0", c["domain"] == 0)
ok("arm_sdk -> motion_mode", c["motion_mode"] is True and real_live_cfg("lowcmd", "x", False)["motion_mode"] is False)
ok("real is not simulation_mode", c["simulation_mode"] is False)
ok("slower than sim", c["track_vel"] <= np.deg2rad(90) + 1e-9 and c["track_acc"] <= np.deg2rad(600) + 1e-9)
ok("no reset on real", c["allow_reset"] is False)
ok("gripper step clamp", c["grip_step"] == 0.18)
expect_exit("real cfg needs arm mode", lambda: real_live_cfg(None, "x", False))

# 3) SimSource never runs on another domain unless RealSource asks
expect_exit("SimSource domain 0", lambda: SimSource(live={"domain": 0}))
ok("sim default domain 1", sim_source.SIM_LIVE["domain"] == 1)

threading.Timer(90, lambda: (print("FAIL timeout"), os._exit(2))).start()
# 4) dry run sends nothing (domain 1, loopback, fake robot state)
if "--no-dds" not in sys.argv:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    import subprocess
    FEED = ("import time\nfrom unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher\n"
            "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_\n"
            "from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_\n"
            "ChannelFactoryInitialize(1)\np = ChannelPublisher('rt/lowstate', LowState_); p.Init()\n"
            "m = unitree_hg_msg_dds__LowState_()\nt = time.time()\n"
            "while time.time() - t < 80:\n    p.Write(m); time.sleep(0.005)\n")
    feeder = subprocess.Popen([sys.executable, "-c", FEED])          # fake robot state, own process, domain 1
    ChannelFactoryInitialize(1)          # sim domain 1, default interface
    seen = {"lowcmd": 0, "dex1": 0}
    s1 = ChannelSubscriber("rt/lowcmd", LowCmd_); s1.Init(lambda m: seen.__setitem__("lowcmd", seen["lowcmd"] + 1), 10)
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    s2 = ChannelSubscriber("rt/dex1/right/cmd", MotorCmds_); s2.Init(lambda m: seen.__setitem__("dex1", seen["dex1"] + 1), 10)
    stop = threading.Event()
    sim_source.ChannelFactoryInitialize = lambda *a, **k: None          # already initialised above
    import unitree_sdk2py.core.channel as ch
    _init = ch.ChannelFactoryInitialize; ch.ChannelFactoryInitialize = lambda *a, **k: None
    try:
        src = SimSource(None, haptics=True, live={"dry_run": True})
        src.on_xr({"t": "xr", "right": {"ee": np.eye(4).ravel().tolist(), "trigger": 1.0, "clutch": True, "gripper": "dex1",
                                        "width_phys": 83.0}})
        time.sleep(1.5)
    finally:
        ch.ChannelFactoryInitialize = _init
        feeder.kill()
    ok(f"dry run: nothing on rt/lowcmd (got {seen['lowcmd']})", seen["lowcmd"] == 0)
    ok(f"dry run: nothing on rt/dex1/right/cmd (got {seen['dex1']})", seen["dex1"] == 0)
    ok("dry run: the loop did run (null writes counted)", sim_source._NullPublisher.sent > 10)
    print(f"dry run: null writes {sim_source._NullPublisher.sent}, real lowcmd {seen['lowcmd']}, real dex1 {seen['dex1']}")

for f in fails:
    print("FAIL", f)
print("PASS" if not fails else "FAIL")
sys.stdout.flush(); os._exit(1 if fails else 0)
