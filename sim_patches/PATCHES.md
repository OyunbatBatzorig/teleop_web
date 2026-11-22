# Changes to unitree_sim_isaaclab used by the live sim mode

Base commit: unitreerobotics/unitree_sim_isaaclab `e30c25b`.

| What | Where |
|---|---|
| Edits to tracked files (wrist / head cameras, gripper target kept between DDS reads, Dex1 log, task registration, ...) | `unitree_sim_isaaclab_tracked.diff`, apply with `git apply` |
| New "Real" / Chess task, task ids `Isaac-Stack-RgyBlock-Real-G129-Dex1-Joint` and `Isaac-Chess-Pieces-G129-Dex1-Joint-Factory` | `files/tasks/g1_tasks/stack_rgyblock_real_g1_29dof_dex1/` |
| Its scene (table, solid chess board, pieces) and reward | `files/tasks/common_scene/base_scene_stack_rgyblock_real.py`, `files/tasks/common_rewards/base_reward_stack_rgyblock_real.py` |
| Head depth + colour publisher for the point cloud (zmq 55558) | `files/tools/head_depth_pub.py` |
| Table look and chess pieces | `files/assets/objects/` |

The table look `table_no_yellowbox.usda` references Unitree's `table_with_yellowbox.usd`, which comes with the sim's
assets download. The chess pieces are converted from the CC-BY "Low Poly Chess Set" (JustinARay, opengameart.org);
keep that attribution.

Install:
```bash
cd unitree_sim_isaaclab
git apply <this repo>/sim_patches/unitree_sim_isaaclab_tracked.diff
cp -r <this repo>/sim_patches/files/* .
```

## xr_teleoperate (`../xr_patches/`)
Base commit: `817fb00c`.
- `joint_limiter.py` goes to `teleop/utils/`. The server needs it.
- `robot_hand_unitree.diff` is optional: the sim-only Dex1 gripper clamp fix.
