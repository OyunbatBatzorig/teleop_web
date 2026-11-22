# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0     
"""
Scene for Isaac-Stack-RgyBlock-Real-G129-Dex1-Joint: a copy of base_scene_stack_rgyblock.py made to look like
the real G1_Stack_Block training data (episode 1). Unitree's original scene is not changed.
Block and mat positions come from the first frame of episode 1: each object's pixel in the real head-camera image,
back-projected onto the table with a model of the real head camera (Unitree's head camera, 94 deg lens, tilted 8.6 deg
down / 8.7 deg left; it reproduces sim renders within ~1 px). The robot is leaned 7.4 deg forward and its links have no
gravity so the arms track the recording to ~1 deg (see the task config).
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  
import os
project_root = os.environ.get("PROJECT_ROOT")
import math
# Gap between the robot's pelvis and the centre of the near table edge (metres), and the table's yaw. Unitree: 0.119, 0.
# Measured 2026-10-11 with the fitted head camera: near-edge rows in episode 1 frame 0 (columns 180/340/500/580:
# 314/311/309/307 px) back-projected onto the table top give the edge 0.20 m from the pelvis on the robot's left and
# 0.22 m on its right: gap 0.22 at the edge centre (x = -4.3), yaw +2.1 deg (CCW from above). The earlier 0.24 / 0 deg
# left the sim edge 8-20 px higher in the image than the real one.
# The robot stands at y = -3.7 facing -y. Override with G1_REAL_TABLE_GAP / G1_REAL_TABLE_YAW_DEG.
TABLE_GAP = float(os.environ.get("G1_REAL_TABLE_GAP", "0.22"))
TABLE_YAW = math.radians(float(os.environ.get("G1_REAL_TABLE_YAW_DEG", "2.1")))
# Table size. The asset is 2.44 m long (x) and 0.762 m deep (y), spanning x -5.52..-3.08 at the original placement.
# In the real head image (episode 1 frame 0, fitted camera, 2026-10-11) the far edge is beyond the top of the picture
# (> 0.87 m deep) and the table's left end is at x = -3.6 (0.6 m to the robot's left; another table and the floor
# show beyond it). G1_REAL_TABLE_SIZE=real: 1.2 m deep, left end at -3.6, right end kept at -5.52 (out of view); the
# wood texture and the frame are stretched with it, which looked wrong (2026-10-16), and it did not change the model's
# answers. Default: the original 2.44 x 0.762 m table.
if os.environ.get("G1_REAL_TABLE_SIZE", "asset") != "real":
    TABLE_LEFT, TABLE_DEPTH = -3.08, 0.762
else:
    TABLE_LEFT, TABLE_DEPTH = float(os.environ.get("G1_REAL_TABLE_LEFT", "-3.6")), float(os.environ.get("G1_REAL_TABLE_DEPTH", "1.2"))
TABLE_LEN = TABLE_LEFT - (-5.52)
TABLE_SCALE = (TABLE_LEN / 2.44, TABLE_DEPTH / 0.762, 1.0)
# the near edge passes through (-4.3, -3.7 - TABLE_GAP) at angle TABLE_YAW; the table turns about its vertical axis
TABLE_CX = (TABLE_LEFT - 5.52) / 2                                      # near-edge centre x
TABLE_EY = -3.7 - TABLE_GAP + (TABLE_CX + 4.3) * math.tan(TABLE_YAW)    # near-edge centre y
TABLE_X = TABLE_CX + TABLE_DEPTH / 2 * math.sin(TABLE_YAW)
TABLE_Y = TABLE_EY - TABLE_DEPTH / 2 * math.cos(TABLE_YAW)
TABLE_ROT = [math.cos(TABLE_YAW / 2), 0.0, 0.0, math.sin(TABLE_YAW / 2)]
# Yellow block. In episode 1 the human's left hand pushed the yellow block ~3.5 cm to the robot's right (frames ~200-250,
# after placing green), so it grasped yellow (frame 308-310) away from its first-frame place, and a sim replay of the
# recorded hand misses it. G1_REAL_YELLOW=grasp puts yellow where the human's jaws closed on it (recorded joints, sim
# Dex1 finger geometry: jaw centre (-4.140, -4.051), jaw yaw +25.5 deg) for oracle/replay tests. Unset: frame-0 layout.
# Block layout. Default ("image"): episode 1 frame 0 as seen by the fitted head camera (cube fits, frames 0 and 15).
# G1_REAL_LAYOUT=kinematic: where the human's jaws closed on each block in episode 1, from the recorded joints and the
# sim Dex1 finger geometry (jaw centre when the measured gripper stops on the block; yaw = jaw yaw). In the recording
# these sit 1.5-2.5 cm from the image positions (red -1.8/-1.7 cm, green -0.4/-1.5 cm in x/y), and the model learned
# hand targets in that kinematic frame: with the image layout its fingers land on the blocks (2026-10-11 live runs).
# Yellow was pushed before it was grasped, so it gets green's image->kinematic offset and green's yaw offset (-3 deg).
if os.environ.get("G1_REAL_LAYOUT") == "kinematic":
    RED_POS, RED_ROT = [-4.382, -4.094, 0.816], [0.99692, 0, 0, -0.07846]          # yaw -9 deg (human jaw -8.7)
    GREEN_POS, GREEN_ROT = [-4.161, -4.119, 0.816], [0.9763, 0, 0, 0.21644]      # yaw +25 deg (human jaw +24.8)
    YELLOW_POS, YELLOW_ROT = [-4.105, -4.064, 0.816], [0.97237, 0, 0, 0.23345]    # yaw +27 deg
else:
    RED_POS, RED_ROT = [-4.364, -4.077, 0.816], [0.99692, 0, 0, 0.07846]       # yaw +9 deg, uncertain by ~4 deg
    GREEN_POS, GREEN_ROT = [-4.157, -4.104, 0.816], [0.97030, 0, 0, 0.24192]   # yaw +28 deg, uncertain by ~4 deg
    YELLOW_POS, YELLOW_ROT = [-4.101, -4.049, 0.816], [0.96593, 0, 0, 0.25882]   # yaw +30 deg; the weakest fit (yellow on pine)
if os.environ.get("G1_REAL_YELLOW") == "grasp":
    YELLOW_POS, YELLOW_ROT = [-4.140, -4.051, 0.816], [0.97534, 0, 0, 0.22069]   # yaw +25.5 deg
# 2026-11-22: which chess piece (converted CC-BY "Low Poly Chess Set", JustinARay/opengameart.org) goes in each of
# the 3 block slots. Override with G1_RED_PIECE / G1_YELLOW_PIECE / G1_GREEN_PIECE = pawn|queen|king|bishop|knight|rook.
_CHESS_PIECE_USD = {
    "pawn": "chess_pawn_rigid/Pawn.usd", "queen": "chess_queen_rigid/Queen.usd",
    "king": "chess_king_rigid/King.usd", "bishop": "chess_bishop_rigid/Bishop.usd",
    "knight": "chess_knight_rigid/Knight.usd", "rook": "chess_rook_rigid/Rook.usd",
}
RED_PIECE = _CHESS_PIECE_USD[os.environ.get("G1_RED_PIECE", "king").lower()]
YELLOW_PIECE = _CHESS_PIECE_USD[os.environ.get("G1_YELLOW_PIECE", "queen").lower()]
GREEN_PIECE = _CHESS_PIECE_USD[os.environ.get("G1_GREEN_PIECE", "bishop").lower()]


def _mul_quat(q1, q2):
    """Hamilton product (w,x,y,z): rotate by q2 first, then by q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# 2026-11-22, found from a screenshot: the chess set's OBJ meshes are Y-up (Blender's OBJ-export default) and
# convert_mesh.py does not reorient them, so the board spawned standing on its edge and a piece spawned lying on
# its side. Fix: +90 deg about local X (Y-up -> this world's Z-up), applied BEFORE each object's own yaw so the
# existing (block-calibrated) yaw still turns the now-upright piece about world Z. If this comes out upside down
# or backwards instead, flip the sign: (0.70710678, -0.70710678, 0.0, 0.0).
_AXIS_FIX_Y_UP_TO_Z_UP = (0.70710678, 0.70710678, 0.0, 0.0)
RED_ROT = _mul_quat(RED_ROT, _AXIS_FIX_Y_UP_TO_Z_UP)
YELLOW_ROT = _mul_quat(YELLOW_ROT, _AXIS_FIX_Y_UP_TO_Z_UP)
GREEN_ROT = _mul_quat(GREEN_ROT, _AXIS_FIX_Y_UP_TO_Z_UP)

# 2026-11-22: the textured board mesh rendered as a plain grey slab (its wood/checker texture didn't carry over
# in conversion) and sat close enough to the robot to visually overlap it. Replaced with a procedural 8x8 grid of
# flat coloured squares instead (below) — no texture needed, and each square's exact world (x, y) is known, which
# answers "which point do I move from/to". Also pushes the whole board+pieces forward (away from the robot).
# 2026-11-22 (g): board placed for REACH and made SOLID (the user could not reach the queen / king, and pieces fell
# through the board: the squares were visual-only cuboids floating 1.6 cm above the table).
#   * Robot = White: rank 1 nearest the robot, a-file on the robot's LEFT (world +x), so the robot faces -y down the files.
#   * Board centre 0.38 m in front of the pelvis, centred left / right: the closest it can be with its near edge 1 cm
#     inside the table's near edge (TABLE_GAP 0.22 m). Reach check (teleop_web/tools/chess_reach.py + chess_place.py,
#     same G1_29_ArmIK model and waist as this task, fingertip pad 11 cm ahead of the IK wrist frame, joints >= 0.15 rad
#     from their limits, grasp 3 cm above the board, approach +10 cm, +-4 cm margin): 3.5 cm squares -> 54/64 squares
#     with margin, 62/64 reachable (d8 / e8, the far centre, are not); 3.0 cm (G1_SQUARE_SIZE=0.030) -> 61/64 with
#     margin, 64/64 reachable; 2.5 cm -> 64/64 with margin. Default board centre per square size below (G1_BOARD_X).
#   * Solid board: one static box collider (ChessBoardBase) resting on the table top (0.794 m), 7.5 mm thick; the 64
#     coloured squares are thin visual tiles on its top. Pieces (origin at their base) stand on the board top + 1 mm.
TABLE_TOP_Z = 0.794                                   # PackingTable top (asset 0.9941 m high, spawned at z -0.2)
SQUARE_SIZE = float(os.environ.get("G1_SQUARE_SIZE", "0.035"))  # 3.5 cm default (was 5.7cm); whole board = 8x this
# piece size follows the square size (king base 4.4 cm x scale: 2.7 cm at 0.6, 2.4 at 0.55, 2.2 at 0.5)
_PIECE_SCALE_DEFAULT = {0.035: 0.6, 0.030: 0.55, 0.025: 0.5}.get(round(SQUARE_SIZE, 3), 0.6 * SQUARE_SIZE / 0.035)
PIECE_SCALE = float(os.environ.get("G1_PIECE_SCALE", str(_PIECE_SCALE_DEFAULT)))
_BOARD_X_DEFAULT = {0.035: 0.38, 0.030: 0.36, 0.025: 0.35}.get(round(SQUARE_SIZE, 3), max(0.38, 0.24 + 4 * SQUARE_SIZE))
BOARD_X_PELVIS = float(os.environ.get("G1_BOARD_X", str(_BOARD_X_DEFAULT)))   # m in front of the pelvis
BOARD_Y_PELVIS = float(os.environ.get("G1_BOARD_Y", "0.0"))                   # m to the robot's left
ROBOT_XY = (-4.2, -3.7)                               # robot root, facing -y (robot left = world +x)
BOARD_CENTER_X = ROBOT_XY[0] + BOARD_Y_PELVIS
BOARD_CENTER_Y = ROBOT_XY[1] - BOARD_X_PELVIS
BOARD_T, TILE_T, BOARD_BORDER = 0.0075, 0.002, 0.012
BOARD_TOP_Z = TABLE_TOP_Z + BOARD_T                   # collider top: pieces stand here
BOARD_BASE_Z = TABLE_TOP_Z + BOARD_T / 2              # collider centre
BOARD_Z = BOARD_TOP_Z - TILE_T / 2 + 0.0002           # visual tiles: centre (top 0.2 mm above the collider)
SQUARE_FILES = "abcdefgh"
SQUARE_RANKS = "12345678"
SQUARE_CENTERS = {}  # "e4" -> (x, y), world frame, for reference when picking/placing
for _fi, _f in enumerate(SQUARE_FILES):
    for _ri, _r in enumerate(SQUARE_RANKS):
        SQUARE_CENTERS[_f + _r] = (
            round(BOARD_CENTER_X - (_fi - 3.5) * SQUARE_SIZE, 4),    # a-file on the robot's left (+x)
            round(BOARD_CENTER_Y - (_ri - 3.5) * SQUARE_SIZE, 4),    # rank 1 nearest the robot (largest y)
        )
# pieces on squares (slot -> square; the piece type per slot is G1_RED_PIECE / G1_YELLOW_PIECE / G1_GREEN_PIECE)
RED_SQUARE = os.environ.get("G1_RED_SQUARE", "e1")       # king
YELLOW_SQUARE = os.environ.get("G1_YELLOW_SQUARE", "d1")  # queen
GREEN_SQUARE = os.environ.get("G1_GREEN_SQUARE", "c1")    # bishop
_PIECE_Z = BOARD_TOP_Z + 0.001
RED_POS = [*SQUARE_CENTERS[RED_SQUARE], _PIECE_Z]
YELLOW_POS = [*SQUARE_CENTERS[YELLOW_SQUARE], _PIECE_Z]
GREEN_POS = [*SQUARE_CENTERS[GREEN_SQUARE], _PIECE_Z]
RED_ROT = YELLOW_ROT = GREEN_ROT = _AXIS_FIX_Y_UP_TO_Z_UP   # upright, no yaw
print(f"[chess_board] centre ({BOARD_CENTER_X:.3f}, {BOARD_CENTER_Y:.3f}) = {BOARD_X_PELVIS:.2f} m ahead of the pelvis, "
      f"square {SQUARE_SIZE * 100:.1f} cm, board top z {BOARD_TOP_Z:.4f} (table {TABLE_TOP_Z}); pieces: "
      f"red slot {RED_SQUARE} {RED_POS}, yellow slot {YELLOW_SQUARE} {YELLOW_POS}, green slot {GREEN_SQUARE} {GREEN_POS}",
      flush=True)
# print once at import so the coordinates are visible in the sim_main.py console log
print("[chess_board] square -> (x, y) world coords:", flush=True)
for _r in reversed(SQUARE_RANKS):
    print("  " + " ".join(f"{_f}{_r}=({SQUARE_CENTERS[_f + _r][0]:.3f},{SQUARE_CENTERS[_f + _r][1]:.3f})" for _f in SQUARE_FILES), flush=True)


@configclass
class TableRedGreenYellowBlockRealSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",  # use simple room model
        ),
    )


    # 1. table configuration
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TABLE_X, TABLE_Y, -0.2],   # top at 0.794; near edge TABLE_GAP in front of the pelvis
                                                rot=TABLE_ROT), # yaw TABLE_YAW about world z (w, x, y, z)
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_no_yellowbox.usda",    # real-looking table: pine top from the training images, black frame, no yellow outline
            scale=TABLE_SCALE,    # length/depth to match the real table (see "Table size" above); height unchanged
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )
    # Object
    red_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Red_block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=RED_POS,  # see "Block layout" above
            rot=RED_ROT  # yaw about world z (w, x, y, z)
        ),
        spawn=UsdFileCfg(
            # 2026-11-22: swapped the 3.6cm cuboid for a chess piece (G1_RED_PIECE, default king) for the
            # chess-orchestration thesis. Position/rotation above is unchanged.
            usd_path=f"{project_root}/assets/objects/{RED_PIECE}",
            scale=(PIECE_SCALE, PIECE_SCALE, PIECE_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                # a light block squeezed by stiff fingers is otherwise shot out of the grasp by the overlap correction
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
        ),
    )

    yellow_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Yellow_block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=YELLOW_POS,  # see "Block layout" above and G1_REAL_YELLOW
            rot=YELLOW_ROT  # yaw about world z (w, x, y, z)
        ),
        spawn=UsdFileCfg(
            # 2026-11-22: swapped the 3.6cm cuboid for a chess piece (G1_YELLOW_PIECE, default queen).
            usd_path=f"{project_root}/assets/objects/{YELLOW_PIECE}",
            scale=(PIECE_SCALE, PIECE_SCALE, PIECE_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                # a light block squeezed by stiff fingers is otherwise shot out of the grasp by the overlap correction
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
        ),
    )
    # 2026-11-22: 8x8 board as flat coloured squares (not a textured mesh -- see note above). Each square
    # is a named scene entity "sq_<square>" at its exact world (x,y); SQUARE_CENTERS has the same values.
    # solid board (static collider, no rigid body) resting on the table; the squares below are visual tiles on it
    chess_board_base = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ChessBoardBase",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[BOARD_CENTER_X, BOARD_CENTER_Y, BOARD_BASE_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(8 * SQUARE_SIZE + 2 * BOARD_BORDER, 8 * SQUARE_SIZE + 2 * BOARD_BORDER, BOARD_T),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.22, 0.12), roughness=0.7, metallic=0)),
    )
    sq_a1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a1"][0], SQUARE_CENTERS["a1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_a2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a2"][0], SQUARE_CENTERS["a2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_a3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a3"][0], SQUARE_CENTERS["a3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_a4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a4"][0], SQUARE_CENTERS["a4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_a5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a5"][0], SQUARE_CENTERS["a5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_a6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a6"][0], SQUARE_CENTERS["a6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_a7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a7"][0], SQUARE_CENTERS["a7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_a8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_a8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["a8"][0], SQUARE_CENTERS["a8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_b1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b1"][0], SQUARE_CENTERS["b1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_b2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b2"][0], SQUARE_CENTERS["b2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_b3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b3"][0], SQUARE_CENTERS["b3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_b4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b4"][0], SQUARE_CENTERS["b4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_b5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b5"][0], SQUARE_CENTERS["b5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_b6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b6"][0], SQUARE_CENTERS["b6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_b7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b7"][0], SQUARE_CENTERS["b7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_b8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_b8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["b8"][0], SQUARE_CENTERS["b8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_c1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c1"][0], SQUARE_CENTERS["c1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_c2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c2"][0], SQUARE_CENTERS["c2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_c3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c3"][0], SQUARE_CENTERS["c3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_c4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c4"][0], SQUARE_CENTERS["c4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_c5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c5"][0], SQUARE_CENTERS["c5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_c6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c6"][0], SQUARE_CENTERS["c6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_c7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c7"][0], SQUARE_CENTERS["c7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_c8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_c8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["c8"][0], SQUARE_CENTERS["c8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_d1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d1"][0], SQUARE_CENTERS["d1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_d2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d2"][0], SQUARE_CENTERS["d2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_d3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d3"][0], SQUARE_CENTERS["d3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_d4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d4"][0], SQUARE_CENTERS["d4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_d5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d5"][0], SQUARE_CENTERS["d5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_d6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d6"][0], SQUARE_CENTERS["d6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_d7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d7"][0], SQUARE_CENTERS["d7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_d8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_d8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["d8"][0], SQUARE_CENTERS["d8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_e1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e1"][0], SQUARE_CENTERS["e1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_e2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e2"][0], SQUARE_CENTERS["e2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_e3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e3"][0], SQUARE_CENTERS["e3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_e4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e4"][0], SQUARE_CENTERS["e4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_e5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e5"][0], SQUARE_CENTERS["e5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_e6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e6"][0], SQUARE_CENTERS["e6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_e7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e7"][0], SQUARE_CENTERS["e7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_e8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_e8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["e8"][0], SQUARE_CENTERS["e8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_f1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f1"][0], SQUARE_CENTERS["f1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_f2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f2"][0], SQUARE_CENTERS["f2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_f3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f3"][0], SQUARE_CENTERS["f3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_f4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f4"][0], SQUARE_CENTERS["f4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_f5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f5"][0], SQUARE_CENTERS["f5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_f6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f6"][0], SQUARE_CENTERS["f6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_f7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f7"][0], SQUARE_CENTERS["f7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_f8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_f8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["f8"][0], SQUARE_CENTERS["f8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_g1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g1"][0], SQUARE_CENTERS["g1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_g2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g2"][0], SQUARE_CENTERS["g2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_g3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g3"][0], SQUARE_CENTERS["g3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_g4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g4"][0], SQUARE_CENTERS["g4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_g5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g5"][0], SQUARE_CENTERS["g5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_g6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g6"][0], SQUARE_CENTERS["g6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_g7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g7"][0], SQUARE_CENTERS["g7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_g8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_g8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["g8"][0], SQUARE_CENTERS["g8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_h1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h1"][0], SQUARE_CENTERS["h1"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_h2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h2"][0], SQUARE_CENTERS["h2"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_h3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h3"][0], SQUARE_CENTERS["h3"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_h4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h4"][0], SQUARE_CENTERS["h4"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_h5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h5"][0], SQUARE_CENTERS["h5"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_h6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h6"][0], SQUARE_CENTERS["h6"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    sq_h7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h7",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h7"][0], SQUARE_CENTERS["h7"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.88, 0.78), roughness=0.6, metallic=0)),
    )
    sq_h8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Square_h8",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[SQUARE_CENTERS["h8"][0], SQUARE_CENTERS["h8"][1], BOARD_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(size=(SQUARE_SIZE * 0.96, SQUARE_SIZE * 0.96, TILE_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.08, 0.05), roughness=0.6, metallic=0)),
    )
    # black target mat: 4.7 cm square (fitted through the real head camera, episode 1 frames 0 and 15) where the stack
    # was built in episode 1 (episode 0 within 1.3 cm),
    # 35 cm in front of the pelvis, 5 cm to the robot's right. Visual only, no collision.
    black_mat = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Black_mat",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.234, -4.062, 0.7945], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.047, 0.047, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03), roughness=0.9, metallic=0),
        ),
    )

    green_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Green_block",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GREEN_POS,  # see "Block layout" above
            rot=GREEN_ROT  # yaw about world z (w, x, y, z)
        ),
        spawn=UsdFileCfg(
            # 2026-11-22: swapped the 3.6cm cuboid for a chess piece (G1_GREEN_PIECE, default bishop).
            usd_path=f"{project_root}/assets/objects/{GREEN_PIECE}",
            scale=(PIECE_SCALE, PIECE_SCALE, PIECE_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                # a light block squeezed by stiff fingers is otherwise shot out of the grasp by the overlap correction
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
        ),
    )
    # Ground plane
    # 3. ground configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",    # ground in the scene
    #     spawn=GroundPlaneCfg( ),    # ground configuration
    # )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=500.0),    # light intensity
    )






    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-4.1, -4.9, 1.8),
                                                    rot_offset=( -0.3173,0.94833, 0.0, 0.0))


