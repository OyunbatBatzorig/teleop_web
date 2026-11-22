# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import copy
import os
import tempfile
# Block randomisation on reset (metres, +- around the episode 1 layout), for generating sim demonstrations.
# Default 0: always the episode 1 layout. Example: G1_REAL_BLOCK_RANGE=0.02 -> each block within +-2 cm in x and y.
BLOCK_RANGE = float(os.environ.get("G1_REAL_BLOCK_RANGE", "0"))
import torch
from dataclasses import MISSING



import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets, CameraBaseCfg  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager, BatchObjectEvent, MultiObjectEvent

# import public scene configuration
from tasks.common_scene.base_scene_stack_rgyblock_real import TableRedGreenYellowBlockRealSceneCfg

# Robot (fixed base) height; the table stays put. Unitree: 0.76. Chosen so that, replaying episode 1 (arms tracking
# to ~1 deg), the fingertips reach about the blocks' mid-height (1.7 cm above the table) at the three grasp moments.
# Override with the environment variable G1_REAL_ROBOT_Z.
ROBOT_Z = float(os.environ.get("G1_REAL_ROBOT_Z", "0.775"))

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(TableRedGreenYellowBlockRealSceneCfg):
    """object table scene configuration class
    
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix(init_pos=(-4.2, -3.7, ROBOT_Z),
        init_rot=(0.7071, 0, 0, -0.7071))


    # 6. add camera configuration 
    # Head camera. Default: Unitree's wide camera (106 deg, shows the whole table).
    # G1_REAL_HEAD_CAM=fitted: the real head camera as fitted to G1_Stack_Block episode 1 (left eye of the stereo
    # camera, ~94 deg, looking 8.6 deg further down and 8.7 deg further left; reproduces the real image within ~1 px).
    # G1_REAL_HEAD_DEPTH=1 (with the fitted camera): also render depth for the operator's point cloud
    # (tools/head_depth_pub.py; 2026-10-27). Recordings and the image server still get RGB only.
    front_camera = (CameraBaseCfg.get_camera_config(focal_length=9.29, rot_offset=(0.50044, -0.56951, 0.49956, -0.41911),
                                                    data_types=(["rgb", "distance_to_image_plane"]
                                                                if os.environ.get("G1_REAL_HEAD_DEPTH") == "1" else ["rgb"]))
                    if os.environ.get("G1_REAL_HEAD_CAM") == "fitted"
                    # default head = the real D435i (camera_configs); G1_REAL_HEAD_DEPTH=1 renders its depth too (2026-11-22)
                    else CameraPresets.g1_front_camera(data_types=(["rgb", "distance_to_image_plane"]
                                                                   if os.environ.get("G1_REAL_HEAD_DEPTH") == "1" else None)))
    # Wrist cameras. Default: Unitree's (80 deg, 13.5 cm behind the finger plane, near clip 0.1 m).
    # G1_REAL_WRIST_CAM=fitted: the real Dex1 wrist cameras as fitted to G1_Stack_Block episode 1 (2026-10-11,
    # prestep3_tools/wrist_fit.py: fingertip pixels and block/mat centroids in the real wrist frames, hand pose from the
    # recorded joints; rms 9 px left, 16 px right vs 53/38 px for the default). Each ~6.5 cm from the finger plane,
    # 3-4 cm further along the fingers and tilted 7-15 deg further down; near clip 0.01 m so the fingertips and a
    # block between them (7-9 cm away) are drawn. Pinhole only: the real wide-angle lens distortion is not modelled.
    if os.environ.get("G1_REAL_WRIST_CAM") == "fitted":
        left_wrist_camera = CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/left_hand_base_link/left_wrist_camera",
            focal_length=10.46, horizontal_aperture=20.0, clipping_range=(0.01, 1.0e5),       # 335 px, 87 deg
            pos_offset=(-0.0087, 0.079, 0.0677), rot_offset=(-0.39367, 0.91853, -0.02148, -0.02931))
        right_wrist_camera = CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/right_hand_base_link/right_wrist_camera",
            focal_length=12.06, horizontal_aperture=20.0, clipping_range=(0.01, 1.0e5),       # 386 px, 79 deg
            pos_offset=(0.0015, 0.059, 0.0642), rot_offset=(-0.46165, 0.88651, 0.01652, -0.02663))
    else:
        left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
        right_wrist_camera = CameraPresets.right_gripper_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    """defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """
        # 1. robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        # 2. gripper joint state observation 
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)

        # 3. camera image observation
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    # check if the object is out of the working range
    success = DoneTerm(func=mdp.reset_object_estimate)# use task completion check function

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward,weight=1.0)

GRIPPER_LINK_KEYS = ("dex1", "finger", "gripper", "hand", "wrist_yaw")   # name parts of each side's gripper links


def filter_gripper_self_collisions(env, env_ids, robot_name: str = "Robot"):
    """With whole-robot self-collision on, stop each Dex1 from colliding with itself (2026-10-27): the two fingers,
    the gripper base, the hand/rubber links and the wrist-yaw link are mutually filtered (PhysX FilteredPairs), so
    the jaws close fully. Arm-vs-body and arm-vs-arm contacts stay on. Runs before the simulation starts."""
    from pxr import UsdPhysics
    stage = env.sim.stage
    for env_root in stage.GetPrimAtPath("/World/envs").GetChildren():
        robot = stage.GetPrimAtPath(f"{env_root.GetPath()}/{robot_name}")
        if not robot.IsValid():
            continue
        for side in ("left", "right"):
            links = [p for p in robot.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                     and p.GetName().lower().startswith(side) and any(k in p.GetName().lower() for k in GRIPPER_LINK_KEYS)]
            for link in links:
                rel = UsdPhysics.FilteredPairsAPI.Apply(link).CreateFilteredPairsRel()
                for other in links:
                    if other != link:
                        rel.AddTarget(other.GetPath())
            print(f"[filter_gripper_self_collisions] {side}: {[l.GetName() for l in links]}")


@configclass
class EventCfg:
    # only with whole-robot self-collision (G1_REAL_SELF_COLLISION != 0, see __post_init__)
    filter_gripper_collisions = (EventTermCfg(func=filter_gripper_self_collisions, mode="prestartup")
                                 if os.environ.get("G1_REAL_SELF_COLLISION", "1") != "0" else None)
    reset_red_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
        mode="reset",   # set event mode to reset
        params={
            # position range parameter
            "pose_range": {
                "x": [-BLOCK_RANGE, BLOCK_RANGE],  # default 0: keep the G1_Stack_Block episode 1 layout
                "y": [-BLOCK_RANGE, BLOCK_RANGE],
            },
            # speed range parameter (empty dictionary means using default value)
            "velocity_range": {},
            # specify the object to reset
            "asset_cfg": SceneEntityCfg("red_block"),
        },
    )
    reset_yellow_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
        mode="reset",   # set event mode to reset
        params={
            "pose_range": {
                "x": [-BLOCK_RANGE, BLOCK_RANGE],  # default 0: keep the G1_Stack_Block episode 1 layout
                "y": [-BLOCK_RANGE, BLOCK_RANGE],
            },
            # speed range parameter (empty dictionary means using default value)
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("yellow_block"),
        },
    )   
    reset_green_block = EventTermCfg(
        func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
        mode="reset",   # set event mode to reset
        params={
            "pose_range": {
                "x": [-BLOCK_RANGE, BLOCK_RANGE],  # default 0: keep the G1_Stack_Block episode 1 layout
                "y": [-BLOCK_RANGE, BLOCK_RANGE],
            },
            # speed range parameter (empty dictionary means using default value)
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("green_block"),
        },
    )   

@configclass
class StackRgyBlockRealG129DEX1BaseFixEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree G1 robot stack red, yellow, green block environment configuration class
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     # 2026-11-22: False so the prestartup self-collision filter
                                                     # (filter_gripper_collisions, G1_REAL_SELF_COLLISION default 1)
                                                     # is allowed by Isaac Lab; one env, so no speed cost
                                                     replicate_physics=False
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
    # 3. MDP settings
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # Lean the torso forward like the robot in the G1_Stack_Block recordings (episode 1 mean waist:
        # yaw -1.0, roll 0.0, pitch +7.4 deg). The waist is held at its default angle by a stiff actuator
        # (the action is an offset from the default), so this also tilts the head camera as on the real robot.
        self.scene.robot.init_state.joint_pos.update({
            "waist_yaw_joint": -0.0175,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.129,
        })
        # No gravity on the robot's own links. With Unitree's soft arm gains (shoulder 25) the replayed shoulders sag
        # 6-8 deg under gravity; the real robot gets gravity-compensating torques the sim never receives. Raising the
        # gains instead made the arms oscillate. Blocks keep gravity.
        spawn = copy.deepcopy(self.scene.robot.spawn)            # copy: don't touch Unitree's shared robot config
        spawn.rigid_props.disable_gravity = True
        # Self-collision for the whole robot (Unitree: off), so the arms/grippers can't pass through each other or the
        # body during teleop (2026-10-27). PhysX skips pairs of links joined by a joint. Override with
        # G1_REAL_SELF_COLLISION=0.
        spawn.articulation_props.enabled_self_collisions = os.environ.get("G1_REAL_SELF_COLLISION", "1") != "0"
        self.scene.robot.spawn = spawn
        # 2026-11-22f: Dex1 finger joint friction. Unitree's config has friction=200 on the four finger joints (PhysX joint
        # friction scales with the force the joint transmits, Isaac default 0). While the arm follows the ghost the
        # fingers carry inertial load, so the friction grew past the drive (800 N/m) and the jaws stuck half-way in
        # mid-air (teleop log 16:23-16:26: cmd closed, state 2.2-3.3, always with the arm following). Real-task copy
        # only; override with G1_REAL_GRIPPER_FRICTION (Unitree's value: 200).
        acts = copy.deepcopy(self.scene.robot.actuators)
        if "hands" in acts:
            acts["hands"].friction = float(os.environ.get("G1_REAL_GRIPPER_FRICTION", "0.0"))
            print(f"[real task] Dex1 finger joint friction = {acts['hands'].friction}")
        self.scene.robot.actuators = acts
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.contact_offset = 0.015
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4
        # create event manager
        self.event_manager = SimpleEventManager() 

        self.event_manager.register_multi_object_reset(
            name="reset_object_self",
            object_names=["red_block", "yellow_block", "green_block"],
            pose_ranges={"x": [-BLOCK_RANGE, BLOCK_RANGE], "y": [-BLOCK_RANGE, BLOCK_RANGE]},  # default 0: episode 1 layout
            velocity_ranges={}
        )
        
        # 方法2：如果需要为不同物体设置不同的重置范围，可以使用这种方式
        # self.event_manager.register("reset_object_self", BatchObjectEvent(
        #     object_names=["red_block", "yellow_block", "green_block"],
        #     pose_ranges={
        #         "red_block": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
        #         "yellow_block": {"x": [-0.03, 0.03], "y": [-0.03, 0.03]},
        #         "green_block": {"x": [-0.08, 0.08], "y": [-0.08, 0.08]}
        #     },
        #     velocity_ranges={}
        # ))
        
        # 方法3：使用MultiObjectEvent的详细配置方式
        # self.event_manager.register("reset_object_self", MultiObjectEvent(
        #     reset_configs=[
        #         {
        #             "asset_cfg": SceneEntityCfg("red_block"),
        #             "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
        #             "velocity_range": {}
        #         },
        #         {
        #             "asset_cfg": SceneEntityCfg("yellow_block"),
        #             "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03]},
        #             "velocity_range": {}
        #         },
        #         {
        #             "asset_cfg": SceneEntityCfg("green_block"),
        #             "pose_range": {"x": [-0.08, 0.08], "y": [-0.08, 0.08]},
        #             "velocity_range": {}
        #         }
        #     ]
        # ))
        
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))