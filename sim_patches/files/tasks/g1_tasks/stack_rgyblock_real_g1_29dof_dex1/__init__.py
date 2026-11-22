# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import gymnasium as gym
import os

from . import stack_rgyblock_real_g1_29dof_dex1_joint_env_cfg


gym.register(
    id="Isaac-Stack-RgyBlock-Real-G129-Dex1-Joint",   # G1_Stack_Block look-alike; Unitree's original task is unchanged
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_rgyblock_real_g1_29dof_dex1_joint_env_cfg.StackRgyBlockRealG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)


# 2026-11-22: new name for the same task (chess pieces on the board in place of the blocks, factory look-alike scene).
# Same env cfg, cameras, DDS and resets; the old id above stays as an alias.
gym.register(
    id="Isaac-Chess-Pieces-G129-Dex1-Joint-Factory",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_rgyblock_real_g1_29dof_dex1_joint_env_cfg.StackRgyBlockRealG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)
