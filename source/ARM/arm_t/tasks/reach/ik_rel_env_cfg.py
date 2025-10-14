# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from arm_t.robots import ARM_T_HIGH_PD_CFG  # noqa: F401

from .joint_pos_env_cfg import ArmTReachEnvCfg

# ----------------------------------------------------------------
# --------------- ARM-T 6-DOF IK Configuration -------------------
# ----------------------------------------------------------------


@configclass
class ArmTReach_IK_EnvCfg(ArmTReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set ARM-T with high PD gains
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = ARM_T_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (ARM-T 6-DOF)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            body_name="link6",  # Using link6 as end effector base
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.3,
            # 直接使用 link6 位置，不偏移
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )


@configclass
class ArmTReach_IK_EnvCfg_PLAY(ArmTReach_IK_EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
