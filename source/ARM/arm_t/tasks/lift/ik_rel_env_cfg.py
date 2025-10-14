# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARM-T抓取任务 - IK控制配置"""

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg

from arm_t.robots import ARM_T_HIGH_PD_CFG
from arm_t.tasks.lift.lift_env_cfg import ArmTLiftEnvCfg


@configclass
class ArmTLift_IK_EnvCfg(ArmTLiftEnvCfg):
    """ARM-T抓取任务 - IK控制"""

    def __post_init__(self):
        # 调用父类后初始化
        super().__post_init__()

        # 设置ARM-T机器人（使用高PD增益配置）
        self.scene.robot = ARM_T_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            "joint1": 0.0,
            "joint2": -1.5,
            "joint3": 1.5,
            "joint4": 0.0,
            "joint5": 1.57,
            "joint6": 0.0,
            "gripper_1_joint": 0.0,
            "gripper_2_joint": 0.0,
        }

        # 设置末端执行器frame传感器（用于观测和奖励计算）
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link6",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.10],  # link6到TCP的偏移（10cm）
                    ),
                ),
            ],
        )

        # 设置抓取物体（立方体）
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # 设置差分IK动作
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        # 设置二元夹爪动作
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_1_joint", "gripper_2_joint"],
            open_command_expr={"gripper_1_joint": 0.0, "gripper_2_joint": 0.0},
            close_command_expr={"gripper_1_joint": 0.04, "gripper_2_joint": 0.04},
        )

        # 设置命令生成器的body_name
        self.commands.object_pose.body_name = "link6"

        # 设置观测中的夹爪关节名称
        self.observations.policy.gripper_joint_pos.params["asset_cfg"].joint_names = [
            "gripper_1_joint",
            "gripper_2_joint",
        ]

        # 设置观测中的TCP位姿body_names
        self.observations.policy.tcp_pose.params["robot_cfg"].body_names = ["link6"]

        # 设置奖励中的方向追踪body_names
        self.rewards.object_orientation_tracking.params["asset_cfg"].body_names = ["link6"]


@configclass
class ArmTLift_IK_EnvCfg_PLAY(ArmTLift_IK_EnvCfg):
    """ARM-T抓取任务 - IK控制（Play模式）"""

    def __post_init__(self):
        # 调用父类后初始化
        super().__post_init__()
        
        # 缩小场景规模
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # 禁用观测噪声
        self.observations.policy.enable_corruption = False
