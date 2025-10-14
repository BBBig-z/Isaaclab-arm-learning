# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARM-T Lift任务基础配置

支持多种物体类型的抓取任务
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import MDP functions
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from arm_t.tasks.lift import mdp as arm_mdp

##
# Scene definition
##

# 最小抬起高度
MIN_HEIGHT = 0.1  # 米


@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    """ARM-T抓取任务场景配置"""

    # 机器人：由具体环境配置设置
    robot: ArticulationCfg = MISSING
    
    # 末端执行器传感器：由具体环境配置设置
    ee_frame: FrameTransformerCfg = MISSING
    
    # 目标物体：由具体环境配置设置（支持立方体、圆柱体、球体等）
    object: RigidObjectCfg = MISSING

    # 桌子
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """命令配置 - 定义物体目标位姿"""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # 由具体环境配置设置（如"link6"）
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35),      # 机器人前方15-35cm
            pos_y=(-0.15, 0.15),     # 左右±15cm
            pos_z=(0.20, 0.35),      # 抬起高度20-35cm
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  # 末端执行器朝下
            yaw=(-math.pi, math.pi),   # 允许任意旋转
        ),
    )


@configclass
class ActionsCfg:
    """动作配置"""

    # 机械臂动作：由具体环境配置设置（关节位置控制或IK控制）
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    
    # 夹爪动作：由具体环境配置设置
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """观测配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组"""

        # 夹爪关节位置
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},  # 由具体环境设置
        )
        
        # TCP位姿（在机器人基座坐标系中）
        tcp_pose = ObsTerm(
            func=arm_mdp.get_current_tcp_pose,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=MISSING)},  # 由具体环境设置
        )
        
        # 物体位姿（在机器人基座坐标系中）
        object_pose = ObsTerm(
            func=arm_mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # 目标物体位姿
        target_object_pose = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置"""

    # 重置所有状态
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 随机化物体初始位置
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """奖励配置 - 参考UR5e lift项目"""

    # 接近物体的奖励
    reaching_object = RewTerm(
        func=arm_mdp.object_ee_distance,
        params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    # 抬起物体的奖励
    lifting_object = RewTerm(
        func=arm_mdp.object_is_lifted,
        params={"minimal_height": MIN_HEIGHT},
        weight=15.0,
    )

    # 物体到达目标位置的奖励（粗粒度）
    object_goal_tracking = RewTerm(
        func=arm_mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": MIN_HEIGHT, "command_name": "object_pose"},
        weight=16.0,
    )

    # 物体到达目标位置的奖励（精细粒度）
    object_goal_tracking_fine_grained = RewTerm(
        func=arm_mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": MIN_HEIGHT, "command_name": "object_pose"},
        weight=5.0,
    )

    # 物体方向追踪奖励
    object_orientation_tracking = RewTerm(
        func=arm_mdp.orientation_command_error,
        weight=-6.0,
        params={
            "minimal_height": MIN_HEIGHT,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),  # 由具体环境设置
        },
    )

    # 动作惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """终止条件配置"""

    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 物体掉落
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """课程学习配置"""

    # 逐步增加动作惩罚权重
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class ArmTLiftEnvCfg(ManagerBasedRLEnvCfg):
    """ARM-T抓取任务环境配置"""

    # 场景设置
    scene: LiftSceneCfg = LiftSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化"""
        # 通用设置
        self.decimation = 2
        self.episode_length_s = 5.0
        
        # 仿真设置
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # PhysX设置
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
