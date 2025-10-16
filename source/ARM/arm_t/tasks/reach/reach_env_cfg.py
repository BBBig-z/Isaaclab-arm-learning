# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# import custom mdp functions for ARM-T
from arm_t.tasks.reach import mdp as arm_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # 使用预计算的可达位姿数据库
    # 确保所有目标位置都是100%可达的，位置和姿态配合合理
    ee_pose = arm_mdp.ReachablePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        database_path="reachable_poses_database.pkl",
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-6,  # 原始权重（稳定训练的关键）
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2,  # 提高精细位置奖励（0.1 → 0.28）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )
    
    # 分段姿态惩罚：误差大时惩罚大，误差小时惩罚小
    end_effector_orientation_tracking = RewTerm(
        func=arm_mdp.orientation_command_error_piecewise,
        weight=-3,  # 基础权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
            "threshold": 0.23,
            "large_error_weight": 1.0,  # 大误差时：权重×1.0（全惩罚）
            "small_error_weight": 0.1,  # 小误差时：权重×0.6（降低40%）
        },
    )

    # action penalty - 恢复原始权重
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)  # 原始权重
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.08,  # 原始权重
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 保持目标奖励 - 鼓励在目标位置保持稳定
    target_hold_bonus = RewTerm(
        func=arm_mdp.target_hold_bonus,
        weight=5,  # 保持1秒可获得0.12的额外奖励
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),  # 使用link6而不是MISSING
            "command_name": "ee_pose",
            "position_threshold": 0.05,  # 位置容差6cm
            "orientation_threshold": 0.2,  # 姿态容差10度
            "hold_time": 1.5,  # 保持1秒
        },
    )
    
    # === 新增：稳定性奖励项 ===
    # 末端执行器速度惩罚（指数形式）- 鼓励在目标处保持静止
    # 仅在接近目标时生效
    ee_velocity_penalty = RewTerm(
        func=arm_mdp.ee_velocity_exp,
        weight=0.5,  # 正权重，因为函数返回的是奖励（速度低时奖励高）
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std_lin": 0.02,  # 线性速度标准差（m/s）
            "std_ang": 0.1,  # 角速度标准差（rad/s）
            "position_threshold": 0.07,  # 10cm内才生效
            "orientation_threshold": 0.3,  # 约17度内才生效
        },
    )
    
    # 关节速度惩罚（L1范数）- 替代joint_vel_l2的另一种选择
    joint_vel_l1_penalty = RewTerm(
        func=arm_mdp.joint_vel_l1,
        weight=-0.02,  # 负权重，惩罚关节速度
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_ids=[0, 1, 2, 3, 4, 5]
            )
        },
    )
    
    # 关节扭矩惩罚（L2范数）- 鼓励能量高效的控制
    joint_torques_penalty = RewTerm(
        func=arm_mdp.joint_torques_l2,
        weight=-0.002,  # 负权重，惩罚高扭矩
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_ids=[0, 1, 2, 3, 4, 5]
            )
        },
    )
    
    # 关节加速度惩罚（L2范数）- 鼓励平滑运动，避免振荡
    joint_acc_penalty = RewTerm(
        func=arm_mdp.joint_acc_l2,
        weight=-0.0002,  # 负权重，惩罚高加速度
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_ids=[0, 1, 2, 3, 4, 5]
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.12, "num_steps": 1500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.17, "num_steps": 1500}
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings - 减少环境数量以提高数值稳定性
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0  # 增加到20秒，给机械臂更多时间
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

