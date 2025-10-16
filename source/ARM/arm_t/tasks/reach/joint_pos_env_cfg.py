# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from arm_t.robots import ARM_T_CFG  # noqa: F401
from arm_t.tasks.reach.reach_env_cfg import ReachEnvCfg

##
# Scene definition
##

# ----------------------------------------------------------------
# --------------- ARM-T 6-DOF Robot Configuration ----------------
# ----------------------------------------------------------------


@configclass
class ArmTReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ARM-T
        self.scene.robot = ARM_T_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override rewards - using link6 as end effector (last joint)
        # 注意：对于关节位置控制，我们追踪 link6
        # 实际的夹爪位置会在前方约10cm
        body_names = ["link6"]
        self.rewards.end_effector_position_tracking.params[
            "asset_cfg"
        ].body_names = body_names
        self.rewards.end_effector_position_tracking_fine_grained.params[
            "asset_cfg"
        ].body_names = body_names
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = body_names
        self.rewards.ee_velocity_penalty.params[
            "asset_cfg"
        ].body_names = body_names

        # override actions - 6 DOF arm joints only
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6"
            ],
            scale=0.3,
            use_default_offset=True,
        )
        # Gripper stays at default position via PD control (no action needed)

        # IMPORTANT: Override observations to only observe the 6 arm joints
        # (not gripper joints)
        # This prevents NaN/Inf from uncontrolled gripper joints
        joint_ids = [0, 1, 2, 3, 4, 5]
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_ids=joint_ids)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_ids=joint_ids)
        }

        # override command generator body
        # end-effector is link6 with offset to gripper tip (夹爪最前端)
        self.commands.ee_pose.body_name = ["link6"]


@configclass
class ArmTReachEnvCfg_PLAY(ArmTReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
