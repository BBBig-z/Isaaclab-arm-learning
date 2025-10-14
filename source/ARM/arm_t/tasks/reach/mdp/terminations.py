# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the reach task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def ee_reached_target(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    position_threshold: float = 0.06,  # 6cm
    orientation_threshold: float = 0.175,  # 10度（0.175弧度）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link6"]),
) -> torch.Tensor:
    """终止条件：末端执行器到达目标位置和姿态
    
    Args:
        env: 环境实例
        command_name: 命令名称（默认"ee_pose"）
        position_threshold: 位置误差阈值（米），默认6cm
        orientation_threshold: 姿态误差阈值（弧度），默认10度
        asset_cfg: 资产配置
    
    Returns:
        布尔张量，True表示到达目标（触发终止）
    """
    from isaaclab.utils.math import quat_error_magnitude, quat_mul
    
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 计算位置误差
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], 
        asset.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # 计算姿态误差
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    # 判断是否同时满足位置和姿态要求
    position_ok = position_error < position_threshold
    orientation_ok = orientation_error < orientation_threshold
    
    return position_ok & orientation_ok

