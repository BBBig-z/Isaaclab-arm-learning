# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARM-T Lift任务的终止条件函数"""

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
    """物体是否到达目标位置的终止条件
    
    Args:
        env: 环境实例
        command_name: 命令名称
        threshold: 距离阈值（米），默认0.02米
        robot_cfg: 机器人配置
        object_cfg: 物体配置
    
    Returns:
        布尔张量，指示每个环境中物体是否到达目标
    """
    # 提取所需的对象
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 计算世界坐标系中的目标位置
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    
    # 计算物体到目标位置的距离
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    
    # 如果距离小于阈值则返回True
    return distance < threshold

