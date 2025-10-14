# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for ARM-T lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """物体是否被抬起到最小高度以上"""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """末端执行器接近物体的奖励（使用tanh核）"""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 物体位置
    cube_pos_w = object.data.root_pos_w
    # 末端执行器位置
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # 距离
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """物体到达目标位置的奖励（使用tanh核）"""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 计算目标位置（世界坐标系）
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # 物体到目标位置的距离
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # 只有当物体被抬起时才给予奖励
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def orientation_command_error(
    env: ManagerBasedRLEnv, 
    minimal_height: float,
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link6"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """惩罚物体方向追踪误差（使用最短路径）
    
    计算目标方向（来自命令）与物体当前方向（世界坐标系）之间的方向误差。
    仅当物体被抬起到最小高度以上时才应用惩罚。
    
    Args:
        env: 环境实例
        minimal_height: 最小高度阈值
        command_name: 命令名称
        asset_cfg: 资产配置（机器人末端链接）
        object_cfg: 物体配置
    
    Returns:
        方向误差（弧度）
    """
    # 提取资产（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 获取目标和当前方向
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    # 不需要考虑末端执行器到TCP的偏移，因为方向没有变化
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    return (object.data.root_pos_w[:, 2] > minimal_height) * quat_error_magnitude(curr_quat_w, des_quat_w)


