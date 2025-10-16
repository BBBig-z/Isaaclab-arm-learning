# Copyright (c) 2024-2025, ARM-T Reach Task Custom Observations
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_target_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    position_threshold: float = 0.015,  # 1cm
    orientation_threshold: float = 0.1,  # 12度
) -> torch.Tensor:
    """判断是否到达目标（用于统计成功率）
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 资产配置
        position_threshold: 位置误差阈值（米）
        orientation_threshold: 姿态误差阈值（弧度）
    
    Returns:
        布尔张量，True表示到达目标
    """
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
