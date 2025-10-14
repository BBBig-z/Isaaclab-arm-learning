# Copyright (c) 2024-2025, ARM-T Reach Task Custom Rewards
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error_piecewise(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.2,  # 阈值：20度（约0.35弧度）
    large_error_weight: float = 1.0,  # 大误差时的权重系数
    small_error_weight: float = 0.1,  # 小误差时的权重系数
) -> torch.Tensor:
    """分段姿态误差惩罚：误差大时惩罚大，误差小时惩罚小
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 资产配置（指定追踪的body）
        threshold: 误差阈值（弧度），默认0.2（约11.5度）
        large_error_weight: 大误差时的权重系数（默认1.0）
        small_error_weight: 小误差时的权重系数（默认0.1，即降低10倍）
    
    Returns:
        加权的姿态误差
    
    实现逻辑：
        - 当 error > threshold 时：返回 large_error_weight × error（强惩罚）
        - 当 error ≤ threshold 时：返回 small_error_weight × error（弱惩罚）
    """
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 获取期望姿态和当前姿态
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    
    # 计算姿态误差（弧度）
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    # 分段加权
    # 创建权重张量：默认为 small_error_weight
    weights = torch.full_like(error, small_error_weight)
    # 对大误差应用 large_error_weight
    weights[error > threshold] = large_error_weight
    
    # 返回加权误差
    return weights * error


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.3,  # 控制tanh的陡峭度
) -> torch.Tensor:
    """使用tanh核的姿态误差惩罚（平滑版本）
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 资产配置
        std: 标准差，控制tanh的平滑度（值越小，过渡越陡峭）
    
    Returns:
        经过tanh映射的姿态误差（0-1之间）
    
    特性：
        - 误差小时：惩罚接近0（几乎不惩罚）
        - 误差大时：惩罚接近1（饱和）
        - 过渡平滑，避免梯度突变
    """
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 获取期望姿态和当前姿态
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    
    # 计算姿态误差
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    # 使用tanh映射：误差越大，值越接近1
    return torch.tanh(error / std)


def target_hold_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    position_threshold: float = 0.06,  # 6cm
    orientation_threshold: float = 0.175,  # 10度
    hold_time: float = 1.0,  # 需要保持1秒
) -> torch.Tensor:
    """奖励在目标范围内保持一段时间
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 资产配置
        position_threshold: 位置容差（米）
        orientation_threshold: 姿态容差（弧度）
        hold_time: 需要保持的时间（秒）
    
    Returns:
        奖励值：在目标范围内每秒获得奖励，累计到hold_time达到最大值
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
    
    # 判断是否在目标范围内
    in_target = (position_error < position_threshold) & (orientation_error < orientation_threshold)
    
    # 初始化持续时间计数器（如果不存在）
    if not hasattr(env, '_target_hold_timer'):
        env._target_hold_timer = torch.zeros(env.num_envs, device=env.device)
    
    # 更新计时器
    dt = env.step_dt  # 每个控制步的时间间隔
    env._target_hold_timer = torch.where(
        in_target,
        torch.clamp(env._target_hold_timer + dt, max=hold_time),  # 在目标内：增加计时，上限为hold_time
        torch.zeros_like(env._target_hold_timer)  # 离开目标：重置为0
    )
    
    # 计算奖励：根据保持时间线性增长
    # hold_time秒后达到最大奖励1.0
    reward = env._target_hold_timer / hold_time
    
    return reward
