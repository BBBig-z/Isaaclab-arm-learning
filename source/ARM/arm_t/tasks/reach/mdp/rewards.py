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


def ee_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std_lin: float = 0.5,
    std_ang: float = 0.5,
    position_threshold: float = 0.1,
    orientation_threshold: float = 0.3,
) -> torch.Tensor:
    """使用指数核奖励末端执行器低速状态（目标速度为0）
    
    模仿track_lin_vel_xy_exp，但目标速度设为0，用于鼓励末端执行器
    在达到目标后减速并保持静止。
    
    仅当末端执行器接近目标（在阈值范围内）时才生效。
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 资产配置（指定追踪的body）
        std_lin: 线性速度的标准差，用于exp核
        std_ang: 角速度的标准差，用于exp核
        position_threshold: 位置距离阈值（米），在此范围内才生效
        orientation_threshold: 姿态误差阈值（弧度），在此范围内才生效
    
    Returns:
        奖励值：接近目标且速度低时奖励高，否则为0
    
    实现逻辑：
        当距离目标 < threshold 时:
            reward = exp(-lin_vel^2/(2*std_lin^2)) * exp(-ang_vel^2/(2*std_ang^2))
        否则:
            reward = 0
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
    near_target = (position_error < position_threshold) & \
                  (orientation_error < orientation_threshold)
    
    # 获取末端执行器的速度（在世界坐标系中）
    # body_state_w的格式为：[pos(3), quat(4), lin_vel(3), ang_vel(3)]
    ee_lin_vel_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10]
    ee_ang_vel_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 10:13]
    
    # 计算速度的L2范数
    lin_vel_norm = torch.norm(ee_lin_vel_w, dim=1)
    ang_vel_norm = torch.norm(ee_ang_vel_w, dim=1)
    
    # 使用指数核计算奖励：exp(-vel^2 / (2 * std^2))
    lin_vel_reward = torch.exp(
        -torch.square(lin_vel_norm) / (2 * std_lin * std_lin)
    )
    ang_vel_reward = torch.exp(
        -torch.square(ang_vel_norm) / (2 * std_ang * std_ang)
    )
    
    # 组合线性和角速度奖励，但只在接近目标时生效
    reward = lin_vel_reward * ang_vel_reward
    return torch.where(near_target, reward, torch.zeros_like(reward))


def joint_vel_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """使用L1范数（绝对值）惩罚关节速度
    
    计算所有配置关节速度的绝对值之和。
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置
    
    Returns:
        关节速度的L1范数（惩罚值，需要配合负权重使用）
    """
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取关节速度
    if asset_cfg.joint_ids is not None:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        joint_vel = asset.data.joint_vel
    
    # 计算L1范数（绝对值之和）
    return torch.sum(torch.abs(joint_vel), dim=1)


def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """使用L2范数惩罚施加在关节上的扭矩
    
    高扭矩往往导致过冲和振荡。此项鼓励能量高效的控制，
    在目标处最小化不必要的力，从而促进稳定。
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置
    
    Returns:
        关节扭矩的L2范数（平方和，惩罚值，需要配合负权重使用）
    """
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取应用的关节扭矩
    if asset_cfg.joint_ids is not None:
        joint_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    else:
        joint_torques = asset.data.applied_torque
    
    # 计算L2范数（平方和）
    return torch.sum(torch.square(joint_torques), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """使用L2范数惩罚关节加速度
    
    振荡往往涉及快速的方向变化（高加速度）。此项鼓励平滑运动，
    并在目标处快速收敛到零加速度，避免来回抖动。
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置
    
    Returns:
        关节加速度的L2范数（平方和，惩罚值，需要配合负权重使用）
    """
    # 提取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取关节加速度
    if asset_cfg.joint_ids is not None:
        joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    else:
        joint_acc = asset.data.joint_acc
    
    # 计算L2范数（平方和）
    return torch.sum(torch.square(joint_acc), dim=1)
