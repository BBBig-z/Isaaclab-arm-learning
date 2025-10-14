# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARM-T Lift任务的观测函数"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """使用四元数旋转向量
    
    Args:
        quat: [..., 4] 四元数张量 (w, x, y, z)
        vec: [..., 3] 要旋转的向量张量
    
    Returns:
        旋转后的向量，形状为 [..., 3]
    """
    # 归一化四元数以避免意外的缩放效果
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    
    # 使用四元数旋转输入向量
    rotated_vec = quat_apply(quat, vec)
    
    return rotated_vec


def get_current_tcp_pose(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算当前TCP在基座坐标系和世界坐标系中的位姿
    
    Args:
        env: 环境实例
        robot_cfg: 机器人配置
    
    Returns:
        tcp_pose_b: TCP在机器人基座坐标系中的位姿 (位置 + 四元数)
    """
    # 从场景中获取机器人对象
    robot: RigidObject = env.scene[robot_cfg.name]

    # 克隆世界坐标系中的body状态，避免修改原始张量
    body_state_w_list = robot.data.body_state_w.clone()

    # 提取世界坐标系中末端执行器的位姿（位置 + 方向）
    ee_pose_w = body_state_w_list[:, robot_cfg.body_ids[0], :7]

    # 定义从末端执行器坐标系到TCP的偏移量（在末端执行器坐标系中）
    # ARM-T的link6到TCP的偏移，根据实际机器人调整
    offset_ee = torch.tensor([0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.scene.num_envs, 1)

    # 将偏移量从末端执行器坐标系旋转到世界坐标系
    offset_w = quat_rotate_vector(ee_pose_w[:, 3:7], offset_ee)

    # 通过将偏移量加到末端执行器位置来计算世界坐标系中的TCP位姿
    tcp_pose_w = torch.cat((ee_pose_w[:, :3] + offset_w, ee_pose_w[:, 3:7]), dim=-1)

    # 将TCP位姿从世界坐标系转换到机器人基座坐标系
    tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # 机器人基座在世界坐标系中的位置
        robot.data.root_state_w[:, 3:7],  # 机器人基座在世界坐标系中的方向
        tcp_pose_w[:, :3],  # TCP在世界坐标系中的位置
        tcp_pose_w[:, 3:7]  # TCP在世界坐标系中的方向
    )

    tcp_pose_b = torch.cat((tcp_pos_b, tcp_quat_b), dim=-1)
    return tcp_pose_b


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """获取物体在机器人根坐标系中的位姿（位置 + 方向）
    
    Args:
        env: 环境实例
        robot_cfg: 机器人配置
        object_cfg: 物体配置
    
    Returns:
        物体在机器人根坐标系中的位姿 (num_envs, 7) - position(3) + quaternion(4)
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        object_pos_w
    )
    object_pose_b = torch.cat((object_pos_b, object_quat_b), dim=-1)
    return object_pose_b

