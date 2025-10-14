# Copyright (c) 2024-2025, ARM-T Reach Task Custom Commands
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import pickle
from pathlib import Path
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReachablePoseCommand(UniformPoseCommand):
    """基于预计算数据库的可达位姿命令生成器
    
    从预计算的可达位姿数据库中采样目标，确保：
    1. 所有目标位置都100%可达
    2. 位置和姿态配合合理（来自实际FK计算）
    3. 训练效率更高（不会浪费时间尝试不可达目标）
    """
    
    cfg: ReachablePoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ReachablePoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration for the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        
        # 加载可达位姿数据库
        database_path = Path(__file__).parent.parent / cfg.database_path
        if not database_path.exists():
            raise FileNotFoundError(
                f"可达位姿数据库不存在: {database_path}\n"
                f"请先运行: python generate_reachable_poses.py"
            )
        
        with open(database_path, 'rb') as f:
            database = pickle.load(f)
        
        # 转换为torch tensors
        self.database_positions = torch.tensor(
            database['positions'], 
            dtype=torch.float32, 
            device=self.device
        )  # (N, 3)
        
        self.database_orientations = torch.tensor(
            database['orientations_quat'], 
            dtype=torch.float32, 
            device=self.device
        )  # (N, 4) [w, x, y, z]
        
        self.num_database_poses = database['num_poses']
        
        print(f"[ReachablePoseCommand] 加载可达位姿数据库")
        print(f"  路径: {database_path}")
        print(f"  位姿数量: {self.num_database_poses}")
        print(f"  工作空间: {database['workspace']}")
    
    def _resample_command(self, env_ids: torch.Tensor):
        """从数据库中随机采样可达位姿
        
        Args:
            env_ids: 需要重新采样的环境ID
        """
        num_envs = len(env_ids)
        
        # 从数据库中随机采样索引
        indices = torch.randint(
            0, self.num_database_poses, 
            (num_envs,), 
            device=self.device
        )
        
        # 获取对应的位置和姿态
        sampled_positions = self.database_positions[indices]  # (num_envs, 3)
        sampled_orientations = self.database_orientations[indices]  # (num_envs, 4)
        
        # 设置命令 - 组合为完整的位姿 (position + quaternion)
        # pose_command_b shape: (num_envs, 7) = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        self.pose_command_b[env_ids, :3] = sampled_positions
        self.pose_command_b[env_ids, 3:] = sampled_orientations


@configclass
class ReachablePoseCommandCfg(UniformPoseCommandCfg):
    """基于预计算数据库的可达位姿命令配置
    
    使用方法：
        commands = CommandsCfg()
        commands.ee_pose = ReachablePoseCommandCfg(
            asset_name="robot",
            body_name=["link6"],
            resampling_time_range=(5.0, 5.0),
            database_path="reachable_poses_database.pkl",
        )
    """
    
    class_type: type = ReachablePoseCommand
    
    # 数据库路径（相对于mdp目录的上级）
    database_path: str = "reachable_poses_database.pkl"
    
    # 提供默认的ranges（虽然不会使用，但需要存在以满足基类验证）
    # 这些值永远不会被实际使用，因为我们从数据库采样
    ranges: UniformPoseCommandCfg.Ranges = UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0),
        pos_y=(0.0, 0.0),
        pos_z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )

