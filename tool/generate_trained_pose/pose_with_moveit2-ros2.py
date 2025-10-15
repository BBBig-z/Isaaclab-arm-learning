#!/usr/bin/env python3
"""
ARM-T 机械臂可达位姿数据库生成工具 - ROS2 + MoveIt2 版本

使用 MoveIt2 框架进行正运动学计算和碰撞检测，批量生成可达位姿数据库

功能特性:
    ✓ 正运动学计算 (FK)
    ✓ 自碰撞检测 (Self-collision detection)
    ✓ 工作空间过滤
    ✓ 多种采样策略 (uniform/centered/mixed)
    ✓ 实时进度报告

依赖:
    - ROS2 (Humble/Foxy/Iron)
    - MoveIt2
    - Python3
    - numpy

使用方法:
    1. 确保 ROS2 环境已激活: source /opt/ros/humble/setup.bash
    2. 启动 MoveIt2 服务:
       ros2 launch moveit2_tutorials demo.launch.py
    3. 运行此脚本:
       python3 pose_with_moveit2-ros2.py --num_samples 10000
    
    禁用碰撞检测（更快，但可能包含自碰撞位姿）:
       python3 pose_with_moveit2-ros2.py --num_samples 10000 --no_collision_check

示例:
    # 生成 5000 个位姿，启用碰撞检测
    python3 pose_with_moveit2-ros2.py --num_samples 5000
    
    # 使用均匀采样策略
    python3 pose_with_moveit2-ros2.py --num_samples 5000 --strategy uniform
    
    # 指定输出路径
    python3 pose_with_moveit2-ros2.py --num_samples 5000 --output custom_poses.pkl
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK, GetStateValidity
from moveit_msgs.msg import RobotState, Constraints
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
import pickle
from pathlib import Path
import time
from typing import List, Tuple, Dict
import argparse


class ReachablePoseGenerator(Node):
    """使用 MoveIt2 生成可达位姿数据库"""
    
    def __init__(self, 
                 robot_name="panther_description",
                 group_name="arm",
                 ee_link="link6",
                 base_link="base_link",
                 enable_collision_check=True):
        super().__init__('reachable_pose_generator')
        
        self.robot_name = robot_name
        self.group_name = group_name
        self.ee_link = ee_link
        self.base_link = base_link
        self.enable_collision_check = enable_collision_check
        
        # 创建 FK 服务客户端
        self.fk_client = self.create_client(
            GetPositionFK, 
            '/compute_fk'
        )
        
        # 等待 FK 服务可用
        self.get_logger().info('等待 FK 服务...')
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('FK 服务不可用！请先启动 MoveIt2')
            raise RuntimeError('FK 服务不可用')
        
        self.get_logger().info('✓ FK 服务已连接')
        
        # 创建碰撞检测服务客户端（如果启用）
        if self.enable_collision_check:
            self.collision_client = self.create_client(
                GetStateValidity,
                '/check_state_validity'
            )
            
            self.get_logger().info('等待碰撞检测服务...')
            if not self.collision_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().warn('碰撞检测服务不可用，将禁用碰撞检测')
                self.enable_collision_check = False
            else:
                self.get_logger().info('✓ 碰撞检测服务已连接')
        
        # 关节名称 (根据 URDF)
        self.joint_names = [
            'joint1',
            'joint2', 
            'joint3',
            'joint4',
            'joint5',
            'joint6'
        ]
        
        # 关节限位 (弧度)
        self.joint_limits = [
            (-np.pi, np.pi),   # joint1
            (-np.pi, np.pi),   # joint2
            (-np.pi, np.pi),   # joint3
            (-np.pi, np.pi),   # joint4
            (-np.pi, np.pi),   # joint5
            (-np.pi, np.pi),   # joint6
        ]
        
        # 工作空间边界 (米)
        self.workspace_bounds = {
            'x': (0.10, 0.35),
            'y': (-0.20, 0.20),
            'z': (0.10, 0.40)
        }
        
        self.get_logger().info(f'机器人: {robot_name}')
        self.get_logger().info(f'规划组: {group_name}')
        self.get_logger().info(f'末端执行器: {ee_link}')
    
    def compute_fk(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        计算正运动学
        
        Args:
            joint_positions: 关节角度列表 (弧度)
        
        Returns:
            (position, orientation_quat, success)
            - position: [x, y, z] 位置 (米)
            - orientation_quat: [w, x, y, z] 四元数
            - success: 是否成功
        """
        # 创建请求
        request = GetPositionFK.Request()
        
        # 设置 header
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = self.base_link
        
        # 设置末端执行器链接
        request.fk_link_names = [self.ee_link]
        
        # 设置机器人状态
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = joint_positions
        robot_state.joint_state = joint_state
        request.robot_state = robot_state
        
        # 调用服务
        try:
            future = self.fk_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            if future.done():
                response = future.result()
                
                if response.error_code.val == 1:  # SUCCESS
                    pose = response.pose_stamped[0].pose
                    
                    # 提取位置
                    position = np.array([
                        pose.position.x,
                        pose.position.y,
                        pose.position.z
                    ])
                    
                    # 提取四元数 [w, x, y, z]
                    orientation_quat = np.array([
                        pose.orientation.w,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z
                    ])
                    
                    return position, orientation_quat, True
                else:
                    return np.zeros(3), np.array([1, 0, 0, 0]), False
            else:
                return np.zeros(3), np.array([1, 0, 0, 0]), False
                
        except Exception as e:
            self.get_logger().warn(f'FK 计算失败: {e}')
            return np.zeros(3), np.array([1, 0, 0, 0]), False
    
    def check_collision(self, joint_positions: List[float]) -> bool:
        """
        检查给定关节配置是否有碰撞
        
        Args:
            joint_positions: 关节角度列表 (弧度)
        
        Returns:
            True 如果有碰撞, False 如果无碰撞
        """
        if not self.enable_collision_check:
            return False  # 未启用碰撞检测，默认无碰撞
        
        # 创建请求
        request = GetStateValidity.Request()
        
        # 设置机器人状态
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = joint_positions
        robot_state.joint_state = joint_state
        request.robot_state = robot_state
        
        # 设置规划组
        request.group_name = self.group_name
        
        # 调用服务
        try:
            future = self.collision_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            if future.done():
                response = future.result()
                # valid = True 表示无碰撞
                return not response.valid
            else:
                # 超时，假设有碰撞（保守策略）
                return True
                
        except Exception as e:
            self.get_logger().warn(f'碰撞检测失败: {e}')
            return True  # 检测失败，假设有碰撞
    
    def is_in_workspace(self, position: np.ndarray) -> bool:
        """检查位置是否在工作空间内"""
        x, y, z = position
        return (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1])
    
    def sample_joint_angles(self, num_samples: int, strategy: str = 'mixed') -> List[List[float]]:
        """
        采样关节角度
        
        Args:
            num_samples: 样本数量
            strategy: 采样策略
                - 'uniform': 均匀随机
                - 'centered': 偏向中心
                - 'mixed': 混合 (70% 中心 + 30% 均匀)
        
        Returns:
            关节角度列表
        """
        samples = []
        
        if strategy == 'uniform':
            # 完全随机
            for _ in range(num_samples):
                angles = [np.random.uniform(low, high) for low, high in self.joint_limits]
                samples.append(angles)
        
        elif strategy == 'centered':
            # 偏向中间位置
            for _ in range(num_samples):
                angles = []
                for low, high in self.joint_limits:
                    center = (low + high) / 2
                    std = (high - low) / 4
                    angle = np.random.normal(center, std)
                    angle = np.clip(angle, low, high)
                    angles.append(angle)
                samples.append(angles)
        
        elif strategy == 'mixed':
            # 混合策略
            n_centered = int(num_samples * 0.7)
            n_uniform = num_samples - n_centered
            
            # 70% 中心区域
            for _ in range(n_centered):
                angles = []
                for low, high in self.joint_limits:
                    center = (low + high) / 2
                    std = (high - low) / 6
                    angle = np.random.normal(center, std)
                    angle = np.clip(angle, low, high)
                    angles.append(angle)
                samples.append(angles)
            
            # 30% 全范围
            for _ in range(n_uniform):
                angles = [np.random.uniform(low, high) for low, high in self.joint_limits]
                samples.append(angles)
        
        return samples
    
    def generate_database(self, 
                         num_samples: int = 10000,
                         output_path: str = None,
                         sampling_strategy: str = 'mixed') -> Dict:
        """
        生成可达位姿数据库
        
        Args:
            num_samples: 目标样本数量
            output_path: 输出文件路径
            sampling_strategy: 采样策略
        
        Returns:
            数据库字典
        """
        self.get_logger().info("=" * 70)
        self.get_logger().info("开始生成可达位姿数据库")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f'目标样本数: {num_samples}')
        self.get_logger().info(f'采样策略: {sampling_strategy}')
        self.get_logger().info(f'工作空间: X{self.workspace_bounds["x"]}, '
                             f'Y{self.workspace_bounds["y"]}, '
                             f'Z{self.workspace_bounds["z"]}')
        
        positions = []
        orientations_quat = []
        joint_configs = []
        
        # 统计
        fk_failed = 0
        collision_rejected = 0
        workspace_filtered = 0
        attempts = 0
        
        # 增加尝试次数以确保达到目标
        max_attempts = num_samples * 10
        candidate_angles = self.sample_joint_angles(max_attempts, sampling_strategy)
        
        start_time = time.time()
        last_report_time = start_time
        
        self.get_logger().info('开始计算...')
        
        for joint_angles in candidate_angles:
            if len(positions) >= num_samples:
                break
            
            attempts += 1
            
            # 进度报告 (每5秒)
            current_time = time.time()
            if current_time - last_report_time > 5.0:
                progress = len(positions) / num_samples * 100
                speed = len(positions) / (current_time - start_time)
                self.get_logger().info(
                    f'进度: {len(positions)}/{num_samples} ({progress:.1f}%) | '
                    f'速度: {speed:.1f} 位姿/秒 | '
                    f'尝试: {attempts}'
                )
                last_report_time = current_time
            
            # 碰撞检测（先检查，避免不必要的 FK 计算）
            if self.enable_collision_check:
                has_collision = self.check_collision(joint_angles)
                if has_collision:
                    collision_rejected += 1
                    continue
            
            # 计算 FK
            position, quat, success = self.compute_fk(joint_angles)
            
            if not success:
                fk_failed += 1
                continue
            
            # 检查工作空间
            if not self.is_in_workspace(position):
                workspace_filtered += 1
                continue
            
            # 有效位姿
            positions.append(position)
            orientations_quat.append(quat)
            joint_configs.append(joint_angles)
        
        elapsed_time = time.time() - start_time
        
        # 转换为 numpy 数组
        positions = np.array(positions)
        orientations_quat = np.array(orientations_quat)
        joint_configs = np.array(joint_configs)
        
        # 创建数据库
        database = {
            'num_poses': len(positions),
            'positions': positions,
            'orientations_quat': orientations_quat,
            'joint_configs': joint_configs,
            'workspace': {
                'x_range': (positions[:, 0].min(), positions[:, 0].max()),
                'y_range': (positions[:, 1].min(), positions[:, 1].max()),
                'z_range': (positions[:, 2].min(), positions[:, 2].max()),
            },
            'generation_method': 'ros2_moveit2_fk',
            'robot_name': self.robot_name,
            'group_name': self.group_name,
            'ee_link': self.ee_link,
            'num_samples_requested': num_samples,
            'sampling_strategy': sampling_strategy,
            'collision_check_enabled': self.enable_collision_check,
            'generation_time_seconds': elapsed_time,
            'statistics': {
                'total_attempts': attempts,
                'collision_rejected': collision_rejected,
                'fk_failed': fk_failed,
                'workspace_filtered': workspace_filtered,
                'success_rate': len(positions) / attempts if attempts > 0 else 0,
            },
        }
        
        # 保存数据库
        if output_path is None:
            script_dir = Path(__file__).parent.parent.parent
            output_path = script_dir / "source/ARM/arm_t/tasks/reach/reachable_poses_database.pkl"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(database, f)
        
        # 打印总结
        self.get_logger().info("=" * 70)
        self.get_logger().info("✓ 生成完成!")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f'输出文件: {output_path}')
        self.get_logger().info(f'有效位姿: {len(positions)}')
        self.get_logger().info(f'生成时间: {elapsed_time:.1f} 秒')
        self.get_logger().info(f'生成速度: {len(positions) / elapsed_time:.1f} 位姿/秒')
        self.get_logger().info('')
        self.get_logger().info('统计:')
        self.get_logger().info(f'  总尝试: {attempts}')
        self.get_logger().info(f'  碰撞拒绝: {collision_rejected}')
        self.get_logger().info(f'  FK 失败: {fk_failed}')
        self.get_logger().info(f'  工作空间过滤: {workspace_filtered}')
        self.get_logger().info(f'  成功率: {database["statistics"]["success_rate"]*100:.1f}%')
        self.get_logger().info(f'  碰撞检测: {"✓ 已启用" if self.enable_collision_check else "✗ 已禁用"}')
        self.get_logger().info('')
        self.get_logger().info('工作空间范围:')
        self.get_logger().info(f'  X: [{database["workspace"]["x_range"][0]:.3f}, '
                             f'{database["workspace"]["x_range"][1]:.3f}] m')
        self.get_logger().info(f'  Y: [{database["workspace"]["y_range"][0]:.3f}, '
                             f'{database["workspace"]["y_range"][1]:.3f}] m')
        self.get_logger().info(f'  Z: [{database["workspace"]["z_range"][0]:.3f}, '
                             f'{database["workspace"]["z_range"][1]:.3f}] m')
        
        return database


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ARM-T 可达位姿数据库生成工具 - ROS2 + MoveIt2"
    )
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='目标样本数量 (默认: 10000)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    parser.add_argument('--strategy', type=str, default='mixed',
                       choices=['uniform', 'centered', 'mixed'],
                       help='采样策略 (默认: mixed)')
    parser.add_argument('--no_collision_check', action='store_true',
                       help='禁用碰撞检测（生成速度更快，但可能包含自碰撞位姿）')
    parser.add_argument('--robot_name', type=str, default='panther_description',
                       help='机器人名称')
    parser.add_argument('--group_name', type=str, default='arm',
                       help='MoveIt 规划组名称')
    parser.add_argument('--ee_link', type=str, default='link6',
                       help='末端执行器链接名称')
    
    args = parser.parse_args()
    
    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建节点
        generator = ReachablePoseGenerator(
            robot_name=args.robot_name,
            group_name=args.group_name,
            ee_link=args.ee_link,
            enable_collision_check=not args.no_collision_check
        )
        
        # 生成数据库
        generator.generate_database(
            num_samples=args.num_samples,
            output_path=args.output,
            sampling_strategy=args.strategy
        )
        
    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception as e:
        print(f'\n错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        rclpy.shutdown()


if __name__ == '__main__':
    main()

