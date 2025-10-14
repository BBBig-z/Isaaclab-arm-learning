#!/usr/bin/env python3
"""
生成ARM-T机械臂的可达位姿数据库（使用真实URDF模型）

该脚本通过前向运动学（FK）采样生成100%可达的末端执行器位姿数据库，
并进行碰撞检测，确保所有目标位置都是无碰撞的可达位姿。

特性：
- 使用真实URDF模型进行前向运动学
- PyBullet物理引擎进行自碰撞检测
- 环境碰撞检测（地面）
- 工作空间过滤
"""

import numpy as np
import pickle
from pathlib import Path
import argparse
import os
import pybullet as p
import pybullet_data


class URDFRobotSimulator:
    """使用PyBullet加载URDF并进行FK和碰撞检测"""
    
    def __init__(self, urdf_path, end_effector_link_name="link6"):
        """
        初始化PyBullet模拟器
        
        Args:
            urdf_path: URDF文件路径
            end_effector_link_name: 末端执行器连杆名称
        """
        # 连接到PyBullet（使用DIRECT模式，无GUI）
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加PyBullet数据路径
        
        # 加载地面平面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 处理URDF文件中的ROS package路径
        urdf_path = Path(urdf_path)
        modified_urdf_path = self._fix_urdf_mesh_paths(urdf_path)
        
        # 加载机器人URDF
        self.robot_id = p.loadURDF(
            str(modified_urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # 只关注旋转关节
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
        
        # 查找末端执行器连杆索引
        self.ee_link_index = None
        for i in range(self.num_joints):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if link_name == end_effector_link_name:
                self.ee_link_index = i
                break
        
        if self.ee_link_index is None:
            # 如果找不到，使用最后一个连杆
            self.ee_link_index = self.num_joints - 1
        
        # 碰撞容差设置
        self.collision_tolerance = 0.03  
        self.initial_contact_position_tolerance = 0.06  # 5cm，用于判断接触点是否为初始接触点
        
        # 记录初始状态的接触点信息（零位时存在的接触）
        self._initial_contact_points = self._get_initial_contact_points()
        
        print(f"✓ PyBullet模拟器已初始化")
        print(f"  机器人ID: {self.robot_id}")
        print(f"  控制关节数: {len(self.joint_indices)}")
        print(f"  关节名称: {self.joint_names}")
        print(f"  末端执行器连杆索引: {self.ee_link_index}")
        print(f"  碰撞容差: {self.collision_tolerance*100:.1f}cm")
        print(f"  忽略的初始接触点: {len(self._initial_contact_points)} 个")
        print(f"  接触点位置容差: {self.initial_contact_position_tolerance*100:.1f}cm")
    
    def _fix_urdf_mesh_paths(self, urdf_path):
        """
        修复URDF文件中的mesh路径（ROS package格式 -> 绝对路径）
        
        Args:
            urdf_path: 原始URDF文件路径
            
        Returns:
            修复后的临时URDF文件路径
        """
        import tempfile
        import re
        
        # 读取URDF文件
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        
        # 获取meshes目录的绝对路径
        meshes_dir = urdf_path.parent.parent / "meshes"
        
        # 替换ROS package路径为绝对路径
        # package://panther_description/meshes/ -> 绝对路径
        pattern = r'package://[^/]+/meshes/'
        replacement = f'file://{meshes_dir}/'
        urdf_content = re.sub(pattern, replacement, urdf_content)
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.urdf', text=True)
        with os.fdopen(temp_fd, 'w') as f:
            f.write(urdf_content)
        
        self._temp_urdf_path = temp_path  # 保存以便清理
        return temp_path
    
    def _get_initial_contact_points(self):
        """
        获取机械臂在零位（初始状态）时的接触点信息
        记录接触点的连杆对和世界坐标位置，用于精细的碰撞过滤
        
        Returns:
            list: 初始状态下的接触点信息列表
                  每个元素为 dict: {
                      'link_pair': (linkA, linkB),
                      'position_on_A': np.array([x, y, z]),
                      'position_on_B': np.array([x, y, z])
                  }
        """
        # 将机械臂设置为初始位置（根据URDF默认配置）
        # joint1=0°, joint2=25°, joint3=17°, joint4=-35°, joint5=0°, joint6=0°
        initial_angles = [
            0.0,                          # joint1: 0°
            np.deg2rad(25.0),            # joint2: 25°
            np.deg2rad(17.0),            # joint3: 17°
            np.deg2rad(-35.0),           # joint4: -35°
            0.0,                          # joint5: 0°
            0.0,                          # joint6: 0°
        ]
        self.set_joint_angles(initial_angles)
        
        # 执行碰撞检测
        p.performCollisionDetection()
        
        # 收集零位时的接触点详细信息
        initial_contacts = []
        contact_points = p.getContactPoints(self.robot_id, self.robot_id)
        for contact in contact_points:
            linkA = contact[3]  # 连杆A的索引
            linkB = contact[4]  # 连杆B的索引
            positionOnA = np.array(contact[5])  # linkA上的接触点位置（世界坐标）
            positionOnB = np.array(contact[6])  # linkB上的接触点位置（世界坐标）
            
            # 保存接触点信息
            contact_info = {
                'link_pair': tuple(sorted([linkA, linkB])),
                'position_on_A': positionOnA,
                'position_on_B': positionOnB,
            }
            initial_contacts.append(contact_info)
        
        return initial_contacts
    
    def set_joint_angles(self, joint_angles):
        """设置关节角度"""
        for i, angle in enumerate(joint_angles):
            if i < len(self.joint_indices):
                p.resetJointState(self.robot_id, self.joint_indices[i], angle)
    
    def get_end_effector_pose(self):
        """
        获取末端执行器位姿
        
        Returns:
            position: [x, y, z]
            orientation_quat: [qw, qx, qy, qz] (PyBullet格式: [x,y,z,w])
        """
        link_state = p.getLinkState(self.robot_id, self.ee_link_index)
        position = np.array(link_state[0])  # 世界坐标系中的位置
        orientation = np.array(link_state[1])  # 四元数 [x,y,z,w]
        
        # PyBullet返回的是[x,y,z,w]格式，转换为[w,x,y,z]
        orientation_quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
        
        return position, orientation_quat
    
    def check_collision(self, distance_threshold=0.01):
        """
        检查碰撞（自碰撞和与地面的碰撞）
        使用精细的接触点位置比较来过滤初始状态下的碰撞
        
        Args:
            distance_threshold: 碰撞距离阈值（米）。
        
        Returns:
            True if collision detected, False otherwise
        """
        # 执行碰撞检测
        p.performCollisionDetection()
        
        # 检查自碰撞（精细过滤：只忽略位置相近的初始接触点）
        contact_points = p.getContactPoints(self.robot_id, self.robot_id)
        for contact in contact_points:
            contact_distance = contact[8]  # 接触点距离
            if contact_distance < -distance_threshold:  # 负值表示穿透深度
                linkA = contact[3]
                linkB = contact[4]
                positionOnA = np.array(contact[5])
                positionOnB = np.array(contact[6])
                pair = tuple(sorted([linkA, linkB]))
                
                # 检查是否为初始状态下位置相近的接触点
                is_initial_contact = self._is_near_initial_contact(
                    pair, positionOnA, positionOnB
                )
                
                if is_initial_contact:
                    continue
                    
                return True
        
        # 检查与地面的碰撞
        contact_points = p.getContactPoints(self.robot_id, self.plane_id)
        for contact in contact_points:
            contact_distance = contact[8]  # 接触点距离
            if contact_distance < -distance_threshold:  # 负值表示穿透深度
                return True
        
        return False
    
    def _is_near_initial_contact(self, link_pair, position_on_A, position_on_B):
        """
        判断当前接触点是否接近初始状态下的接触点
        
        Args:
            link_pair: 连杆对 (linkA, linkB)
            position_on_A: 当前在linkA上的接触点位置
            position_on_B: 当前在linkB上的接触点位置
        
        Returns:
            bool: 如果接近初始接触点则返回True
        """
        for initial_contact in self._initial_contact_points:
            if initial_contact['link_pair'] != link_pair:
                continue
            
            # 计算当前接触点与初始接触点的距离
            dist_A = np.linalg.norm(position_on_A - initial_contact['position_on_A'])
            dist_B = np.linalg.norm(position_on_B - initial_contact['position_on_B'])
            
            # 如果两个接触点都接近初始位置，则认为是初始接触
            if (dist_A < self.initial_contact_position_tolerance and 
                dist_B < self.initial_contact_position_tolerance):
                return True
        
        return False
    
    def forward_kinematics_and_collision_check(self, joint_angles):
        """
        执行FK并检查碰撞
        
        Returns:
            position, orientation_quat, is_collision_free
        """
        self.set_joint_angles(joint_angles)
        position, orientation_quat = self.get_end_effector_pose()
        is_collision_free = not self.check_collision(distance_threshold=self.collision_tolerance)
        
        return position, orientation_quat, is_collision_free
    
    def __del__(self):
        """清理PyBullet连接和临时文件"""
        try:
            p.disconnect(self.physics_client)
        except:
            pass
        
        # 清理临时URDF文件
        try:
            if hasattr(self, '_temp_urdf_path'):
                import os
                os.unlink(self._temp_urdf_path)
        except:
            pass


def generate_pose_database(num_samples=5000, output_path=None, enable_collision_check=True, urdf_path=None):
    """
    生成可达位姿数据库（使用真实URDF模型）
    
    Args:
        num_samples: 要生成的样本数量
        output_path: 输出文件路径
        enable_collision_check: 是否启用碰撞检测
        urdf_path: URDF文件路径
    """
    print(f"开始生成 {num_samples} 个可达位姿...")
    print(f"  URDF路径: {urdf_path}")
    if enable_collision_check:
        print("  ✓ 碰撞检测已启用（PyBullet物理引擎）")
    else:
        print("  ✗ 碰撞检测已禁用")
    
    positions = []
    orientations_quat = []
    joint_configs = []  # 保存关节配置用于调试
    
    # 初始化PyBullet模拟器
    if urdf_path is None:
        # 使用默认URDF路径
        script_dir = Path(__file__).parent
        urdf_path = script_dir / "source/ARM/data/Robots/arm_t/urdf/urdf/ARM_T.urdf"
    
    simulator = URDFRobotSimulator(urdf_path, end_effector_link_name="link6")
    
    # ARM-T的关节限位（弧度）
    joint_limits = [
        (-np.pi, np.pi),      # joint1
        (-np.pi, np.pi),      # joint2
        (-np.pi, np.pi),      # joint3
        (-np.pi, np.pi),      # joint4
        (-np.pi, np.pi),      # joint5
        (-np.pi, np.pi),      # joint6
    ]
    
    # 统计信息
    collision_count = 0
    workspace_filter_count = 0
    
    # 随机采样关节空间
    attempts = 0
    max_attempts = num_samples * 10  # 最大尝试次数
    
    while len(positions) < num_samples and attempts < max_attempts:
        attempts += 1
        
        if attempts % 1000 == 0:
            print(f"  进度: {len(positions)}/{num_samples} (尝试: {attempts}, "
                  f"碰撞: {collision_count}, 工作空间过滤: {workspace_filter_count})")
        
        # 在关节限位内随机采样
        joint_angles = np.array([
            np.random.uniform(low, high) 
            for low, high in joint_limits
        ])
        
        # 使用PyBullet进行FK和碰撞检测
        pos, quat, is_collision_free = simulator.forward_kinematics_and_collision_check(joint_angles)
        
        # 碰撞检测
        if enable_collision_check and not is_collision_free:
            collision_count += 1
            continue
        
        # 工作空间过滤（基于现有数据库的实际范围，稍微放宽）
        # 现有数据库范围: X[0.15-0.30], Y[-0.15-0.15], Z[0.15-0.35]
        # 放宽约20%以获得更多样本并覆盖边缘情况
        if not (0.10 < pos[0] < 0.35 and   # X轴: 0.10m到0.35m（前后范围）
                -0.20 < pos[1] < 0.20 and  # Y轴: -0.20m到0.20m（左右范围）
                0.10 < pos[2] < 0.40):     # Z轴: 0.10m到0.40m（高度范围）
            workspace_filter_count += 1
            continue
        
        # 有效位姿，添加到数据库
        positions.append(pos)
        orientations_quat.append(quat)
        joint_configs.append(joint_angles)
    
    # 检查是否生成了足够的位姿
    if len(positions) == 0:
        print("\n❌ 错误：未能生成任何有效位姿！")
        print("可能的原因：")
        print("  1. 碰撞检测过于严格")
        print("  2. 工作空间过滤范围太小")
        print("  3. 关节限位设置不正确")
        print("\n建议：")
        print("  - 尝试禁用碰撞检测: --no_collision_check")
        print("  - 增加样本数量: --num_samples 50000")
        del simulator
        return None
    
    # 转换为numpy数组
    positions = np.array(positions)
    orientations_quat = np.array(orientations_quat)
    joint_configs = np.array(joint_configs)
    
    # 保存数据库
    database = {
        'num_poses': len(positions),
        'positions': positions,
        'orientations_quat': orientations_quat,
        'joint_configs': joint_configs,  # 保存关节配置
        'workspace': {
            'x_range': (positions[:, 0].min(), positions[:, 0].max()),
            'y_range': (positions[:, 1].min(), positions[:, 1].max()),
            'z_range': (positions[:, 2].min(), positions[:, 2].max()),
        },
        'generation_method': 'urdf_fk_with_pybullet_collision' if enable_collision_check else 'urdf_fk_no_collision',
        'urdf_path': str(urdf_path),
        'num_samples_requested': num_samples,
        'collision_check_enabled': enable_collision_check,
        'statistics': {
            'total_attempts': attempts,
            'collision_rejected': collision_count,
            'workspace_filtered': workspace_filter_count,
            'success_rate': len(positions) / attempts if attempts > 0 else 0,
        },
    }
    
    # 清理PyBullet
    del simulator
    
    # 确定输出路径
    if output_path is None:
        output_path = Path(__file__).parent / "source/ARM/arm_t/tasks/reach/reachable_poses_database.pkl"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"\n✓ 数据库已保存到: {output_path}")
    print(f"  有效位姿数量: {database['num_poses']}")
    print(f"  总尝试次数: {database['statistics']['total_attempts']}")
    print(f"  碰撞拒绝: {database['statistics']['collision_rejected']}")
    print(f"  工作空间过滤: {database['statistics']['workspace_filtered']}")
    print(f"  成功率: {database['statistics']['success_rate']*100:.1f}%")
    print(f"  工作空间范围:")
    print(f"    X: [{database['workspace']['x_range'][0]:.3f}, {database['workspace']['x_range'][1]:.3f}]")
    print(f"    Y: [{database['workspace']['y_range'][0]:.3f}, {database['workspace']['y_range'][1]:.3f}]")
    print(f"    Z: [{database['workspace']['z_range'][0]:.3f}, {database['workspace']['z_range'][1]:.3f}]")
    
    return database


def main():
    parser = argparse.ArgumentParser(description="生成ARM-T可达位姿数据库（使用真实URDF模型）")
    parser.add_argument("--num_samples", type=int, default=5000, help="样本数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--urdf", type=str, default=None, help="URDF文件路径")
    parser.add_argument("--no_collision_check", action="store_true", help="禁用碰撞检测")
    
    args = parser.parse_args()
    
    # 确定URDF路径
    if args.urdf:
        urdf_path = Path(args.urdf)
    else:
        # 使用默认路径
        script_dir = Path(__file__).parent
        urdf_path = script_dir / "source/ARM/data/Robots/arm_t/urdf/urdf/ARM_T.urdf"
    
    if not urdf_path.exists():
        print(f"错误: URDF文件不存在: {urdf_path}")
        print("请使用 --urdf 参数指定正确的URDF文件路径")
        return
    
    generate_pose_database(
        num_samples=args.num_samples,
        output_path=args.output,
        enable_collision_check=not args.no_collision_check,
        urdf_path=urdf_path
    )


if __name__ == "__main__":
    main()

