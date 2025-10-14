#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包共享目录
    panther_description_share = get_package_share_directory('panther_description')
    
    # URDF 文件路径
    urdf_file = os.path.join(panther_description_share, 'urdf', 'panther_description.urdf')
    
    # 读取 URDF 文件
    with open(urdf_file, 'r') as file:
        robot_description = file.read()
    
    # RViz 配置文件
    rviz_config_file = os.path.join(panther_description_share, 'urdf.rviz')
    
    # 参数
    use_gui = LaunchConfiguration('use_gui', default='true')
    
    # 声明参数
    declare_use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='是否启动 joint_state_publisher_gui'
    )
    
    # robot_state_publisher 节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )
    
    # joint_state_publisher 节点（无 GUI）
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(use_gui)
    )
    
    # joint_state_publisher_gui 节点（带 GUI）
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(use_gui)
    )
    
    # RViz 节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else []
    )
    
    return LaunchDescription([
        declare_use_gui,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ])

