#!/usr/bin/env python3
"""
Mock 硬件模式启动文件
使用 mock_components 进行控制器测试，不需要 Gazebo
适用于快速测试控制器逻辑、MoveIt 配置等
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包共享目录
    panther_description_share = get_package_share_directory('panther_description')
    panther_moveit_config_share = get_package_share_directory('panther_moveit_config')
    
    # URDF xacro 文件路径 - Mock 模式专用
    xacro_file = os.path.join(panther_description_share, 'urdf', 'panther_with_mock.urdf.xacro')
    
    # Mock 模式的控制器配置文件路径
    ros2_controllers_config = os.path.join(
        panther_moveit_config_share,
        'config',
        'ros2_controllers_mock.yaml'
    )
    
    # 生成 URDF - 使用 Mock 硬件（不需要参数，文件已专用）
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), 
        value_type=str
    )
    
    # robot_state_publisher 节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': False}  # Mock 模式不使用仿真时间
        ]
    )
    
    # controller_manager 节点 - Mock 模式需要手动启动
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': False},
            ros2_controllers_config
        ]
    )
    
    # 控制器 spawner 节点
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen',
        name='joint_state_broadcaster_spawner'
    )
    
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'panther_arm_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen',
        name='panther_arm_controller_spawner'
    )
    
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen',
        name='gripper_controller_spawner'
    )
    
    # 事件处理：按顺序加载控制器
    # 等待 joint_state_broadcaster 加载完成后加载机械臂控制器
    load_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner]
        )
    )
    
    # 等待机械臂控制器加载完成后加载夹爪控制器
    load_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner]
        )
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        controller_manager,
        joint_state_broadcaster_spawner,
        load_arm_controller,
        load_gripper_controller,
    ])

