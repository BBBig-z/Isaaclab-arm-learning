#!/usr/bin/env python3
"""
Gazebo 仿真模式启动文件
使用 gazebo_ros2_control 插件在 Gazebo 中仿真机器人
controller_manager 由 gazebo_ros2_control 插件自动管理
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 清理残留的 Gazebo 进程和缓存（避免 "Entity already exists" 错误）
    try:
        import time
        # 强制终止所有 Gazebo 相关进程
        subprocess.run(['pkill', '-9', 'gzserver'], check=False, capture_output=True)
        subprocess.run(['pkill', '-9', 'gzclient'], check=False, capture_output=True)
        subprocess.run(['pkill', '-9', 'gz'], check=False, capture_output=True)
        
        # 等待进程完全关闭
        time.sleep(1.0)
        
        # 清理 Gazebo 缓存和日志
        home_dir = os.path.expanduser('~')
        gazebo_paths = [
            os.path.join(home_dir, '.gazebo', 'log'),
            os.path.join(home_dir, '.gazebo', 'server-*'),
            '/tmp/gazebo-*'
        ]
        
        for path in gazebo_paths:
            if '*' in path:
                # 使用 shell 通配符删除
                subprocess.run(f'rm -rf {path}', shell=True, check=False, capture_output=True)
            elif os.path.exists(path):
                subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
        
        print("[INFO] 已清理残留的 Gazebo 进程和缓存")
    except Exception as e:
        print(f"[WARN] 清理 Gazebo 时出错: {e}")
    
    # 获取包共享目录
    panther_description_share = get_package_share_directory('panther_description')
    panther_moveit_config_share = get_package_share_directory('panther_moveit_config')
    
    # URDF xacro 文件路径 - Gazebo 模式专用
    xacro_file = os.path.join(panther_description_share, 'urdf', 'panther_with_gazebo.urdf.xacro')
    
    # Gazebo 模式的控制器配置文件路径
    ros2_controllers_config = os.path.join(
        panther_moveit_config_share,
        'config',
        'ros2_controllers_gazebo.yaml'
    )
    
    # 生成 URDF - 使用 Gazebo 硬件接口
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), 
        value_type=str
    )
    
    # 声明启动参数
    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='是否启动 Gazebo GUI'
    )
    declare_x = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='机器人在 Gazebo 中的 X 坐标'
    )
    declare_y = DeclareLaunchArgument(
        'y',
        default_value='0.0',
        description='机器人在 Gazebo 中的 Y 坐标'
    )
    declare_z = DeclareLaunchArgument(
        'z',
        default_value='0.1',
        description='机器人在 Gazebo 中的 Z 坐标'
    )
    
    # robot_state_publisher 节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description
        }]
    )
    
    # 启动 Gazebo 仿真环境
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': 'true',  # 启用详细输出以调试崩溃问题
            'pause': 'false',
            'gui': LaunchConfiguration('gui')
        }.items()
    )
    
    # Spawn 机器人到 Gazebo
    # gazebo_ros2_control 插件会在机器人生成时自动启动 controller_manager
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_panther',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'panther',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-timeout', '120.0'  # 增加超时时间
        ]
    )
    
    # 延迟启动 spawn_entity，确保 Gazebo 完全启动
    delayed_spawn_entity = TimerAction(
        period=3.0,  # 等待 2 秒让 Gazebo 完全启动
        actions=[spawn_entity]
    )
    
    # 控制器 spawner 节点
    # 注意：不需要传递 --param-file，因为参数已通过 gazebo_ros2_control 插件加载
    # 但由于插件参数加载有问题，我们仍然在 spawner 中传递配置文件
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '50',
            '--param-file', ros2_controllers_config
        ],
        output='screen',
        name='joint_state_broadcaster_spawner',
        parameters=[{'use_sim_time': True}]
    )
    
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'panther_arm_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '50',
            '--param-file', ros2_controllers_config
        ],
        output='screen',
        name='panther_arm_controller_spawner',
        parameters=[{'use_sim_time': True}]
    )
    
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '50',
            '--param-file', ros2_controllers_config
        ],
        output='screen',
        name='gripper_controller_spawner',
        parameters=[{'use_sim_time': True}]
    )
    
    # 事件处理：等待机器人生成后再加载控制器
    # 延迟启动确保 gazebo_ros2_control 插件完全初始化
    load_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[
                TimerAction(
                    period=3.0,  # 等待 3 秒确保 controller_manager 就绪
                    actions=[joint_state_broadcaster_spawner]
                )
            ]
        )
    )
    
    # 按顺序加载控制器
    load_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner]
        )
    )
    
    load_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner]
        )
    )
    
    return LaunchDescription([
        declare_gui,
        declare_x,
        declare_y,
        declare_z,
        robot_state_publisher_node,
        gazebo,
        delayed_spawn_entity,  # 使用延迟启动的 spawn
        load_joint_state_broadcaster,
        load_arm_controller,
        load_gripper_controller,
    ])
