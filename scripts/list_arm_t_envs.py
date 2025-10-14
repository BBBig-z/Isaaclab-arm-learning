# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all available ARM-T environments.

This script lists all registered ARM-T environments and their configurations.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
try:
    import arm_t.tasks  # noqa: F401
except ImportError as e:
    print(f"错误: 无法导入 arm_t.tasks 模块")
    print(f"详细信息: {e}")
    print("\n请确保:")
    print("1. 在项目根目录运行此脚本")
    print("2. Python 路径包含 source/ARM")
    print("\n尝试运行:")
    print("  export PYTHONPATH=\"${PYTHONPATH}:$(pwd)/source/ARM\"")
    print("  python scripts/list_arm_t_envs.py")
    simulation_app.close()
    exit(1)
    
from prettytable import PrettyTable


def main():
    """Print all ARM-T environments."""
    # print all the available environments
    table = PrettyTable(["编号", "任务名称", "入口点", "配置文件"])
    table.title = "ARM-T 可用环境列表"
    # set alignment of table columns
    table.align["任务名称"] = "l"
    table.align["入口点"] = "l"
    table.align["配置文件"] = "l"

    # count of environments
    index = 0
    # acquire all ARM-T environment names
    for task_spec in gym.registry.values():
        if "ARM-T" in task_spec.id:
            # add details to table
            table.add_row([
                index + 1,
                task_spec.id,
                task_spec.entry_point,
                task_spec.kwargs.get("env_cfg_entry_point", "N/A")
            ])
            # increment count
            index += 1

    print(table)
    print(f"\n总共找到 {index} 个 ARM-T 环境")
    
    # Print usage examples
    print("\n" + "="*80)
    print("训练示例:")
    print("="*80)
    print("\n1. 使用关节位置控制训练:")
    print("   python scripts/rsl_rl/train.py --task ARM-T-Reach-v0")
    print("\n2. 使用 IK 控制训练:")
    print("   python scripts/rsl_rl/train.py --task ARM-T-Reach-IK-v0")
    print("\n3. 测试已训练模型:")
    print("   python scripts/rsl_rl/play.py --task ARM-T-Reach-Play-v0 --checkpoint /path/to/model.pt")
    print("\n4. 使用 headless 模式训练:")
    print("   python scripts/rsl_rl/train.py --task ARM-T-Reach-v0 --headless")
    print("\n5. 调整环境数量:")
    print("   python scripts/rsl_rl/train.py --task ARM-T-Reach-v0 --num_envs 2048")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()

