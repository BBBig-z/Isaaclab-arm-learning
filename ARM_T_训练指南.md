# ARM-T 六自由度机械臂强化学习训练指南

本指南介绍如何使用 Isaac Lab 训练 ARM-T 六自由度机械臂完成操作任务。

## 📋 目录

- [环境概述](#环境概述)
- [快速开始](#快速开始)
- [训练命令](#训练命令)
- [配置说明](#配置说明)
- [可达位姿数据库](#可达位姿数据库)
- [常见问题](#常见问题)

---

## 🤖 环境概述

### ARM-T 机械臂特性

- **自由度**: 6 个旋转关节 (joint1-joint6)
- **末端执行器**: link6（TCP）
- **夹爪**: 双指平行夹爪 (gripper_1_joint, gripper_2_joint)
- **控制方式**: 
  - 关节位置直接控制
  - 差分逆运动学（IK）控制
- **RL算法**: PPO (Proximal Policy Optimization)

### 已注册的训练环境

| 环境 ID | 任务 | 控制方式 | 描述 |
|---------|------|----------|------|
| `ARM-T-Reach-v0` | Reach | 关节位置控制 | 末端执行器到达目标位姿 |
| `ARM-T-Reach-IK-v0` | Reach | 逆运动学控制 | 末端执行器到达目标位姿（IK） |
| `ARM-T-Lift-Cube-v0` | Lift | 关节位置控制 | 抓取立方体到目标位置 |
| `ARM-T-Lift-Cube-IK-v0` | Lift | 逆运动学控制 | 抓取立方体到目标位置（IK） |

**测试环境**: 将 `-v0` 改为 `-Play-v0`（如 `ARM-T-Reach-Play-v0`）

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活 Isaac Lab conda 环境
conda activate isaaclab_env

# 进入项目目录
cd /home/y/works/rl/isaac_so_arm101

# 设置 Python 路径（确保能找到 arm_t 模块）
export PYTHONPATH="${PWD}/source/ARM:${PYTHONPATH}"
```

### 2. 生成可达位姿数据库（首次使用）

Reach 任务使用预计算的可达位姿数据库，确保所有目标都是 100% 可达的。

```bash
# 生成 5000 个可达位姿（启用碰撞检测）
python generate_reachable_poses.py --num_samples 5000

# 分析生成的数据库
python analyze_pose_database.py
```

**数据库特性**:
- ✅ 使用真实 URDF 模型进行前向运动学（FK）
- ✅ PyBullet 物理引擎进行精细碰撞检测（cm 级）
- ✅ 忽略机械臂初始状态下的固有碰撞
- ✅ 工作空间过滤（X: 0.10-0.35m, Y: -0.20-0.20m, Z: 0.10-0.40m）

### 3. 开始训练

#### 使用便捷脚本（推荐）

**标准训练（TensorBoard日志）**:
```bash
# Reach 任务（关节控制）
./train_arm_t.sh reach --headless

# Reach 任务（IK 控制）
./train_arm_t.sh reach-ik --headless

# Lift 任务（关节控制）
./train_arm_t.sh lift --headless

# Lift 任务（IK 控制）
./train_arm_t.sh lift-ik --headless
```

**WandB训练（超参数搜索和云端日志）**:
```bash
# Reach 任务（关节控制）
./train_arm_t_wandb.sh reach --headless

# Reach 任务（IK 控制）
./train_arm_t_wandb.sh reach-ik --headless

# Lift 任务（关节控制）
./train_arm_t_wandb.sh lift --headless

# Lift 任务（IK 控制）
./train_arm_t_wandb.sh lift-ik --headless
```

#### 直接调用训练脚本

```bash
# Reach 任务
python scripts/rsl_rl/train.py \
    --task ARM-T-Reach-v0 \
    --headless \
    --num_envs 2048

# Lift 任务
python scripts/rsl_rl/train.py \
    --task ARM-T-Lift-Cube-v0 \
    --headless \
    --num_envs 2048
```

### 4. 测试训练好的模型

```bash
# 测试 Reach 模型
python scripts/rsl_rl/play.py \
    --task ARM-T-Reach-Play-v0 \
    --checkpoint logs/rsl_rl/arm_t_reach/*/model_*.pt \
    --num_envs 50

# 测试 Lift 模型（关节控制）
python scripts/rsl_rl/play.py \
    --task ARM-T-Lift-Cube-Play-v0 \
    --checkpoint logs/rsl_rl/arm_t_lift/*/model_*.pt \
    --num_envs 50

# 测试 Lift 模型（IK控制）
python scripts/rsl_rl/play.py \
    --task ARM-T-Lift-Cube-IK-Play-v0 \
    --checkpoint logs/rsl_rl/arm_t_lift/*/model_*.pt \
    --num_envs 50
```

---

## 📝 训练命令详解

### 基础训练参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--task` | 任务环境 ID | 必填 | `ARM-T-Reach-v0` |
| `--headless` | 无图形界面模式 | False | - |
| `--num_envs` | 并行环境数量 | 2048 | 4096 |
| `--max_iterations` | 最大训练迭代次数 | 1000 | 2000 |
| `--seed` | 随机种子 | 42 | 123 |

### Headless 模式（推荐用于训练）

无图形界面，训练速度更快，节省显存：

```bash
python scripts/rsl_rl/train.py \
    --task ARM-T-Reach-v0 \
    --headless \
    --num_envs 4096
```

### 调整环境数量

```bash
# 较少环境（节省显存）
python scripts/rsl_rl/train.py --task ARM-T-Reach-v0 --num_envs 1024

# 较多环境（更快训练，需要更多显存）
python scripts/rsl_rl/train.py --task ARM-T-Reach-v0 --num_envs 8192
```

### 从检查点继续训练

```bash
python scripts/rsl_rl/train.py \
    --task ARM-T-Reach-v0 \
    --resume \
    --load_run 2025-10-14_20-32-24
```

### 录制训练视频

```bash
python scripts/rsl_rl/train.py \
    --task ARM-T-Reach-v0 \
    --video \
    --video_interval 500 \
    --video_length 200
```

---

## ⚙️ 配置说明

### 环境配置文件

#### Reach 任务

- **基础配置**: `source/ARM/arm_t/tasks/reach/reach_env_cfg.py`
- **关节控制**: `source/ARM/arm_t/tasks/reach/joint_pos_env_cfg.py`
- **IK 控制**: `source/ARM/arm_t/tasks/reach/ik_rel_env_cfg.py`

**关键参数**:
```python
num_envs = 2048              # 并行环境数量
episode_length_s = 20.0      # 每个回合时长（秒）
decimation = 2               # 动作频率
dt = 1.0 / 60.0             # 仿真时间步长 (60Hz)
```

#### Lift 任务

- **基础配置**: `source/ARM/arm_t/tasks/lift/lift_env_cfg.py`
- **关节控制**: `source/ARM/arm_t/tasks/lift/joint_pos_env_cfg.py`
- **IK 控制**: `source/ARM/arm_t/tasks/lift/ik_rel_env_cfg.py`

**关键参数**:
```python
num_envs = 2048              # 并行环境数量
episode_length_s = 5.0       # 每个回合时长（秒）
MIN_HEIGHT = 0.1             # 最小抬起高度（米）
```

**物体配置**:
- 立方体：DexCube（缩放0.8x）
- 初始位置：机器人前方15cm，高度5.5cm
- 目标范围：X(0.15-0.35m), Y(-0.15-0.15m), Z(0.20-0.35m)

### 机器人配置

文件: `source/ARM/arm_t/robots/arm_t.py`

#### 关节刚度（Stiffness）

```python
stiffness = {
    "joint1": 200.0,  # 基座旋转
    "joint2": 180.0,  # 肩部
    "joint3": 150.0,  # 肘部
    "joint4": 100.0,  # 腕部关节 1
    "joint5": 80.0,   # 腕部关节 2
    "joint6": 60.0,   # 腕部关节 3
}
```

#### 阻尼（Damping）

```python
damping = {
    "joint1": 80.0,
    "joint2": 70.0,
    "joint3": 60.0,
    "joint4": 40.0,
    "joint5": 30.0,
    "joint6": 25.0,
}
```

#### 高 PD 增益配置（IK 控制专用）

```python
ARM_T_HIGH_PD_CFG = ARM_T_CFG.copy()
ARM_T_HIGH_PD_CFG.actuators["arm"].stiffness = {
    "joint1": 500.0,
    "joint2": 450.0,
    "joint3": 400.0,
    "joint4": 350.0,
    "joint5": 300.0,
    "joint6": 250.0,
}
```

### PPO 算法配置

文件: `source/ARM/arm_t/tasks/*/agents/rsl_rl_ppo_cfg.py`

**Reach 任务超参数**:
```python
runner = RslRlPpoActorCriticCfg.RslRlPpoAlgorithmCfg(
    num_steps_per_env = 24
    max_iterations = 1000
    learning_rate = 1.0e-3
    gamma = 0.99
    lam = 0.95
    num_learning_epochs = 5
    num_mini_batches = 4
)

policy = RslRlPpoActorCriticCfg.RslRlPpoActorCriticCfg(
    actor_hidden_dims = [128, 128]
    critic_hidden_dims = [128, 128]
    activation = "elu"
)
```

**Lift 任务超参数**:
```python
runner = RslRlPpoActorCriticCfg.RslRlPpoAlgorithmCfg(
    num_steps_per_env = 24
    max_iterations = 1500
    learning_rate = 3.0e-4
    gamma = 0.99
    lam = 0.95
    num_learning_epochs = 8
    num_mini_batches = 64
    init_noise_std = 0.4        # 更高的初始噪声鼓励探索
)

policy = RslRlPpoActorCriticCfg.RslRlPpoActorCriticCfg(
    actor_hidden_dims = [256, 128, 64]  # 更深的网络处理复杂任务
    critic_hidden_dims = [256, 128, 64]
    activation = "relu"
)
```

---

## 📊 训练监控

### TensorBoard

```bash
# 启动 TensorBoard（Reach 任务）
tensorboard --logdir logs/rsl_rl/arm_t_reach

# 启动 TensorBoard（Lift 任务）
tensorboard --logdir logs/rsl_rl/arm_t_lift

# 在浏览器中打开
# http://localhost:6006
```

### 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| `ep_rew_mean` | 平均回合奖励 | 越高越好 |
| `ep_len_mean` | 平均回合长度 | 稳定 |
| `value_loss` | 价值函数损失 | 下降趋势 |
| `policy_loss` | 策略损失 | 下降趋势 |
| `learning_rate` | 学习率 | 稳定/衰减 |

### 日志文件结构

```
logs/rsl_rl/arm_t_reach/
├── 2025-10-14_20-32-24/
│   ├── events.out.tfevents.*    # TensorBoard 日志
│   ├── model_0.pt                # 初始模型
│   ├── model_500.pt              # 第 500 次迭代
│   ├── model_1000.pt             # 最终模型
│   ├── git/                      # Git 差异记录
│   └── params/
│       ├── env.pkl               # 环境配置（序列化）
│       ├── env.yaml              # 环境配置（可读）
│       ├── agent.pkl             # 智能体配置（序列化）
│       └── agent.yaml            # 智能体配置（可读）
```

---

## 🎯 可达位姿数据库

### 生成数据库

```bash
# 基础用法（5000 个位姿，启用碰撞检测）
python generate_reachable_poses.py --num_samples 5000

# 生成更多位姿（增加多样性）
python generate_reachable_poses.py --num_samples 10000

# 使用自定义 URDF 文件
python generate_reachable_poses.py \
    --urdf source/ARM/data/Robots/arm_t/urdf/urdf/ARM_T_fixed.urdf \
    --num_samples 5000

# 禁用碰撞检测（调试用）
python generate_reachable_poses.py --num_samples 1000 --no_collision_check
```

### 碰撞检测特性

**精细碰撞过滤**（基于接触点位置）:
- ✅ **cm 级精度**: 只忽略位置相近（5cm 容差）的初始接触点
- ✅ **智能过滤**: 记录初始姿态下的接触点位置，运动中新位置的碰撞会被检测
- ✅ **自碰撞检测**: 检测机械臂各连杆之间的碰撞
- ✅ **环境碰撞检测**: 检测与地面的碰撞

**初始姿态**（用于记录固有碰撞）:
```python
initial_angles = [
    0.0,                    # joint1: 0°
    np.deg2rad(25.0),      # joint2: 25°
    np.deg2rad(17.0),      # joint3: 17°
    np.deg2rad(-35.0),     # joint4: -35°
    0.0,                    # joint5: 0°
    0.0,                    # joint6: 0°
]
```

### 分析数据库

```bash
# 查看数据库统计信息和可视化
python analyze_pose_database.py

# 输出示例：
# 📊 数据库统计
#   位姿数量: 5000
#   生成方法: urdf_fk_with_pybullet_collision
#   
# 📐 工作空间范围
#   X: [0.123, 0.348] m
#   Y: [-0.198, 0.199] m
#   Z: [0.102, 0.397] m
#   
# 📈 生成统计
#   总尝试次数: 15234
#   碰撞拒绝: 8432
#   工作空间过滤: 1802
#   成功率: 32.8%
```

### 数据库文件位置

```
source/ARM/arm_t/tasks/reach/reachable_poses_database.pkl
```

**重要提示**：Lift任务不需要可达位姿数据库，物体位置是随机生成的。

数据库内容:
```python
{
    'num_poses': 5000,
    'positions': np.array([[x, y, z], ...]),        # (5000, 3)
    'orientations_quat': np.array([[w,x,y,z], ...]), # (5000, 4)
    'joint_configs': np.array([...]),                # (5000, 6)
    'workspace': {
        'x_range': (0.10, 0.35),
        'y_range': (-0.20, 0.20),
        'z_range': (0.10, 0.40),
    },
    'generation_method': 'urdf_fk_with_pybullet_collision',
    'statistics': {...}
}
```

---

## 🌐 Weights & Biases 集成

### 使用 WandB 训练和超参数搜索

#### 方式一：直接WandB训练（推荐用于单次训练）

```bash
# Reach任务（关节控制）
./train_arm_t_wandb.sh reach --headless --num_envs 2048

# Reach任务（IK控制）
./train_arm_t_wandb.sh reach-ik --headless --num_envs 2048

# Lift任务（关节控制）
./train_arm_t_wandb.sh lift --headless --num_envs 2048

# Lift任务（IK控制）
./train_arm_t_wandb.sh lift-ik --headless --num_envs 2048
```

#### 方式二：WandB Sweep超参数搜索（推荐用于优化）

**步骤1：创建Sweep**

```bash
# Reach任务超参数搜索
cd source/ARM/arm_t/tasks/reach/agents
wandb sweep wandb_sweep_rsl_rl_ppo_reach.yaml
# 输出：wandb: Created sweep with ID: abc123xyz
# 输出：wandb: View sweep at: https://wandb.ai/your-username/project/sweeps/abc123xyz

# Reach-IK任务超参数搜索
wandb sweep wandb_sweep_rsl_rl_ppo_reach_ik.yaml

# Lift任务超参数搜索
cd ../lift/agents
wandb sweep wandb_sweep_rsl_rl_ppo_lift.yaml

# Lift-IK任务超参数搜索
wandb sweep wandb_sweep_rsl_rl_ppo_lift_ik.yaml
```

**步骤2：启动Agent**

```bash
# 单个Agent
wandb agent <your-username/project-name/sweep_id>

# 多个并行Agent（充分利用多GPU）
# 终端1
wandb agent <sweep_id>
# 终端2
wandb agent <sweep_id>
# 终端3
wandb agent <sweep_id>
```

**Sweep训练时间估算**：

| 任务 | 配置数量 | 单次训练时间 | 串行总时间 | 3 Agent并行 |
|------|---------|-------------|----------|-----------|
| Reach | 27 | ~3小时 | ~81小时 | ~27小时 |
| Reach-IK | 27 | ~2.5小时 | ~67.5小时 | ~22.5小时 |
| Lift | 27 | ~8小时 | ~216小时 | ~72小时 |
| Lift-IK | 27 | ~8小时 | ~216小时 | ~72小时 |

**提示**：可以在WandB网页界面实时查看所有配置的训练进度和结果对比。

### WandB Sweep 配置详解

所有任务都有对应的sweep配置文件：

| 任务 | Sweep配置文件 | 训练脚本 |
|------|--------------|---------|
| Reach（关节） | `source/ARM/arm_t/tasks/reach/agents/wandb_sweep_rsl_rl_ppo_reach.yaml` | `scripts/rsl_rl/train_wandb_reach.py` |
| Reach（IK） | `source/ARM/arm_t/tasks/reach/agents/wandb_sweep_rsl_rl_ppo_reach_ik.yaml` | `scripts/rsl_rl/train_wandb_reach_ik.py` |
| Lift（关节） | `source/ARM/arm_t/tasks/lift/agents/wandb_sweep_rsl_rl_ppo_lift.yaml` | `scripts/rsl_rl/train_wandb_lift.py` |
| Lift（IK） | `source/ARM/arm_t/tasks/lift/agents/wandb_sweep_rsl_rl_ppo_lift_ik.yaml` | `scripts/rsl_rl/train_wandb_lift.py` |

#### Reach任务Sweep配置

**超参数搜索空间** (`wandb_sweep_rsl_rl_ppo_reach.yaml`):

```yaml
program: scripts/rsl_rl/train_wandb_reach.py
method: grid  # 网格搜索（遍历所有组合）
metric:
  goal: maximize
  name: train/episode_reward_mean

parameters:
  # 探索噪声（3个值）
  init_noise_std:
    values: [0.2, 0.3, 0.15]
  
  # 学习率（3个值）
  learning_rate:
    values: [0.0004, 0.0002, 0.001]
  
  # 熵系数（3个值）
  entropy_coef:
    values: [0.005, 0.01, 0.001]
  
  # 固定参数
  num_envs: 2048
  max_iterations: 1000
  actor_hidden_dims: [128, 128]
  critic_hidden_dims: [128, 128]
  num_learning_epochs: 8
  num_mini_batches: 64
```

**搜索空间大小**：3 × 3 × 3 = **27种配置组合**

#### Lift任务Sweep配置

**超参数搜索空间** (`wandb_sweep_rsl_rl_ppo_lift.yaml`):

```yaml
program: scripts/rsl_rl/train_wandb_lift.py
method: grid
metric:
  goal: maximize
  name: train/episode_reward_mean

parameters:
  # 探索噪声（3个值）
  init_noise_std:
    values: [0.4, 0.3, 0.5]
  
  # 学习率（3个值）
  learning_rate:
    values: [0.0003, 0.0001, 0.001]
  
  # 熵系数（3个值）
  entropy_coef:
    values: [0.01, 0.005, 0.02]
  
  # 固定参数
  num_envs: 2048
  max_iterations: 1500
  actor_hidden_dims: [256, 128, 64]  # 更深的网络
  critic_hidden_dims: [256, 128, 64]
  num_learning_epochs: 8
  num_mini_batches: 512  # 更大的批次
```

**搜索空间大小**：3 × 3 × 3 = **27种配置组合**

#### 超参数说明

| 参数 | Reach任务 | Lift任务 | 作用 |
|------|----------|---------|------|
| `init_noise_std` | 0.15-0.3 | 0.3-0.5 | 初始探索噪声强度 |
| `learning_rate` | 2e-4 ~ 1e-3 | 1e-4 ~ 1e-3 | 学习率 |
| `entropy_coef` | 0.001-0.01 | 0.005-0.02 | 熵正则化系数 |
| `actor_hidden_dims` | [128, 128] | [256, 128, 64] | Actor网络层数 |
| `max_iterations` | 1000 | 1500 | 最大训练迭代次数 |

**关键差异**：
- Lift任务使用更深的网络（3层 vs 2层）
- Lift任务需要更多训练迭代（1500 vs 1000）
- Lift任务使用更高的探索噪声（适应复杂任务）

### WandB Config文件（单次训练）

除了Sweep配置，每个任务还有用于单次WandB训练的配置文件：

| 任务 | Config文件 |
|------|-----------|
| Reach（关节） | `source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach.yaml` |
| Reach（IK） | `source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach_ik.yaml` |
| Lift（关节） | `source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift.yaml` |
| Lift（IK） | `source/ARM/arm_t/tasks/lift/agents/config_rsl_rl_ppo_lift_ik.yaml` |

这些文件包含单次训练的默认超参数，由 `train_wandb_*.py` 脚本读取。

**使用方式**：
```bash
# 使用默认配置文件训练
./train_arm_t_wandb.sh reach --headless

# 或直接调用脚本
python scripts/rsl_rl/train_wandb_reach.py \
    --task ARM-T-Reach-v0 \
    --headless \
    --config source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach.yaml
```

### 查看Sweep结果

**在WandB网页界面**：
1. 访问：`https://wandb.ai/<your-username>/<project-name>/sweeps/<sweep-id>`
2. 查看超参数重要性图表
3. 查看并行坐标图（Parallel Coordinates Plot）
4. 按照reward排序，找到最佳配置
5. 下载最佳模型的checkpoint

**命令行查询**：
```bash
# 查看sweep状态
wandb sweep <sweep-id> --stop  # 停止sweep

# 导出最佳配置
# 在WandB界面选择最佳run，查看其Hyperparameters面板
```

