# ARM-T 六自由度机械臂强化学习项目说明文档

## 一、项目概述

本项目基于Isaac Lab框架，实现了ARM-T六自由度机械臂的强化学习训练系统，支持多种操作任务和控制方式。

### 项目特点
- **仿真平台**：基于 Isaac Lab（NVIDIA Isaac Sim）
- **机器人平台**：ARM-T 6-DOF 机械臂（带双指夹爪）
- **强化学习算法**：PPO（来自 RSL-RL 库）
- **控制方式**：
  - 关节位置直接控制
  - 差分逆运动学（IK）控制
- **支持任务**：
  - **Reach任务**：末端执行器到达目标位姿
  - **Lift任务**：抓取物体并移动到目标位置
- **超参数优化**：使用 Weights & Biases (WandB) 进行自动超参数搜索
- **可达位姿数据库**：基于PyBullet的精确碰撞检测，保证目标100%可达

### 项目亮点
- ✅ 完整的端到端训练pipeline（从环境配置到模型部署）
- ✅ 精细的碰撞检测系统（cm级精度）
- ✅ 模块化的奖励函数设计（位置、姿态、保持奖励分离）
- ✅ 完善的WandB集成（单次训练+Sweep超参数搜索）
- ✅ 双控制模式支持（关节控制+IK控制）
- ✅ 详细的训练文档和配置说明

---

## 二、项目文件结构

### 2.1 根目录文件
```
isaac_so_arm101/
├── README.md                       # 项目简介
├── ARM_T_训练指南.md               # 完整训练指南
├── 项目说明文档.md                 # 本文档
│
├── train_arm_t.sh                 # 标准训练启动脚本
├── train_arm_t_wandb.sh           # WandB训练启动脚本
│
├── generate_reachable_poses.py    # 生成可达位姿数据库
├── analyze_pose_database.py       # 分析位姿数据库
│
├── scripts/                       # 训练和测试脚本
│   ├── rsl_rl/
│   │   ├── train.py              # 标准PPO训练脚本
│   │   ├── play.py               # 模型测试脚本
│   │   ├── train_wandb_reach.py  # Reach任务WandB训练
│   │   ├── train_wandb_reach_ik.py
│   │   └── train_wandb_lift.py   # Lift任务WandB训练
│   └── list_arm_t_envs.py        # 列出所有注册环境
│
├── logs/                          # 训练日志和模型
│   └── rsl_rl/
│       ├── arm_t_reach/
│       └── arm_t_lift/
│
└── source/                        # 源代码
    └── ARM/
        └── arm_t/                 # ARM-T机器人包
            ├── robots/
            │   └── arm_t.py      # 机器人配置（PD增益等）
            ├── tasks/
            │   ├── reach/        # Reach任务
            │   │   ├── reach_env_cfg.py
            │   │   ├── joint_pos_env_cfg.py
            │   │   ├── ik_rel_env_cfg.py
            │   │   ├── mdp/      # MDP组件
            │   │   │   ├── commands.py
            │   │   │   ├── observations.py
            │   │   │   ├── rewards.py
            │   │   │   └── terminations.py
            │   │   └── agents/   # PPO配置和WandB配置
            │   │       ├── rsl_rl_ppo_cfg.py
            │   │       ├── config_rsl_rl_ppo_reach.yaml
            │   │       ├── wandb_sweep_rsl_rl_ppo_reach.yaml
            │   │       └── ...
            │   └── lift/         # Lift任务（结构同reach）
            │       ├── lift_env_cfg.py
            │       ├── joint_pos_env_cfg.py
            │       ├── ik_rel_env_cfg.py
            │       ├── mdp/
            │       └── agents/
            └── data/
                └── Robots/
                    └── arm_t/
                        ├── arm_t.usd  # USD模型
                        └── urdf/      # URDF模型
```

### 2.2 核心组件说明

#### **机器人配置** (`source/ARM/arm_t/robots/arm_t.py`)
- **ARM_T_CFG**：标准配置，启用重力，用于关节控制
  - PD增益：Kp(200→60), Kd(80→25)
- **ARM_T_HIGH_PD_CFG**：高PD增益配置，禁用重力，用于IK控制
  - PD增益：Kp(500→250), Kd(150→75)

#### **环境配置文件**
每个任务包含三层配置：

1. **基础配置** (`*_env_cfg.py`)：
   - 场景配置（机器人、物体、桌子、灯光）
   - MDP组件（动作、观测、奖励、终止、命令）
   - 课程学习策略

2. **关节控制配置** (`joint_pos_env_cfg.py`)：
   - 使用 ARM_T_CFG
   - JointPositionActionCfg（6个关节+夹爪）
   - scale=0.3

3. **IK控制配置** (`ik_rel_env_cfg.py`)：
   - 使用 ARM_T_HIGH_PD_CFG
   - DifferentialInverseKinematicsActionCfg
   - scale=0.5，相对模式

#### **MDP定义文件**
- **commands.py**：目标位姿生成（Reach：从数据库采样；Lift：随机生成）
- **observations.py**：关节状态、TCP位姿、目标位姿、物体位姿
- **rewards.py**：位置追踪、姿态追踪、保持奖励、惩罚项
- **terminations.py**：超时、目标达成、物体掉落（Lift）

---

## 三、训练任务与内容

### 3.1 任务概述

项目实现了两类核心任务，每类任务支持两种控制方式：

| 任务类型 | 任务描述 | 训练难度 |
|---------|---------|---------|
| **Reach** | 控制末端执行器到达目标位姿并保持 | ⭐⭐ 中等 |
| **Lift** | 抓取物体并移动到目标位置 | ⭐⭐⭐ 困难 |

| 控制方式 | 特点 | 适用场景 |
|---------|------|---------|
| **关节控制** | 直接控制6个关节角度 | 动作空间小，训练稳定 |
| **IK控制** | 控制末端执行器位姿增量 | 更接近任务空间，易于理解 |

### 3.2 已注册的环境

所有环境在 `source/ARM/arm_t/tasks/*/__ init__.py` 中注册：

#### Reach任务
| 环境ID | 控制方式 | 用途 |
|--------|---------|------|
| `ARM-T-Reach-v0` | 关节控制 | 训练 |
| `ARM-T-Reach-Play-v0` | 关节控制 | 测试/演示 |
| `ARM-T-Reach-IK-v0` | IK控制 | 训练 |
| `ARM-T-Reach-IK-Play-v0` | IK控制 | 测试/演示 |

#### Lift任务
| 环境ID | 控制方式 | 用途 |
|--------|---------|------|
| `ARM-T-Lift-Cube-v0` | 关节控制 | 训练 |
| `ARM-T-Lift-Cube-Play-v0` | 关节控制 | 测试/演示 |
| `ARM-T-Lift-Cube-IK-v0` | IK控制 | 训练 |
| `ARM-T-Lift-Cube-IK-Play-v0` | IK控制 | 测试/演示 |

### 3.3 动作空间

#### Reach任务 - 关节控制
```python
JointPositionActionCfg(
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    scale=0.3,  # 动作缩放因子
    use_default_offset=True,
)
```
- **动作维度**：6维（6个关节角度）
- **动作范围**：每个维度 [-1, 1]，映射到关节空间的增量

#### Reach任务 - IK控制
```python
DifferentialInverseKinematicsActionCfg(
    body_name="link6",
    command_type="pose",
    use_relative_mode=True,  # 相对模式
    ik_method="dls",         # 阻尼最小二乘法
    scale=0.5,
)
```
- **动作维度**：6维 (dx, dy, dz, droll, dpitch, dyaw)
- **特点**：输出相对于当前TCP的位姿增量

#### Lift任务 - 关节控制
```python
# 机械臂动作
JointPositionActionCfg(joint_names=[6个关节], scale=0.3)
# 夹爪动作
BinaryJointPositionActionCfg(
    joint_names=["gripper_1_joint", "gripper_2_joint"],
    open_command_expr={...},   # 打开：0.0
    close_command_expr={...},  # 关闭：0.04
)
```
- **动作维度**：7维（6个关节 + 1个夹爪二元控制）

#### Lift任务 - IK控制
- **动作维度**：7维（6维位姿增量 + 1维夹爪）

### 3.4 观测空间

#### Reach任务观测（17维）
| 观测项 | 维度 | 说明 |
|--------|------|------|
| joint_pos | 6 | 关节位置（归一化） |
| joint_vel | 6 | 关节速度 |
| target_pose | 3 | 目标位置（机器人坐标系） |
| target_quat | 2 | 目标方向（压缩四元数，去除w和z） |

#### Lift任务观测（约30维）
| 观测项 | 维度 | 说明 |
|--------|------|------|
| gripper_joint_pos | 2 | 夹爪关节位置 |
| tcp_pose | 7 | TCP位姿（位置3 + 四元数4）|
| object_position | 3 | 物体位置（机器人坐标系） |
| target_object_position | 3 | 目标物体位置 |
| actions | ~12 | 上一步动作 |

---

## 四、奖励函数设计

### 4.1 奖励项汇总
定义在 `gym_env/env/lift_cube_env_cfg.py` 的 `RewardsCfg` 类中：

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| reaching_object | 1.0 | `object_ee_distance` | 末端执行器接近物体 |
| lifting_object | 15.0 | `object_is_lifted` | 物体举起超过最小高度 |
| object_goal_tracking | 16.0 | `object_goal_distance` (std=0.3) | 物体接近目标位置（粗粒度）|
| object_goal_tracking_fine_grained | 5.0 | `object_goal_distance` (std=0.05) | 物体接近目标位置（细粒度）|
| end_effector_orientation_tracking | -6.0 | `orientation_command_error` | 末端方向跟踪误差（惩罚）|
| action_rate | -1e-4（初始）| `action_rate_l2` | 动作变化率（平滑性惩罚）|

### 4.2 关键奖励函数实现

#### (1) 物体-末端执行器距离奖励
```python
def object_ee_distance(env, std=0.1):
    distance = ||cube_pos - ee_pos||
    return 1 - tanh(distance / std)
```

#### (2) 物体举起奖励
```python
def object_is_lifted(env, minimal_height=0.1):
    return 1.0 if object_height > minimal_height else 0.0
```

#### (3) 目标追踪奖励
```python
def object_goal_distance(env, std, minimal_height, command_name):
    distance = ||desired_pos - object_pos||
    reward = (object_height > minimal_height) * (1 - tanh(distance / std))
    return reward
```

#### (4) 方向误差惩罚
```python
def orientation_command_error(env, minimal_height, command_name):
    quat_error = quat_error_magnitude(current_quat, desired_quat)
    return (object_height > minimal_height) * quat_error
```

### 4.3 课程学习
动作率惩罚权重从 `-1e-4` 逐渐增加到 `-1e-1`（10000步内）：
```python
action_rate = CurrTerm(
    func=mdp.modify_reward_weight,
    params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
)
```

---

## 五、模型约束条件

### 5.1 物理约束

#### UR5e 机械臂约束
| 约束类型 | 关节 | 限制值 |
|----------|------|--------|
| **速度限制** | 所有机械臂关节 | 180.0 deg/s |
| | 夹爪 | 1000000.0 |
| **力矩限制** | 所有机械臂关节 | 87.0 Nm |
| | 夹爪 | 200.0 N |
| **刚度** | 所有机械臂关节 | 1000.0 |
| | 夹爪 | 3000.0 |
| **阻尼** | shoulder_pan | 121.66 |
| | shoulder_lift | 183.23 |
| | elbow | 96.54 |
| | wrist_1/2 | 69.83 |
| | wrist_3 | 27.42 |

#### 立方体物理属性
```python
scale=(0.3, 0.3, 1.0)  # 尺寸缩放
max_angular_velocity=1000.0
max_linear_velocity=1000.0
disable_gravity=False
```

### 5.2 任务约束

#### 初始位置
- **机器人基座**：`(0.175, -0.175, 0.0)`
- **立方体初始**：`[0.04, 0.35, 0.055]`
- **立方体随机化范围**：
  - x: [-0.1, 0.1]
  - y: [-0.25, 0.25]
  - z: 0.0（相对于初始高度）

#### 目标位姿范围
```python
pos_x=(0.25, 0.35)
pos_y=(0.3, 0.4)
pos_z=(0.25, 0.35)
roll=(0.0, 0.0)
pitch=(π, π)
yaw=(-π, π)
```
重采样间隔：5秒

### 5.3 终止条件
1. **超时**：5秒（500步，假设decimation=2）
2. **物体掉落**：物体高度 < -0.05m
3. **目标达成**（可选）：物体与目标距离 < 0.02m

### 5.4 仿真约束
- **时间步长**：0.01s (100Hz)
- **控制频率**：50Hz (decimation=2)
- **并行环境数**：4096（训练）/ 4（测试）

---

## 六、训练命令

### 6.1 环境安装
```bash
# 1. 安装Isaac Sim和Isaac Lab
# 参考：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# 2. 安装项目依赖
pip install -r /path/to/requirements.txt
```

### 6.2 使用 Weights & Biases 进行超参数搜索

#### PPO
```bash
source /path/to/venv/bin/activate
cd /path/to/repository
wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube config_sb3_ppo.yaml
wandb agent <sweep_id>  # 运行返回的sweep_id
```

#### TD3
```bash
wandb sweep --project rel_ik_sb3_td3_ur5e_lift_cube config_sb3_td3.yaml
wandb agent <sweep_id>
```

#### DDPG
```bash
wandb sweep --project rel_ik_sb3_ddpg_ur5e_lift_cube config_sb3_ddpg.yaml
wandb agent <sweep_id>
```

### 6.3 使用 WandB 与不使用 WandB 的区别

#### 6.3.1 对比总览

| 特性 | 使用 WandB | 不使用 WandB |
|------|-----------|-------------|
| **训练脚本** | `train_sb3_wandb_ppo.py` | `train_sb3_ppo.py` |
| **日志系统** | WandB云端 + TensorBoard | 本地TensorBoard + CSV |
| **超参数管理** | 自动网格/随机搜索 | 手动指定 |
| **并行实验** | 多agent并行搜索 | 单次运行 |
| **结果对比** | 自动可视化对比 | 需手动对比 |
| **需要联网** | 是 | 否 |
| **训练性能** | **相同** | **相同** |
| **最终模型质量** | **相同（相同超参数下）** | **相同（相同超参数下）** |

#### 6.3.2 核心影响分析

**✅ WandB 不影响的方面：**
- ✓ 训练算法本身（都使用Stable-Baselines3）
- ✓ 环境交互速度
- ✓ 模型收敛性（相同超参数下）
- ✓ 物理仿真过程
- ✓ 奖励函数计算
- ✓ 网络前向/反向传播

**🔄 WandB 影响的方面：**

1. **超参数搜索能力**
   ```python
   # 使用WandB：自动运行36种配置
   wandb sweep config_sb3_ppo.yaml  # 定义搜索空间
   wandb agent <sweep_id>           # 自动执行
   
   # 不使用WandB：需手动修改配置文件，运行36次
   python train_sb3_ppo.py --task UR5e-Lift-Cube-IK  # 第1次
   # 修改 gym_env/env/agents/sb3_ppo_cfg.yaml
   python train_sb3_ppo.py --task UR5e-Lift-Cube-IK  # 第2次
   # ... 重复34次
   ```

2. **日志记录与可视化**
   ```
   使用WandB：
   ├── 实时云端同步
   ├── 多实验自动对比图表
   ├── 超参数重要性分析
   ├── 模型检查点自动上传
   └── 团队协作共享
   
   不使用WandB：
   ├── 本地TensorBoard文件
   ├── CSV文本日志
   ├── 需手动整理对比
   └── 本地存储管理
   ```

3. **实验管理效率**
   | 任务 | WandB时间 | 无WandB时间 | 效率提升 |
   |------|----------|------------|---------|
   | 配置36组实验 | 5分钟 | 3-6小时 | **36倍** |
   | 结果可视化 | 实时 | 30-60分钟 | **即时** |
   | 找到最佳超参数 | 自动排序 | 手动对比 | **10倍+** |

#### 6.3.3 代码层面差异

**使用WandB的训练脚本特点** (`train_sb3_wandb_ppo.py`)：
```python
import wandb

# 1. 初始化WandB运行
run = wandb.init(
    project="rel_ik_sb3_ppo_ur5e_lift_cube",
    config=config,  # 从sweep配置自动读取
    sync_tensorboard=True,  # 自动同步TensorBoard指标
)

# 2. 超参数从WandB配置读取
env_cfg.seed = wandb.config["seed"]
agent = PPO(
    wandb.config["policy"],
    learning_rate=wandb.config.learning_rate,
    batch_size=wandb.config.batch_size,
    # ... 其他超参数从wandb.config动态读取
)

# 3. 使用WandB回调
agent.learn(
    total_timesteps=wandb.config["n_timesteps"],
    callback=WandbCallback(  # 自动记录指标、保存模型
        gradient_save_freq=10000,
        model_save_path=f"models/{run.id}",
    ),
)
```

**不使用WandB的训练脚本特点** (`train_sb3_ppo.py`)：
```python
from stable_baselines3.common.logger import configure

# 1. 从Hydra配置文件读取超参数
@hydra_task_config(args_cli.task, "sb3_ppo_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 2. 超参数从YAML文件读取
    agent_cfg = process_sb3_cfg(agent_cfg)
    
    # 3. 使用本地TensorBoard日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    agent.set_logger(new_logger)
    
    # 4. 使用本地检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_dir,
    )
    
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
```

#### 6.3.4 选择建议

**推荐使用WandB的场景：**
- ✅ 需要进行超参数搜索（寻找最佳配置）
- ✅ 同时运行多个实验对比
- ✅ 团队协作项目
- ✅ 需要详细的实验记录和回溯
- ✅ 有稳定的网络连接

**推荐不使用WandB的场景：**
- ✅ 已知最优超参数，只需训练一次
- ✅ 离线环境或网络受限
- ✅ 对数据隐私有严格要求
- ✅ 快速原型验证
- ✅ 计算资源受限（避免额外的日志上传开销）

#### 6.3.5 性能开销对比

**训练速度影响（4096环境）：**
```
纯训练（无日志）:        100% 基准速度
+ 本地TensorBoard:       ~98% (-2%)
+ WandB（网络良好）:     ~95% (-5%)
+ WandB（网络较差）:     ~90% (-10%)
```

**磁盘/网络开销：**
| 项目 | 无WandB | 有WandB |
|------|---------|---------|
| 本地磁盘占用 | ~2-5 GB | ~1-2 GB（部分上传云端）|
| 网络上传流量 | 0 | ~500 MB - 2 GB |
| 日志查看速度 | 即时（本地）| 稍延迟（需加载云端）|

#### 6.3.6 实际使用示例

**场景1：超参数搜索（推荐WandB）**
```bash
# 一次性配置36组实验
wandb sweep config_sb3_ppo.yaml
# sweep_id: abc123xyz

# 启动3个并行agent（多GPU情况下）
wandb agent abc123xyz  # 终端1
wandb agent abc123xyz  # 终端2
wandb agent abc123xyz  # 终端3

# 等待完成后，在WandB网页查看最佳配置
# 网址：https://wandb.ai/your-project/runs
```

**场景2：单次训练（推荐本地）**
```bash
# 使用已知的最佳超参数进行训练
python3 train_sb3_ppo.py \
    --num_envs 4096 \
    --task UR5e-Lift-Cube-IK \
    --headless \
    --seed 42

# 使用TensorBoard监控
tensorboard --logdir=logs/
```

### 6.4 直接训练（无 WandB）- 详细步骤

**适用场景**：已知最佳超参数，只需单次训练

#### 方法1（推荐）
```bash
source /path/to/venv/bin/activate
cd /path/to/repository
python3 train_sb3_ppo.py --num_envs 4096 --task UR5e-Lift-Cube-IK --headless
```

#### 方法2（使用Isaac Lab启动器）
```bash
source /path/to/venv/bin/activate
cd /path/to/isaac/lab/installation/directory
./isaaclab.sh -p /path/to/repository/train_sb3_ppo.py --num_envs 4096 --task UR5e-Lift-Cube-IK --headless
```

#### 训练参数说明
| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | 任务ID | UR5e-Lift-Cube-IK |
| `--num_envs` | 并行环境数 | 4096 |
| `--headless` | 无界面模式 | - |
| `--seed` | 随机种子 | 42 |
| `--max_iterations` | 最大迭代次数 | 1000 |
| `--checkpoint` | 检查点路径（续训） | ./model.zip |

**注意**：超参数在 `gym_env/env/agents/sb3_ppo_cfg.yaml` 中定义

### 6.5 可视化训练过程
```bash
tensorboard --logdir='./logs/sb3/ppo/UR5e-Lift-Cube-IK'
```

### 6.6 测试训练好的模型

#### PPO
```bash
python3 play_sb3_ppo.py --num_envs 4 --task UR5e-Lift-Cube-IK \
    --checkpoint ./models/PPO_trained_agent/model.zip
```

#### TD3
```bash
python3 play_sb3_td3.py --num_envs 4 --task UR5e-Lift-Cube-IK \
    --checkpoint ./models/TD3_trained_agent/model.zip
```

---

## 七、训练网络结构

### 7.1 PPO 网络架构

#### 策略网络 (Policy/Actor)
```
Input (Observation): 35维
    ↓
Dense Layer 1: 256 neurons + Activation
    ↓
Dense Layer 2: 128 neurons + Activation
    ↓
Dense Layer 3: 64 neurons + Activation
    ↓
Output Layer: 动作维度（7维）+ Tanh激活
```

#### 价值网络 (Value/Critic)
```
Input (Observation): 35维
    ↓
Dense Layer 1: 256 neurons + Activation
    ↓
Dense Layer 2: 128 neurons + Activation
    ↓
Dense Layer 3: 64 neurons + Activation
    ↓
Output Layer: 1维（状态价值）
```

**激活函数搜索空间**：ELU / Tanh / ReLU

### 7.2 TD3 网络架构

#### Actor网络
```
Input (Observation): 35维
    ↓
Dense Layer 1: 256 neurons + Activation
    ↓
Dense Layer 2: 256 neurons + Activation
    ↓
Output Layer: 7维 + Tanh激活
```

#### Critic网络（双Q网络）
```
Input (Observation + Action): 35 + 7 = 42维
    ↓
Dense Layer 1: 256 neurons + Activation
    ↓
Dense Layer 2: 256 neurons + Activation
    ↓
Output Layer: 1维（Q值）

（两个独立的Critic网络）
```

**激活函数搜索空间**：ELU / ReLU / Tanh

### 7.3 DDPG 网络架构
与TD3相同，但只有单个Critic网络。

---

## 八、超参数配置

### 8.1 PPO 超参数

#### 核心参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `policy` | MlpPolicy | - | 多层感知机策略 |
| `n_timesteps` | 262,144,000 | - | 总训练步数 |
| `n_steps` | 64 | - | 每次更新的采样步数 |
| `batch_size` | 8192/16384/32768 | ✓ | 小批量大小 |
| `n_epochs` | 8 | - | 每次更新的训练轮数 |
| `gamma` | 0.95 | - | 折扣因子 |
| `gae_lambda` | 0.95 | - | GAE优势估计参数 |

#### 优化参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `learning_rate` | 1e-4 / 3e-4 | ✓ | 学习率 |
| `clip_range` | 0.2 | - | PPO裁剪范围 |
| `target_kl` | 0.02 | - | 目标KL散度 |
| `max_grad_norm` | 1.0 | - | 梯度裁剪 |
| `ent_coef` | 0.01 / 0.001 | ✓ | 熵系数（探索） |
| `vf_coef` | 0.1 | - | 价值函数损失系数 |

#### 网络参数
| 参数 | 值 | 搜索空间 |
|------|-----|----------|
| `activation_fn` | ELU/Tanh/ReLU | ✓ |
| `net_arch.pi` | [256, 128, 64] | - |
| `net_arch.vf` | [256, 128, 64] | - |

#### 归一化参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `normalize_input` | False | 是否归一化观测 |
| `normalize_value` | False | 是否归一化奖励 |
| `clip_obs` | 50.0 | 观测裁剪范围 |

#### 计算效率
```
总步数 = 1000 iterations × 64 n_steps × 4096 envs = 262,144,000
训练时长 ≈ 数小时到数天（取决于硬件）
```

### 8.2 TD3 超参数

#### 核心参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `policy` | MlpPolicy | - | 多层感知机策略 |
| `n_timesteps` | 209,715,200 | - | 总训练步数 |
| `buffer_size` | 1,000,000 | - | 经验回放缓冲区大小 |
| `batch_size` | 256 / 512 | ✓ | 小批量大小 |
| `gamma` | 0.95 | - | 折扣因子 |

#### 优化参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `learning_rate` | 1e-4 / 3e-4 | ✓ | 学习率 |
| `learning_starts` | 1000 | - | 开始学习的步数 |
| `train_freq` | 4 | - | 训练频率（步） |
| `gradient_steps` | 4 | - | 每次训练的梯度步数 |
| `tau` | 0.02 | - | 目标网络软更新系数 |

#### TD3特有参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `policy_delay` | 2 | - | 策略更新延迟 |
| `target_policy_noise` | 0.2 / 0.4 | ✓ | 目标策略平滑噪声 |
| `target_noise_clip` | 0.5 | - | 目标噪声裁剪范围 |
| `action_noise` | NormalActionNoise | - | 探索噪声（σ=0.1） |

#### 网络参数
| 参数 | 值 | 搜索空间 |
|------|-----|----------|
| `activation_fn` | ELU/ReLU/Tanh | ✓ |
| `net_arch` | [256, 256] | - |

### 8.3 DDPG 超参数

#### 核心参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `policy` | MlpPolicy | - | 多层感知机策略 |
| `n_timesteps` | 209,715,200 | - | 总训练步数 |
| `buffer_size` | 1,000,000 | - | 经验回放缓冲区大小 |
| `batch_size` | 512 / 1024 | ✓ | 小批量大小 |
| `gamma` | 0.95 | - | 折扣因子 |

#### 优化参数
| 参数 | 值 | 搜索空间 | 说明 |
|------|-----|----------|------|
| `learning_rate` | 1e-4 / 3e-4 | ✓ | 学习率 |
| `learning_starts` | 1000 | - | 开始学习的步数 |
| `train_freq` | 4 | - | 训练频率 |
| `gradient_steps` | 4 | - | 梯度步数 |
| `tau` | 0.02 | - | 软更新系数 |
| `action_noise` | NormalActionNoise | - | 探索噪声（σ=0.1） |

#### 网络参数
| 参数 | 值 | 搜索空间 |
|------|-----|----------|
| `activation_fn` | ELU/ReLU/Tanh | ✓ |
| `net_arch` | [256, 256] | - |

### 8.4 超参数搜索总结

#### 网格搜索（Grid Search）配置
```yaml
# config_sb3_ppo.yaml
method: grid
metric:
  goal: maximize
  name: rollout/ep_rew_mean

parameters:
  batch_size:
    values: [8192, 16384, 32768]
  ent_coef:
    values: [0.01, 0.001]
  learning_rate:
    values: [1e-4, 3e-4]
  activation_fn:
    values: [nn.ELU, nn.Tanh, nn.ReLU]
```

#### 搜索空间大小
- **PPO**：3 × 2 × 2 × 3 = **36 种配置**

---

## 九、算法对比

| 特性 | PPO | TD3 | DDPG |
|------|-----|-----|------|
| **类型** | On-policy | Off-policy | Off-policy |
| **策略类型** | 随机策略 | 确定性策略 | 确定性策略 |
| **Critic数量** | 1 | 2（双Q） | 1 |
| **样本效率** | 低 | 高 | 中 |
| **稳定性** | 高 | 高 | 中 |
| **探索策略** | 熵正则化 | 噪声注入 | 噪声注入 |
| **主要优势** | 稳定、易调参 | 高效、鲁棒 | 简单 |
| **典型应用** | 连续控制 | 连续控制 | 连续控制 |

---

## 十、参考资料

1. **Isaac Lab官方文档**：https://isaac-sim.github.io/IsaacLab/
2. **Stable-Baselines3文档**：https://stable-baselines3.readthedocs.io/
3. **RL Baselines3 Zoo超参数**：https://github.com/DLR-RM/rl-baselines3-zoo
4. **项目GitHub**：（填入您的仓库链接）

---

**文档版本**：1.0  
**更新日期**：2024年10月  
**联系方式**：2186808025@qq.com



