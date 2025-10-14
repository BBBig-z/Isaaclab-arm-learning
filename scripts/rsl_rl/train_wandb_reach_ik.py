#!/usr/bin/env python3
# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# SPDX-License-Identifier: BSD-3-Clause

"""ARM-T Reach-IK训练脚本 - WandB集成版"""

import argparse
import os
import sys
from datetime import datetime
import yaml

from isaaclab.app import AppLauncher

# 添加参数
parser = argparse.ArgumentParser(description="Train ARM-T Reach-IK with RSL-RL and WandB")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ARM-T-Reach-IK-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb_project", type=str, default="arm_t_reach_ik_rsl_rl", help="WandB project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (username/team)")
parser.add_argument("--config", type=str, default="/home/y/works/rl/isaac_so_arm101/source/ARM/arm_t/tasks/reach/agents/config_rsl_rl_ppo_reach_ik.yaml", help="Config YAML file")
# 添加AppLauncher参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 启用视频录制
if args_cli.video:
    args_cli.enable_cameras = True

# 启动OmniverseApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""导入其他模块"""

# 添加arm_t模块路径到sys.path
import sys
from pathlib import Path
workspace_path = Path(__file__).resolve().parents[2]  # 项目根目录
arm_t_path = workspace_path / "source" / "ARM"
if str(arm_t_path) not in sys.path:
    sys.path.insert(0, str(arm_t_path))

import gymnasium as gym
import torch
import wandb

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import arm_t.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """主训练函数"""
    
    # 加载配置
    print(f"[INFO] Loading configuration from: {args_cli.config}")
    with open(args_cli.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # 初始化WandB（设置环境变量避免进程重启）
    os.environ["WANDB_START_METHOD"] = "thread"
    print("[INFO] Initializing WandB...")
    wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        config=yaml_config,
        sync_tensorboard=True,
    )
    config = wandb.config
    
    # 解析环境配置
    print(f"[INFO] Creating environment: {args_cli.task}")
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else config.num_envs,
    )
    
    # 设置种子
    seed = args_cli.seed if args_cli.seed is not None else config.seed
    env_cfg.seed = seed
    
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 创建agent配置
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    
    agent_cfg = RslRlOnPolicyRunnerCfg()
    agent_cfg.seed = seed
    agent_cfg.device = args_cli.device
    agent_cfg.num_steps_per_env = config.num_steps_per_env
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else config.max_iterations
    agent_cfg.save_interval = config.save_interval
    agent_cfg.experiment_name = "arm_t_reach_ik"
    agent_cfg.empirical_normalization = config.empirical_normalization
    agent_cfg.logger = "tensorboard"
    
    # Policy配置
    agent_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=config.init_noise_std,
        actor_hidden_dims=list(config.actor_hidden_dims),
        critic_hidden_dims=list(config.critic_hidden_dims),
        activation=config.activation,
    )
    
    # Algorithm配置
    agent_cfg.algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=config.value_loss_coef,
        use_clipped_value_loss=config.use_clipped_value_loss,
        clip_param=config.clip_param,
        entropy_coef=config.entropy_coef,
        num_learning_epochs=config.num_learning_epochs,
        num_mini_batches=config.num_mini_batches,
        learning_rate=config.learning_rate,
        schedule=config.schedule,
        gamma=config.gamma,
        lam=config.lam,
        desired_kl=config.desired_kl,
        max_grad_norm=config.max_grad_norm,
    )
    
    print("[INFO] Agent configuration:")
    print_dict(agent_cfg.to_dict(), nesting=4)
    
    # 创建日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # 视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # 包装环境
    print("[INFO] Wrapping environment for RSL-RL...")
    env = RslRlVecEnvWrapper(env)
    
    # 创建runner
    print("[INFO] Creating OnPolicyRunner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 保存配置
    print("[INFO] Saving configurations...")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # 训练
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    print("[INFO] Training completed!")
    
    # 关闭
    env.close()
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
