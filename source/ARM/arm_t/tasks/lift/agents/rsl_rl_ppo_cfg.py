# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO配置 - ARM-T抓取任务"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class ArmTLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """ARM-T抓取任务的PPO配置
    
    抓取任务比reach任务更复杂，需要：
    1. 更深的网络（256-128-64）
    2. 更多的训练迭代
    3. 适中的探索策略
    """
    
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    experiment_name = "arm_t_lift"
    run_name = ""
    resume = False
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,  # 适中的初始噪声，鼓励探索
        actor_hidden_dims=[256, 128, 64],  # 更深的网络处理复杂任务
        critic_hidden_dims=[256, 128, 64],
        activation="relu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,  # 标准PPO clip参数
        entropy_coef=0.01,  # 较高的熵系数鼓励探索
        num_learning_epochs=8,
        num_mini_batches=64,
        learning_rate=3.0e-4,  # 标准学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

