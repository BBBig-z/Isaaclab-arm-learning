# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ArmTReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 500  # 每500次迭代保存一次模型
    experiment_name = "arm_t_reach"
    run_name = ""
    resume = False
    empirical_normalization = True  # 启用归一化以稳定训练
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.34,
        use_clipped_value_loss=True,
        clip_param=0.1,  # 极度保守的策略更新
        entropy_coef=0.005,  # 增加熵系数以鼓励探索
        num_learning_epochs=8,
        num_mini_batches=64,  # 增加mini-batch以获得更稳定的梯度
        learning_rate=4.0e-4,  # 极低学习率防止崩溃
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # 更严格的KL散度限制 
        max_grad_norm=0.5,  # 极严格的梯度裁剪 
    )


@configclass
class ArmTReachIKPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 36 # 增加采样步数提升数据利用效率
    max_iterations = 1000
    save_interval = 500  # 每500次迭代保存一次模型
    experiment_name = "arm_t_reach_ik"
    run_name = ""
    resume = False
    empirical_normalization = True  # 启用归一化以稳定训练
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # 大幅降低初始噪声
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 128], # 增强网络容量
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.34,
        use_clipped_value_loss=True,
        clip_param=0.1,  # 极度保守的策略更新
        entropy_coef=0.005,  # 增加熵系数以鼓励探索
        num_learning_epochs=8,
        num_mini_batches=512, # 增加mini-batch数量匹配更大的环境数
        learning_rate=1.0e-3,  # 极低学习率防止崩溃
        schedule="adaptive",
        gamma=0.95, # 降低折扣因子关注短期奖励
        lam=0.95,
        desired_kl=0.01,  # 更严格的KL散度限制
        max_grad_norm=0.5,  # 极严格的梯度裁剪
    )

