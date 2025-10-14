# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift task for ARM-T robot."""

import gymnasium as gym

from . import agents

##
# Register Gym environments
##

gym.register(
    id="ARM-T-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:ArmTLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArmTLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="ARM-T-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:ArmTLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArmTLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="ARM-T-Lift-Cube-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:ArmTLift_IK_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArmTLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="ARM-T-Lift-Cube-IK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:ArmTLift_IK_EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArmTLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

