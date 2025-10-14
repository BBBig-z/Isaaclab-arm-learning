# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ARM-T 6-DOF robot arm.

The following configurations are available:

* :obj:`ARM_T_CFG`: ARM-T robot arm configuration.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path: arm_t/robots/arm_t.py -> arm_t/robots -> arm_t -> ARM -> ARM/data
TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

##
# Configuration
##


ARM_T_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/arm_t/arm_t.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # 固定底座，防止机械臂飘浮
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.7071068, 0.0, 0.0, 0.7071068),  # Quaternion for 90 degrees rotation around X-axis
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.5,  # Slight bend to avoid singularity
            "joint3": 0.5,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_1_joint": 0.0,  # Closed position to avoid instability
            "gripper_2_joint": 0.0,  # Closed position
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 6-DOF arm actuators
        # Based on masses from URDF:
        # link1: 0.64kg, link2: 0.83kg, link3: 0.772kg, link4: 0.427kg, link5: 0.38kg, link6: 0.43kg
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=3.0,  # Based on URDF effort limit
            velocity_limit_sim=1.0,  # Based on URDF velocity limit
            stiffness={
                "joint1": 200.0,  # Base rotation - moves all mass (~3.5kg)
                "joint2": 180.0,  # Shoulder - moves everything except base
                "joint3": 150.0,  # Elbow - moves lower arm and wrist
                "joint4": 100.0,  # Wrist joint 1
                "joint5": 80.0,   # Wrist joint 2
                "joint6": 60.0,   # Wrist joint 3 - smallest load
            },
            damping={
                "joint1": 80.0,
                "joint2": 70.0,
                "joint3": 60.0,
                "joint4": 40.0,
                "joint5": 30.0,
                "joint6": 25.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_.*_joint"],
            effort_limit_sim=0.18,  # Based on URDF
            velocity_limit_sim=1.0,
            stiffness=50.0,
            damping=15.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of ARM-T 6-DOF robot arm."""


ARM_T_HIGH_PD_CFG = ARM_T_CFG.copy()
ARM_T_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
ARM_T_HIGH_PD_CFG.actuators["arm"].stiffness = {
    "joint1": 500.0,
    "joint2": 450.0,
    "joint3": 400.0,
    "joint4": 350.0,
    "joint5": 300.0,
    "joint6": 250.0,
}
ARM_T_HIGH_PD_CFG.actuators["arm"].damping = {
    "joint1": 150.0,
    "joint2": 140.0,
    "joint3": 130.0,
    "joint4": 110.0,
    "joint5": 90.0,
    "joint6": 80.0,
}

"""Configuration of ARM-T robot with stiffer PD control for IK."""
