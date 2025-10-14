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
ARM-T 6-DOF Robot Extension Package.

This package contains robot configurations and task implementations for the ARM-T 6-DOF robotic arm.
"""

# Register Gym environments.
from .tasks import *

__all__ = ["tasks", "robots"]
