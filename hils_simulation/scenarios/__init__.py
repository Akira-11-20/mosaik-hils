"""
Scenario module for HILS simulation v2.

This module provides different simulation scenarios:
- HILS: Hardware-in-the-Loop Simulation with communication delays
- RT: Real-Time simulation without delays (Mosaik-based)
- InverseComp: HILS with inverse compensation
- PurePython: Pure Python simulation without Mosaik framework
"""

from .base_scenario import BaseScenario
from .hils_scenario import HILSScenario
from .rt_scenario import RTScenario
from .inverse_comp_scenario import InverseCompScenario
from .pure_python_scenario import PurePythonScenario

__all__ = [
    "BaseScenario",
    "HILSScenario",
    "RTScenario",
    "InverseCompScenario",
    "PurePythonScenario",
]
