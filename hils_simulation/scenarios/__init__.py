"""
Scenario module for HILS simulation v2.

This module provides different simulation scenarios:
- HILS: Hardware-in-the-Loop Simulation with communication delays
- RT: Real-Time simulation without delays (Mosaik-based)
- InverseComp: HILS with inverse compensation
- InverseCompDualFeedback: HILS with inverse compensation and dual feedback paths
- PurePython: Pure Python simulation without Mosaik framework
"""

from .base_scenario import BaseScenario
from .hils_scenario import HILSScenario
from .inverse_comp_dual_feedback_scenario import InverseCompDualFeedbackScenario
from .inverse_comp_scenario import InverseCompScenario
from .pure_python_scenario import PurePythonScenario
from .rt_scenario import RTScenario

__all__ = [
    "BaseScenario",
    "HILSScenario",
    "InverseCompScenario",
    "InverseCompDualFeedbackScenario",
    "PurePythonScenario",
    "RTScenario",
]
