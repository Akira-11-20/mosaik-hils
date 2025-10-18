"""
Configuration module for HILS simulation v2.

This module provides unified parameter loading and configuration management
for all simulation scenarios.
"""

from .parameters import SimulationParameters
from .sim_config import get_simulator_config

__all__ = ["SimulationParameters", "get_simulator_config"]
