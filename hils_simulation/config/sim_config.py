"""
Simulator configuration for Mosaik-based HILS simulation.

This module provides the standard simulator configuration dictionary
used by all Mosaik-based scenarios.
"""


def get_simulator_config(
    include_bridge: bool = True,
    include_inverse_comp: bool = False,
    include_dual_feedback_inverse_comp: bool = False,
) -> dict:
    """
    Get simulator configuration for Mosaik world.

    Args:
        include_bridge: Whether to include communication bridge simulators
        include_inverse_comp: Whether to include inverse compensator simulator
        include_dual_feedback_inverse_comp: Whether to include dual feedback inverse compensator

    Returns:
        Dictionary of simulator configurations for Mosaik
    """
    config = {
        "ControllerSim": {
            "python": "simulators.controller_simulator:ControllerSimulator",
        },
        "PlantSim": {
            "python": "simulators.plant_simulator_with_model:PlantSimulator",
        },
        "EnvSim": {
            "python": "simulators.env_simulator:EnvSimulator",
        },
        "DataCollector": {
            "python": "simulators.data_collector:DataCollectorSimulator",
        },
    }

    if include_bridge:
        config["BridgeSim"] = {
            "python": "simulators.bridge_simulator:BridgeSimulator",
        }

    if include_inverse_comp:
        config["InverseCompSim"] = {
            "python": "simulators.inverse_compensator_simulator:InverseCompensatorSimulator",
        }

    if include_dual_feedback_inverse_comp:
        config["DualFeedbackInverseCompSim"] = {
            "python": "simulators.inverse_compensator_dual_feedback_simulator:DualFeedbackInverseCompensatorSimulator",
        }

    return config
