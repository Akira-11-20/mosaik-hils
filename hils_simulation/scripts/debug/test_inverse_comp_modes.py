"""
Test script for InverseCompensator dual-mode operation

This script tests both modes:
1. Constant mode (direct gain): tau_model_type="constant" → uses gain parameter directly
2. Adaptive mode (tau-based): tau_model_type="linear" → calculates gain from thrust
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulators.inverse_compensator_simulator import InverseCompensator


def test_constant_mode():
    """Test constant mode with direct gain"""
    print("\n" + "=" * 60)
    print("TEST 1: Constant Mode (Direct Gain)")
    print("=" * 60)

    comp = InverseCompensator(
        comp_id="test_constant",
        gain=10.0,  # Direct gain value
        comp_type="command",
        tau_model_type="constant",  # Use constant mode
    )

    print(f"Mode: {'Adaptive' if comp.use_adaptive_gain else 'Constant (Direct Gain)'}")
    print(f"Initial gain: {comp.gain:.2f}")
    print(f"Base gain: {comp.base_gain:.2f}")

    # Test with different thrust values
    test_thrusts = [5.0, 10.0, 20.0, 5.0]
    print("\nProcessing thrust commands:")
    for thrust in test_thrusts:
        command = {"thrust": thrust, "duration": 10}
        comp.process_input(command)
        output = comp.get_output()
        print(f"  Input thrust: {thrust:5.1f}N → Output thrust: {output['thrust']:6.2f}N, Gain: {comp.gain:.2f}")

    print("\n✓ In constant mode, gain should remain fixed at 10.0")


def test_adaptive_mode():
    """Test adaptive mode with tau-based gain calculation"""
    print("\n" + "=" * 60)
    print("TEST 2: Adaptive Mode (Tau-based Gain)")
    print("=" * 60)

    comp = InverseCompensator(
        comp_id="test_adaptive",
        gain=10.0,  # This will be overridden by tau-based calculation
        comp_type="command",
        tau_to_gain_ratio=0.1,  # tau [ms] * 0.1 = gain
        base_tau=100.0,  # Base time constant [ms]
        tau_model_type="linear",  # Use adaptive mode
        tau_model_params={"sensitivity": 0.5},  # sensitivity to thrust rate changes
        time_resolution=0.001,
    )

    print(f"Mode: {'Adaptive' if comp.use_adaptive_gain else 'Constant (Direct Gain)'}")
    print(f"Initial gain: {comp.gain:.2f}")
    print(f"Base tau: {comp.base_tau:.2f} ms")
    print(f"Tau-to-gain ratio: {comp.tau_to_gain_ratio:.2f}")

    # Test with different thrust values
    test_thrusts = [5.0, 10.0, 20.0, 5.0]
    print("\nProcessing thrust commands (tau should vary with thrust):")
    for thrust in test_thrusts:
        command = {"thrust": thrust, "duration": 10}
        comp.process_input(command)
        output = comp.get_output()
        print(
            f"  Input thrust: {thrust:5.1f}N → Output thrust: {output['thrust']:6.2f}N, "
            f"Tau: {comp.current_tau:6.2f}ms, Gain: {comp.gain:.2f}"
        )

    print("\n✓ In adaptive mode, tau and gain should vary with thrust level")


def test_stats():
    """Test statistics output for both modes"""
    print("\n" + "=" * 60)
    print("TEST 3: Statistics Output")
    print("=" * 60)

    # Constant mode
    comp_const = InverseCompensator(
        comp_id="const", gain=15.0, tau_model_type="constant"
    )
    print("\nConstant mode stats:")
    print(comp_const.get_stats())

    # Adaptive mode
    comp_adapt = InverseCompensator(
        comp_id="adapt",
        gain=15.0,
        tau_to_gain_ratio=0.1,
        base_tau=100.0,
        tau_model_type="linear",
        tau_model_params={"sensitivity": 0.5},
    )
    print("\nAdaptive mode stats:")
    print(comp_adapt.get_stats())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("InverseCompensator Dual-Mode Test")
    print("=" * 60)

    test_constant_mode()
    test_adaptive_mode()
    test_stats()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
