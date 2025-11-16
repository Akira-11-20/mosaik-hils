"""Test config override approach"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.orbital_parameters import load_config_from_env

# Test: Load config and override max_thrust
config = load_config_from_env()

print("Before override:")
print(f"  max_thrust = {config.spacecraft.max_thrust} N")
print(f"  mass = {config.spacecraft.mass} kg")

# Override
config.spacecraft.max_thrust = 100.0
config.spacecraft.mass = 1000.0

print("\nAfter override:")
print(f"  max_thrust = {config.spacecraft.max_thrust} N")
print(f"  mass = {config.spacecraft.mass} kg")

print("\n✅ Direct attribute override works!")
