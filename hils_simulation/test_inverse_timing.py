"""Quick test to verify inverse compensator timing"""

from config.parameters import SimulationParameters
from scenarios import InverseCompScenario

params = SimulationParameters.from_env()
params.simulation_time = 0.05  # Only 50ms
params.inverse_comp.enabled = True
params.inverse_comp.gain = 15.0

print("Testing inverse compensator timing...")
print(f"Control period: {params.control_period_steps} steps")
print(f"Time resolution: {params.time_resolution}s per step")
print()

scenario = InverseCompScenario(params)
scenario.run()
