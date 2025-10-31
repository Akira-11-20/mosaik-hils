"""
Run simulations with different communication delay values
Supports both HILS and Inverse Compensation scenarios
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

from config.parameters import SimulationParameters
from scenarios import HILSScenario, InverseCompScenario


def main():
    """Run simulations with different delay values"""

    # ========================================
    # Configuration
    # ========================================

    # Define delay values to test (in milliseconds)
    # You can modify these values as needed
    delays = [140.0, 150.0, 160.0]
    delays = [int(delay / 2) for delay in delays]

    # Set to True to run with inverse compensation, False for HILS only
    use_inverse_compensation = [False]

    # ========================================
    # Run Simulations
    # ========================================

    print("=" * 60)
    print("Communication Delay Sweep")
    print("=" * 60)
    print(f"Testing {len(delays)} delay values: {delays} ms")
    print(f"Inverse compensation: {use_inverse_compensation}")
    print()

    results = []

    for use_comp in use_inverse_compensation:
        comp_label = "WITH Inverse Compensation" if use_comp else "WITHOUT Inverse Compensation"

        print(f"\n{'#'*60}")
        print(f"# {comp_label}")
        print(f"{'#'*60}")

        for delay in delays:
            print(f"\n{'='*60}")
            print(f"Running: delay={delay} ms, inverse_comp={use_comp}")
            print(f"{'='*60}")

            try:
                # Load parameters from .env
                params = SimulationParameters.from_env()

                # Override delay parameters (apply to both cmd and sense)
                params.communication.cmd_delay = delay
                params.communication.sense_delay = delay

                # Select scenario based on inverse compensation flag
                if use_comp:
                    params.inverse_comp.enabled = True
                    scenario = InverseCompScenario(params)
                    scenario_type = "InverseComp"
                else:
                    scenario = HILSScenario(params)
                    scenario_type = "HILS"

                # Run scenario
                scenario.run()

                results.append({
                    "delay": delay,
                    "inverse_comp": use_comp,
                    "scenario_type": scenario_type,
                    "status": "success",
                    "output_dir": scenario.run_dir
                })

                print(f"\n✅ Completed: delay={delay} ms, inverse_comp={use_comp}")
                print(f"   Results saved to: {scenario.run_dir}")

            except Exception as e:
                print(f"\n❌ Failed: delay={delay} ms, inverse_comp={use_comp}")
                print(f"   Error: {e}")
                results.append({
                    "delay": delay,
                    "inverse_comp": use_comp,
                    "scenario_type": scenario_type if 'scenario_type' in locals() else "unknown",
                    "status": "failed",
                    "error": str(e)
                })

    # ========================================
    # Summary
    # ========================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Group by inverse compensation setting
    for use_comp in use_inverse_compensation:
        comp_label = "WITH Inverse Compensation" if use_comp else "WITHOUT Inverse Compensation"
        print(f"\n{comp_label}:")

        comp_results = [r for r in results if r["inverse_comp"] == use_comp]
        for result in comp_results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"  {status_icon} Delay={result['delay']:6.1f} ms: {result['status']}")
            if result["status"] == "success":
                print(f"     → {result['output_dir']}")

    print("\n" + "=" * 60)
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"Completed: {successful}/{total} simulations successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
