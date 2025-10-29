"""
Run inverse compensation simulation with different gain values
"""

import sys
from pathlib import Path

from config.parameters import SimulationParameters
from scenarios import InverseCompScenario


def main():
    """Run simulations with different compensation gains"""

    # Define gain values to test
    gains = [12.0, 14.0, 16.0, 18.0, 20.0]

    print("=" * 60)
    print("Inverse Compensation Gain Sweep")
    print("=" * 60)
    print(f"Testing {len(gains)} gain values: {gains}")
    print()

    results = []

    for gain in gains:
        print(f"\n{'='*60}")
        print(f"Running with gain = {gain}")
        print(f"{'='*60}")

        try:
            # Load parameters from .env
            params = SimulationParameters.from_env()

            # Override compensation gain
            params.inverse_comp.gain = gain
            params.inverse_comp.enabled = True

            # Create and run scenario
            scenario = InverseCompScenario(params)
            scenario.run()

            results.append({
                "gain": gain,
                "status": "success",
                "output_dir": scenario.run_dir
            })

            print(f"\n✅ Completed: gain={gain}")
            print(f"   Results saved to: {scenario.run_dir}")

        except Exception as e:
            print(f"\n❌ Failed: gain={gain}")
            print(f"   Error: {e}")
            results.append({
                "gain": gain,
                "status": "failed",
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} Gain={result['gain']:5.1f}: {result['status']}")
        if result["status"] == "success":
            print(f"   → {result['output_dir']}")

    print("\n" + "=" * 60)
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"Completed: {successful}/{len(results)} simulations successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
