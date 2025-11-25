#!/usr/bin/env python3
"""
Run all figure generation scripts (1-17) to create both normal and large versions.

This script automatically runs all comparison and analysis scripts in order,
generating both normal and large text versions of all figures.
"""

import subprocess
import sys
from pathlib import Path


# Define all figure generation scripts in order
FIGURE_SCRIPTS = [
    ("1_mosaik_vs_pure", "compare_mosaik_vs_pure.py"),
    ("2_delay_sweeps", "compare_delay_sweep.py"),
    ("3_tau_sweeps", "compare_tau_sweep.py"),
    ("4_tau_inverse_comp", "compare_tau_inverse_comp.py"),
    ("5_tau_inverse_comp_precise", "compare_tau_inverse_comp.py"),
    ("6_alpha_sweeps", "compare_alpha_sweep.py"),
    ("7_linear_tau", "compare_linear_tau_sweep.py"),
    ("8_linear_tau_const_inverse", "compare_linear_tau_const_inverse.py"),
    ("9_linear_tau_model_inverse", "compare_linear_tau_model_inverse.py"),
    ("10_cmd_tau_inverse", "compare_cmd_tau_inverse.py"),
    ("11_cmd_tau_inverse_sweeps", "compare_cmd_tau_inverse_sweeps.py"),
    ("12_pre_post_inverse", "compare_linear_tau_sweep.py"),
    ("13_pre_post_inverse_precise", "compare_linear_tau_sweep.py"),
    ("14_hohmann", "compare_hohmann_transfer.py"),
    ("15_formation", "compare_formation.py"),
    ("16_tau_noise", "create_heatmap.py"),
    ("17_noise_inverse_heatmap", "create_comprehensive_analysis.py"),
]


def run_script(directory: str, script: str, base_dir: Path) -> dict:
    """Run a single figure generation script.

    Args:
        directory: Directory name (e.g., "1_mosaik_vs_pure")
        script: Script filename (e.g., "compare_mosaik_vs_pure.py")
        base_dir: Base directory containing all figure directories

    Returns:
        Dictionary with status information
    """
    script_path = base_dir / directory / script

    if not script_path.exists():
        return {
            "status": "skip",
            "reason": f"Script not found: {script_path}",
        }

    print(f"\n{'='*70}")
    print(f"Running: {directory}/{script}")
    print(f"{'='*70}")

    try:
        # Run script with subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per script
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            return {
                "status": "error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "reason": "Script exceeded 5 minute timeout",
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
        }


def main():
    """Run all figure generation scripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all figure generation scripts (1-17)"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue running even if a script fails",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Only run scripts matching this pattern (e.g., '14' or 'hohmann')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all scripts without running them",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Filter scripts if --only is specified
    scripts_to_run = FIGURE_SCRIPTS
    if args.only:
        scripts_to_run = [
            (d, s) for d, s in FIGURE_SCRIPTS
            if args.only.lower() in d.lower() or args.only.lower() in s.lower()
        ]
        print(f"Filtered to {len(scripts_to_run)} script(s) matching '{args.only}'")

    # List mode
    if args.list:
        print("\nFigure generation scripts:")
        print("="*70)
        for i, (directory, script) in enumerate(scripts_to_run, 1):
            script_path = base_dir / directory / script
            exists = "✓" if script_path.exists() else "✗"
            print(f"{i:2d}. {exists} {directory}/{script}")
        print("="*70)
        print(f"Total: {len(scripts_to_run)} scripts")
        return

    # Run all scripts
    print(f"\nRunning {len(scripts_to_run)} figure generation script(s)...")
    print("="*70)

    results = {}
    for i, (directory, script) in enumerate(scripts_to_run, 1):
        print(f"\n[{i}/{len(scripts_to_run)}] Processing {directory}...")

        result = run_script(directory, script, base_dir)
        results[f"{directory}/{script}"] = result

        if result["status"] == "success":
            print(f"✓ Success: {directory}/{script}")
            # Show output if it mentions "Saved:"
            if "Saved:" in result["stdout"]:
                for line in result["stdout"].split("\n"):
                    if "Saved:" in line:
                        print(f"  {line.strip()}")
        elif result["status"] == "skip":
            print(f"⊘ Skipped: {directory}/{script}")
            print(f"  Reason: {result['reason']}")
        elif result["status"] == "timeout":
            print(f"⏱ Timeout: {directory}/{script}")
            print(f"  Reason: {result['reason']}")
            if not args.skip_errors:
                print("\nStopping due to timeout. Use --skip-errors to continue.")
                break
        else:  # error
            print(f"✗ Error: {directory}/{script}")
            if "stderr" in result and result["stderr"]:
                print("  Error output:")
                for line in result["stderr"].split("\n")[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"    {line}")
            if not args.skip_errors:
                print("\nStopping due to error. Use --skip-errors to continue.")
                break

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    skip_count = sum(1 for r in results.values() if r["status"] == "skip")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    timeout_count = sum(1 for r in results.values() if r["status"] == "timeout")

    print(f"Total scripts: {len(scripts_to_run)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⊘ Skipped: {skip_count}")
    print(f"  ✗ Errors:  {error_count}")
    print(f"  ⏱ Timeout: {timeout_count}")

    if error_count > 0:
        print("\nFailed scripts:")
        for name, result in results.items():
            if result["status"] == "error":
                print(f"  - {name}")
                if "reason" in result:
                    print(f"    {result['reason']}")

    print("="*70)

    # Count generated files
    normal_count = 0
    large_count = 0
    for directory, _ in scripts_to_run:
        dir_path = base_dir / directory
        if dir_path.exists():
            normal_count += len(list(dir_path.glob("*.png"))) - len(list(dir_path.glob("*_large.png")))
            large_count += len(list(dir_path.glob("*_large.png")))

    print(f"\nGenerated figures:")
    print(f"  Normal version: ~{normal_count} files")
    print(f"  Large version:  ~{large_count} files")
    print("="*70)

    if success_count == len(scripts_to_run):
        print("\n✓ All scripts completed successfully!")
    else:
        print(f"\n⚠ {len(scripts_to_run) - success_count} script(s) did not complete successfully")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
