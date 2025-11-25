#!/usr/bin/env python3
"""
Fix import paths for plot_config in all figure scripts.

Adds sys.path.insert() before plot_config imports to ensure the module can be found.
"""

import re
from pathlib import Path


def fix_script_imports(script_path: Path) -> bool:
    """Fix imports in a single script.

    Returns:
        True if modifications were made
    """
    content = script_path.read_text()
    original_content = content

    # Check if plot_config is imported
    if "from plot_config import" not in content and "import plot_config" not in content:
        return False

    # Check if sys.path.insert is already there
    if "sys.path.insert(0, str(Path(__file__).parent.parent))" in content:
        return False

    # Check if in subdirectory (needs path fix)
    if script_path.parent.name == "figures_for_paper":
        # Script is directly in figures_for_paper, no fix needed
        return False

    # Need to add sys.path.insert
    lines = content.split("\n")

    # Find import section
    import_start = -1
    import_end = -1
    plot_config_line = -1

    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            if import_start == -1:
                import_start = i
            import_end = i

            if "plot_config" in line:
                plot_config_line = i

    if plot_config_line == -1:
        return False

    # Check if sys and Path are imported
    has_sys = any("import sys" in line for line in lines[:plot_config_line])
    has_path = any("from pathlib import Path" in line or "import pathlib" in line
                   for line in lines[:plot_config_line])

    # Build the fix
    insert_lines = []
    insert_pos = plot_config_line

    if not has_sys:
        insert_lines.append("import sys")

    if not has_path:
        insert_lines.append("from pathlib import Path")

    # Add blank line before comment if we're adding imports
    if insert_lines:
        insert_lines.append("")

    insert_lines.append("# Add parent directory to path for plot_config")
    insert_lines.append("sys.path.insert(0, str(Path(__file__).parent.parent))")

    # Insert the fix
    for offset, line in enumerate(insert_lines):
        lines.insert(insert_pos + offset, line)

    content = "\n".join(lines)

    if content != original_content:
        script_path.write_text(content)
        return True

    return False


def main():
    """Fix all scripts."""
    base_dir = Path(__file__).parent

    # Find all Python scripts in subdirectories
    scripts = []
    for script in base_dir.rglob("*.py"):
        # Skip this script and other utility scripts
        if script.name in ["fix_imports.py", "update_all_scripts.py",
                           "test_save_both_sizes.py", "run_all_figures.py",
                           "plot_config.py"]:
            continue

        # Only process scripts in subdirectories
        if script.parent != base_dir:
            scripts.append(script)

    print(f"Found {len(scripts)} scripts to check")
    print("="*70)

    fixed_count = 0
    for script in sorted(scripts):
        relative_path = script.relative_to(base_dir)
        was_fixed = fix_script_imports(script)

        if was_fixed:
            print(f"✓ Fixed: {relative_path}")
            fixed_count += 1
        else:
            print(f"  Skipped: {relative_path}")

    print("="*70)
    print(f"\nFixed {fixed_count}/{len(scripts)} scripts")

    if fixed_count > 0:
        print("\n✓ All import paths have been fixed!")
        print("Scripts can now import plot_config from subdirectories.")
    else:
        print("\nNo scripts needed fixing.")


if __name__ == "__main__":
    main()
