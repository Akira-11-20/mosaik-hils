#!/usr/bin/env python3
"""
Update all figure scripts to generate both normal and large text versions.

This script automatically modifies all Python scripts in figures_for_paper/
to use the save_figure_both_sizes() function instead of plt.savefig().
"""

import re
from pathlib import Path


def has_plot_config_import(content: str) -> bool:
    """Check if script imports from plot_config."""
    return "from plot_config import" in content or "import plot_config" in content


def add_save_figure_import(content: str) -> str:
    """Add save_figure_both_sizes import to plot_config imports."""
    if "save_figure_both_sizes" in content:
        return content  # Already imported

    # Pattern 1: Multi-line import with parentheses
    pattern = r"(from plot_config import \([^)]+)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Add to existing multi-line import
        old_import = match.group(0)
        # Remove trailing comma and whitespace before closing paren
        base = old_import.rstrip(")").rstrip().rstrip(",")
        new_import = base + ",\n    save_figure_both_sizes,\n)"
        return content.replace(old_import, new_import, 1)

    # Pattern 2: Single line import
    pattern = r"from plot_config import ([^\n]+)"
    match = re.search(pattern, content)
    if match:
        old_import = match.group(0)
        # Remove trailing comma if present
        if old_import.rstrip().endswith(","):
            new_import = old_import.rstrip().rstrip(",") + ", save_figure_both_sizes"
        else:
            new_import = old_import + ", save_figure_both_sizes"
        return content.replace(old_import, new_import, 1)

    # If no plot_config import found, add it
    # Find the last import statement
    lines = content.split("\n")
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_end_idx = i

    # Insert after last import
    lines.insert(
        import_end_idx + 1,
        "from plot_config import save_figure_both_sizes",
    )
    return "\n".join(lines)


def ensure_path_import(content: str) -> str:
    """Ensure Path is imported from pathlib."""
    if "from pathlib import Path" in content or "import pathlib" in content:
        return content

    # Find the last import statement
    lines = content.split("\n")
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_end_idx = i

    # Check if we need to add it
    if "Path" in content:
        lines.insert(import_end_idx + 1, "from pathlib import Path")

    return "\n".join(lines)


def replace_savefig_calls(content: str) -> tuple[str, int]:
    """
    Replace plt.savefig() and fig.savefig() calls with save_figure_both_sizes().

    Returns:
        Tuple of (modified_content, number_of_replacements)
    """
    replacements = 0

    # Pattern to match savefig calls
    # Matches: plt.savefig(path, ...) or fig.savefig(path, ...)
    pattern = r'(fig|plt)\.savefig\(\s*([^,\)]+)(?:\s*,\s*[^)]+)?\s*\)'

    def replace_call(match):
        nonlocal replacements
        fig_var = match.group(1)  # 'fig' or 'plt'
        path_arg = match.group(2).strip()

        # Check if it's a variable (no quotes)
        if not (path_arg.startswith('"') or path_arg.startswith("'")):
            # It's a variable, try to keep using it but wrapped
            # Extract variable name
            var_name = path_arg.strip()
            # Use the variable's parent and stem
            replacements += 1
            return f'save_figure_both_sizes({fig_var}, {var_name}.parent, base_name={var_name}.stem)'

        # Remove quotes from path
        path_clean = path_arg.strip('"').strip("'")

        # Extract directory and filename
        # Check if it's already a Path object or a string
        if "Path(" in path_arg:
            # Extract the path string from Path(...)
            path_match = re.search(r'Path\(["\']([^"\']+)["\']\)', path_arg)
            if path_match:
                full_path = path_match.group(1)
            else:
                # Can't parse, skip this one
                return match.group(0)
        else:
            full_path = path_clean

        # Extract base name (without extension)
        path_obj = Path(full_path)
        base_name = path_obj.stem
        parent_dir = path_obj.parent

        replacements += 1

        # If parent_dir is current directory or empty, use current script directory
        if str(parent_dir) == "." or str(parent_dir) == "" or parent_dir == Path("."):
            return f'save_figure_both_sizes({fig_var}, Path(__file__).parent, base_name="{base_name}")'
        else:
            # Use the parent directory
            return f'save_figure_both_sizes({fig_var}, Path("{parent_dir}"), base_name="{base_name}")'

    content = re.sub(pattern, replace_call, content)
    return content, replacements


def update_script(script_path: Path, dry_run: bool = False) -> dict:
    """
    Update a single script to use save_figure_both_sizes().

    Args:
        script_path: Path to the script
        dry_run: If True, don't write changes to file

    Returns:
        Dictionary with update information
    """
    try:
        content = script_path.read_text()
        original_content = content

        # Check if this script uses matplotlib savefig
        if "savefig" not in content:
            return {"status": "skipped", "reason": "no savefig calls"}

        # Add save_figure_both_sizes import
        if has_plot_config_import(content) or "matplotlib" in content:
            content = add_save_figure_import(content)
            content = ensure_path_import(content)

            # Replace savefig calls
            content, num_replacements = replace_savefig_calls(content)

            if content != original_content:
                if not dry_run:
                    script_path.write_text(content)
                return {
                    "status": "updated",
                    "replacements": num_replacements,
                    "dry_run": dry_run,
                }
            else:
                return {"status": "skipped", "reason": "no changes needed"}
        else:
            return {"status": "skipped", "reason": "no matplotlib/plot_config usage"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    """Update all figure scripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update all figure scripts to use save_figure_both_sizes()"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    figures_dir = Path(__file__).parent

    # Find all Python scripts (excluding this update script and test scripts)
    scripts = []
    for script in figures_dir.rglob("*.py"):
        # Skip certain files
        if script.name in [
            "update_all_scripts.py",
            "__init__.py",
            "plot_config.py",
            "test_save_both_sizes.py",
        ]:
            continue
        scripts.append(script)

    print(f"Found {len(scripts)} scripts to check")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print("=" * 70)

    results = {
        "updated": [],
        "skipped": [],
        "error": [],
    }

    for script in sorted(scripts):
        relative_path = script.relative_to(figures_dir)
        result = update_script(script, dry_run=args.dry_run)

        if result["status"] == "updated":
            results["updated"].append((relative_path, result))
            symbol = "→" if args.dry_run else "✓"
            print(
                f"{symbol} {relative_path} - {result['replacements']} replacement(s)"
            )
        elif result["status"] == "skipped":
            results["skipped"].append((relative_path, result))
            print(f"  {relative_path} - {result['reason']}")
        elif result["status"] == "error":
            results["error"].append((relative_path, result))
            print(f"✗ {relative_path} - Error: {result['error']}")

    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Updated: {len(results['updated'])} scripts")
    print(f"  Skipped: {len(results['skipped'])} scripts")
    print(f"  Errors:  {len(results['error'])} scripts")

    if results["updated"]:
        print("\nAll updated scripts will now generate both versions:")
        print("  - <filename>.png (normal text size)")
        print("  - <filename>_large.png (large text for presentations/papers)")

    if not args.dry_run and results["updated"]:
        print("\n✓ All updates completed successfully!")
    elif args.dry_run and results["updated"]:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
