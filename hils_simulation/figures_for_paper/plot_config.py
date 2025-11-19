"""
Common plotting configuration for paper figures.

This module provides standardized colors, styles, and settings
for consistent visualization across all figures.
"""

# Standard color palette (from matplotlib default colors)
# Reference: compare_with_rt.py and delay sweep comparisons
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
]

# Baseline (RT) styling
BASELINE_STYLE = {
    "color": "black",
    "linewidth": 2.5,
    "linestyle": "--",  # Dashed line
    "alpha": 0.8,
    "label": "RT Baseline (0ms delay)",
}

# Baseline for deviation plots (thinner)
BASELINE_DEVIATION_STYLE = {
    "color": "black",
    "linewidth": 2.5,
    "linestyle": "--",  # Dashed line
    "alpha": 0.8,
    "label": "RT Baseline (0ms delay)",
}

# Scenario line styling
SCENARIO_STYLE = {
    "linewidth": 1.8,
    "alpha": 0.7,
}

# Font settings
FONT_SETTINGS = {
    "title_size": 14,
    "title_weight": "bold",
    "label_size": 12,
    "label_weight": "bold",
    "legend_size": 10,
}

# Figure settings
FIGURE_SETTINGS = {
    "dpi": 300,
    "bbox_inches": "tight",
}

# Grid settings
GRID_SETTINGS = {
    "alpha": 0.3,
}


def get_scenario_color(index):
    """Get color for scenario by index.

    Args:
        index: Scenario index (0-based)

    Returns:
        Color hex string
    """
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def get_scenario_style(index, label=None):
    """Get complete style dict for a scenario.

    Args:
        index: Scenario index (0-based)
        label: Optional label override

    Returns:
        Dictionary of matplotlib plot parameters
    """
    style = SCENARIO_STYLE.copy()
    style["color"] = get_scenario_color(index)
    if label is not None:
        style["label"] = label
    return style
