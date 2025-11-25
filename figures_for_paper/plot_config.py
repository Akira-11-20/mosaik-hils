"""
Common plotting configuration for paper figures.

This module provides standardized colors, styles, and settings
for consistent visualization across all figures.
"""

import matplotlib.pyplot as plt

# Standard color palette (from matplotlib default colors)
# Reference: compare_with_rt.py and delay sweep comparisons
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#FFD700",  # Gold (was Brown - more distinct)
    "#FF1493",  # Deep Pink (was Pink - more vivid)
    "#00CED1",  # Dark Turquoise (was Gray - more colorful)
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

# Font settings (normal size)
FONT_SETTINGS = {
    "title_size": 14,
    "title_weight": "bold",
    "label_size": 12,
    "label_weight": "bold",
    "legend_size": 10,
}

# Font settings (large text for presentations/papers)
FONT_SETTINGS_LARGE = {
    "title_size": 20,
    "title_weight": "bold",
    "label_size": 18,
    "label_weight": "bold",
    "legend_size": 16,
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


def apply_font_settings(ax, use_large_text=False):
    """Apply font settings to a matplotlib axes object.

    Args:
        ax: matplotlib axes object
        use_large_text: If True, use large text settings

    Returns:
        None (modifies ax in place)
    """
    font_settings = FONT_SETTINGS_LARGE if use_large_text else FONT_SETTINGS

    # Apply to title
    if ax.get_title():
        ax.set_title(
            ax.get_title(),
            fontsize=font_settings["title_size"],
            fontweight=font_settings["title_weight"],
        )

    # Apply to labels
    ax.set_xlabel(
        ax.get_xlabel(),
        fontsize=font_settings["label_size"],
        fontweight=font_settings["label_weight"],
    )
    ax.set_ylabel(
        ax.get_ylabel(),
        fontsize=font_settings["label_size"],
        fontweight=font_settings["label_weight"],
    )

    # Apply to z-label if 3D plot
    if hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(
            ax.get_zlabel(),
            fontsize=font_settings["label_size"],
            fontweight=font_settings["label_weight"],
        )

    # Apply to legend
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontsize=font_settings["legend_size"])

    # Apply to tick labels (軸の目盛り数字)
    ax.tick_params(axis="both", which="major", labelsize=font_settings["legend_size"])

    # Apply to offset text (1e-15などの指数表記)
    if ax.xaxis.get_offset_text():
        ax.xaxis.get_offset_text().set_fontsize(font_settings["legend_size"])
    if ax.yaxis.get_offset_text():
        ax.yaxis.get_offset_text().set_fontsize(font_settings["legend_size"])

    # Apply to scientific notation text (軸の上/右に表示される×10^n)
    try:
        # For 2D plots
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 3))
    except:
        pass

    # Apply to all text annotations (図の中の注釈)
    for text in ax.texts:
        current_size = text.get_fontsize()
        # Only increase if it's smaller than target
        if current_size < font_settings["legend_size"]:
            text.set_fontsize(font_settings["legend_size"])


def save_figure_both_sizes(fig, output_path, base_name=None):
    """Save figure in both normal and large text sizes.

    Args:
        fig: matplotlib figure object (or plt module for current figure)
        output_path: Path object or string, directory where figures will be saved
        base_name: Optional base name for the file (without extension).
                   If None, uses the stem of output_path.

    Returns:
        Tuple of (normal_path, large_path)
    """
    from pathlib import Path

    # Handle if plt module is passed instead of fig
    if hasattr(fig, 'gcf'):
        fig = fig.gcf()

    output_path = Path(output_path)

    # If output_path is a file, extract directory and base name
    if output_path.suffix:
        output_dir = output_path.parent
        if base_name is None:
            base_name = output_path.stem
    else:
        output_dir = output_path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save normal version
    normal_path = output_dir / f"{base_name}.png"
    fig.savefig(normal_path, **FIGURE_SETTINGS)

    # Apply large text settings to all axes
    for ax in fig.get_axes():
        apply_font_settings(ax, use_large_text=True)

    # Adjust layout to accommodate larger text
    fig.tight_layout()

    # Save large text version
    large_path = output_dir / f"{base_name}_large.png"
    fig.savefig(large_path, **FIGURE_SETTINGS)

    print(f"Saved: {normal_path}")
    print(f"Saved: {large_path}")

    return normal_path, large_path
