"""Common utilities shared across mosaik-hils projects

This package provides shared utilities for:
- plot_utils: Execution graph and dataflow visualization
- event_logger: Event logging for simulators
"""

__all__ = [
    "plot_execution_graph_with_data_only",
    "plot_dataflow_graph_custom",
    "EventLogger",
    "DataTag",
]

# Note: Actual imports are done in the modules that use them
# to avoid import errors when optional dependencies are missing
