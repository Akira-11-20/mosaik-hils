"""Common utilities shared across mosaik-hils projects

This package provides shared utilities for:
- plot_utils: Execution graph and dataflow visualization
- event_logger: Event logging for simulators
"""

# Import functions for easy access
try:
    from .plot_utils import plot_execution_graph_with_data_only, plot_dataflow_graph_custom
except ImportError:
    plot_execution_graph_with_data_only = None
    plot_dataflow_graph_custom = None

try:
    from .event_logger import EventLogger, DataTag
except ImportError:
    EventLogger = None
    DataTag = None

__all__ = [
    "plot_execution_graph_with_data_only",
    "plot_dataflow_graph_custom",
    "EventLogger",
    "DataTag",
]
