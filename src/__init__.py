"""
ANOS: Predictive Resource Management for Operating Systems.

A platform-independent neural layer for predictive resource management.
"""

from .monitor import NetworkMonitor
from .predictor import NetLSTM, NetMLP, NetTransformer
from .optimizer import NetworkOptimizer
from .utils import set_seed, compute_metrics, baseline_ma, create_sequences

__all__ = [
    "NetworkMonitor",
    "NetLSTM",
    "NetMLP",
    "NetTransformer",
    "NetworkOptimizer",
    "set_seed",
    "compute_metrics",
    "baseline_ma",
    "create_sequences",
]

__version__ = "1.0.0"
__author__ = "Your Name"
