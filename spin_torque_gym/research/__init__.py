"""Research module for advanced spintronic device studies.

This module provides tools for conducting research on spintronic devices,
including experimental validation, parameter optimization, and comparative studies.
"""

from .experiment_runner import ExperimentRunner
from .parameter_optimizer import ParameterOptimizer
from .validation_framework import ValidationFramework
from .comparative_analysis import ComparativeAnalyzer
from .publication_tools import PublicationTools

__all__ = [
    'ExperimentRunner',
    'ParameterOptimizer', 
    'ValidationFramework',
    'ComparativeAnalyzer',
    'PublicationTools'
]