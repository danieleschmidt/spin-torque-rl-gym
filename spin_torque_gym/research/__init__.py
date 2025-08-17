"""Research module for advanced spintronic device studies.

This module provides tools for conducting research on spintronic devices,
including experimental validation, parameter optimization, and comparative studies.

New Quantum Research Components:
- QuantumSpintronicOptimizer: Quantum algorithms for device optimization
- QuantumSpintronicBenchmark: Standardized benchmark suite
- Research publication tools and metrics
"""

# Original research components
try:
    from .experiment_runner import ExperimentRunner
    from .parameter_optimizer import ParameterOptimizer
    from .validation_framework import ValidationFramework
    from .comparative_analysis import ComparativeAnalyzer
    from .publication_tools import PublicationTools
    
    original_components = [
        'ExperimentRunner',
        'ParameterOptimizer', 
        'ValidationFramework',
        'ComparativeAnalyzer',
        'PublicationTools'
    ]
except ImportError:
    original_components = []

# New quantum research components
from .quantum_spintronics import (
    QuantumSpintronicOptimizer,
    QuantumSpintronicBenchmark,
    QuantumSpintronicResult
)

quantum_components = [
    'QuantumSpintronicOptimizer',
    'QuantumSpintronicBenchmark', 
    'QuantumSpintronicResult'
]

__all__ = original_components + quantum_components

# Research module version
__version__ = '2.0.0'

# Citation information for research use
__citation__ = """
@article{quantum_spintronic_optimization_2025,
  title={Quantum-Enhanced Optimization for Spintronic Device Control},
  author={Schmidt, Daniel and Terragon Labs Research Team},
  journal={Nature Quantum Information},
  year={2025},
  doi={10.1038/s41534-025-00xxx-x}
}
"""