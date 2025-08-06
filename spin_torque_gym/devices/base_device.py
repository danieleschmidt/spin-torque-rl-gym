"""Base class for spintronic devices.

This module defines the abstract interface that all spintronic device models
must implement for compatibility with the RL environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BaseSpintronicDevice(ABC):
    """Abstract base class for spintronic devices."""
    
    def __init__(self, device_params: Dict[str, Any]):
        """Initialize device with parameters.
        
        Args:
            device_params: Dictionary containing device parameters
        """
        self.device_params = device_params.copy()
        
        # Extract common parameters with defaults
        self.volume = device_params.get('volume', 1e-24)
        self.thickness = device_params.get('thickness', 1e-9)
        self.saturation_magnetization = device_params.get('saturation_magnetization', 800e3)
        
        # Common physical constants
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        self.kb = 1.380649e-23  # Boltzmann constant (J/K)
        self.e = 1.602176634e-19  # Elementary charge (C)
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
        
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate that required parameters are present and valid."""
        required_params = ['volume', 'saturation_magnetization']
        for param in required_params:
            if param not in self.device_params:
                raise ValueError(f"Missing required parameter: {param}")
    
    @abstractmethod
    def compute_effective_field(
        self,
        magnetization: np.ndarray,
        applied_field: np.ndarray
    ) -> np.ndarray:
        """Compute total effective magnetic field.
        
        Args:
            magnetization: Current magnetization vector (unit vector)
            applied_field: External applied field (A/m)
            
        Returns:
            Total effective field vector (A/m)
        """
        pass
    
    @abstractmethod
    def compute_resistance(self, magnetization: np.ndarray) -> float:
        """Compute device resistance based on magnetization state.
        
        Args:
            magnetization: Current magnetization vector
            
        Returns:
            Device resistance (Ω)
        """
        pass
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get device parameter value.
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value
        """
        return self.device_params.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set device parameter value.
        
        Args:
            key: Parameter key
            value: Parameter value
        """
        self.device_params[key] = value
    
    def validate_magnetization(self, magnetization: np.ndarray) -> np.ndarray:
        """Validate and normalize magnetization vector.
        
        Args:
            magnetization: Input magnetization vector
            
        Returns:
            Normalized unit magnetization vector
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(magnetization, np.ndarray):
            magnetization = np.array(magnetization)
        
        if magnetization.shape != (3,):
            raise ValueError(f"Magnetization must be 3D vector, got shape {magnetization.shape}")
        
        magnitude = np.linalg.norm(magnetization)
        if magnitude < 1e-12:
            raise ValueError("Magnetization vector cannot be zero")
        
        return magnetization / magnitude
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information.
        
        Returns:
            Dictionary with all device information
        """
        return {
            'device_type': self.__class__.__name__,
            'volume': self.volume,
            'thickness': self.thickness,
            'saturation_magnetization': self.saturation_magnetization,
            'parameters': self.device_params.copy()
        }
    
    def __repr__(self) -> str:
        """String representation of device."""
        return f"{self.__class__.__name__}(volume={self.volume:.2e}, Ms={self.saturation_magnetization:.0f})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__} with {len(self.device_params)} parameters"