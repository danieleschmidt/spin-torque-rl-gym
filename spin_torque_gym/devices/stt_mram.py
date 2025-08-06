"""Spin-Transfer Torque MRAM device model.

This module implements the STT-MRAM device physics including:
- Magnetic tunnel junction (MTJ) structure
- Spin-transfer torque effects  
- TMR-based resistance calculation
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_device import BaseSpintronicDevice


class STTMRAMDevice(BaseSpintronicDevice):
    """Spin-Transfer Torque MRAM device model."""
    
    def __init__(self, device_params: Dict[str, Any]):
        """Initialize STT-MRAM device."""
        super().__init__(device_params)
        
        # Initialize reference layer magnetization (typically pinned)
        self.reference_magnetization = self.get_parameter(
            'reference_magnetization', 
            np.array([0, 0, 1])  # Default: +z direction
        )
        
        # Ensure unit vector
        self.reference_magnetization = self.validate_magnetization(self.reference_magnetization)
    
    def _validate_parameters(self) -> None:
        """Validate STT-MRAM specific parameters."""
        required_params = [
            'volume', 'saturation_magnetization', 'damping',
            'uniaxial_anisotropy', 'polarization'
        ]
        
        for param in required_params:
            if param not in self.device_params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Call parent validation
        super()._validate_parameters()
        
        # Validate parameter ranges
        if self.device_params['volume'] <= 0:
            raise ValueError("Volume must be positive")
        if self.device_params['saturation_magnetization'] <= 0:
            raise ValueError("Saturation magnetization must be positive")
        if not 0 <= self.device_params['damping'] <= 1:
            raise ValueError("Damping must be between 0 and 1")
        if not 0 <= self.device_params['polarization'] <= 1:
            raise ValueError("Polarization must be between 0 and 1")
    
    def compute_effective_field(
        self,
        magnetization: np.ndarray,
        applied_field: np.ndarray
    ) -> np.ndarray:
        """Compute total effective magnetic field for STT-MRAM."""
        m = self.validate_magnetization(magnetization)
        
        # Applied field contribution
        h_eff = applied_field.astype(float).copy()
        
        # Uniaxial anisotropy field
        k_u = self.get_parameter('uniaxial_anisotropy', 1e6)
        ms = self.get_parameter('saturation_magnetization', 800e3)
        easy_axis = self.get_parameter('easy_axis', np.array([0, 0, 1]))
        
        cos_theta = np.dot(m, easy_axis)
        h_anis = (2 * k_u / (self.mu0 * ms)) * cos_theta * easy_axis
        h_eff += h_anis
        
        return h_eff
    
    def compute_resistance(self, magnetization: np.ndarray) -> float:
        """Compute MTJ resistance using TMR effect."""
        m = self.validate_magnetization(magnetization)
        p = self.reference_magnetization
        
        # Resistance values
        r_p = self.get_parameter('resistance_parallel', 1e3)
        r_ap = self.get_parameter('resistance_antiparallel', 2e3)
        
        # TMR ratio
        tmr = (r_ap - r_p) / r_p
        
        # Angular dependence (cosine model)
        cos_theta = np.dot(m, p)
        resistance = r_p * (1 + tmr * (1 - cos_theta) / 2)
        
        return max(resistance, r_p * 0.5)  # Ensure positive resistance
    
    def __repr__(self) -> str:
        """String representation of device."""
        return f"STTMRAMDevice(volume={self.volume:.2e}, Ms={self.saturation_magnetization:.0f})"