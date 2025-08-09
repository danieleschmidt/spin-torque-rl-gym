"""Device factory for creating spintronic device instances.

This module provides a factory pattern for creating different types
of spintronic devices with standardized interfaces.
"""

from typing import Any, Dict, Type

from .base_device import BaseSpintronicDevice
from .skyrmion_device import SkyrmionDevice
from .sot_mram import SOTMRAMDevice
from .stt_mram import STTMRAMDevice
from .vcma_mram import VCMAMRAMDevice


class DeviceFactory:
    """Factory for creating spintronic device instances."""

    def __init__(self):
        """Initialize device factory with registered device types."""
        self._device_types: Dict[str, Type[BaseSpintronicDevice]] = {}
        self._register_default_devices()

    def _register_default_devices(self) -> None:
        """Register default device types."""
        self.register_device('stt_mram', STTMRAMDevice)
        self.register_device('sot_mram', SOTMRAMDevice)
        self.register_device('vcma_mram', VCMAMRAMDevice)
        self.register_device('skyrmion', SkyrmionDevice)
        self.register_device('skyrmion_track', SkyrmionDevice)  # Alias for skyrmion environments

    def register_device(self, device_type: str, device_class: Type[BaseSpintronicDevice]) -> None:
        """Register a new device type.
        
        Args:
            device_type: String identifier for device type
            device_class: Device class that inherits from BaseSpintronicDevice
            
        Raises:
            ValueError: If device_class doesn't inherit from BaseSpintronicDevice
        """
        if not issubclass(device_class, BaseSpintronicDevice):
            raise ValueError("Device class must inherit from BaseSpintronicDevice")

        self._device_types[device_type.lower()] = device_class

    def create_device(
        self,
        device_type: str,
        device_params: Dict[str, Any]
    ) -> BaseSpintronicDevice:
        """Create a device instance.
        
        Args:
            device_type: Type of device to create
            device_params: Parameters for device initialization
            
        Returns:
            Device instance
            
        Raises:
            ValueError: If device type is not registered
        """
        device_type = device_type.lower()

        if device_type not in self._device_types:
            available_types = list(self._device_types.keys())
            raise ValueError(f"Unknown device type '{device_type}'. Available types: {available_types}")

        device_class = self._device_types[device_type]

        try:
            return device_class(device_params)
        except Exception as e:
            raise RuntimeError(f"Failed to create {device_type} device: {e}")

    def get_available_devices(self) -> list:
        """Get list of available device types."""
        return list(self._device_types.keys())

    def get_device_info(self, device_type: str) -> Dict[str, Any]:
        """Get information about a device type.
        
        Args:
            device_type: Device type to query
            
        Returns:
            Dictionary with device type information
        """
        device_type = device_type.lower()

        if device_type not in self._device_types:
            raise ValueError(f"Unknown device type '{device_type}'")

        device_class = self._device_types[device_type]

        return {
            'name': device_type,
            'class': device_class.__name__,
            'module': device_class.__module__,
            'docstring': device_class.__doc__
        }

    def create_default_device(self, device_type: str) -> BaseSpintronicDevice:
        """Create device with default parameters.
        
        Args:
            device_type: Type of device to create
            
        Returns:
            Device instance with default parameters
        """
        default_params = self.get_default_parameters(device_type)
        return self.create_device(device_type, default_params)

    def get_default_parameters(self, device_type: str) -> Dict[str, Any]:
        """Get default parameters for a device type.
        
        Args:
            device_type: Device type
            
        Returns:
            Dictionary with default parameters
        """
        device_type = device_type.lower()

        if device_type == 'stt_mram':
            return {
                'volume': 50e-9 * 100e-9 * 2e-9,  # 50×100×2 nm³
                'area': 50e-9 * 100e-9,
                'thickness': 2e-9,
                'aspect_ratio': 2.0,
                'saturation_magnetization': 800e3,  # A/m
                'damping': 0.01,
                'uniaxial_anisotropy': 1.2e6,  # J/m³
                'exchange_constant': 20e-12,  # J/m
                'polarization': 0.7,
                'resistance_parallel': 1e3,  # Ω
                'resistance_antiparallel': 2e3,  # Ω
                'easy_axis': [0, 0, 1],
                'reference_magnetization': [0, 0, 1]
            }
        elif device_type == 'sot_mram':
            return {
                'volume': 100e-9 * 100e-9 * 1e-9,  # 100×100×1 nm³
                'area': 100e-9 * 100e-9,
                'thickness': 1e-9,
                'saturation_magnetization': 800e3,
                'damping': 0.015,
                'uniaxial_anisotropy': 0.8e6,
                'exchange_constant': 20e-12,
                'spin_hall_angle': 0.2,
                'resistance_parallel': 500,
                'resistance_antiparallel': 1000,
                'easy_axis': [0, 0, 1]
            }
        elif device_type == 'vcma_mram':
            return {
                'volume': 80e-9 * 80e-9 * 1.5e-9,  # 80×80×1.5 nm³
                'area': 80e-9 * 80e-9,
                'thickness': 1.5e-9,
                'saturation_magnetization': 800e3,
                'damping': 0.008,
                'uniaxial_anisotropy': 1.5e6,
                'exchange_constant': 20e-12,
                'vcma_coefficient': 100e-6,  # J/V·m
                'resistance_parallel': 2e3,
                'resistance_antiparallel': 4e3,
                'easy_axis': [0, 0, 1]
            }
        elif device_type == 'skyrmion':
            return {
                'volume': 200e-9 * 50e-9 * 0.5e-9,  # Track: 200×50×0.5 nm³
                'area': 200e-9 * 50e-9,
                'thickness': 0.5e-9,
                'saturation_magnetization': 600e3,
                'damping': 0.02,
                'dmi_constant': 3e-3,  # J/m²
                'exchange_constant': 15e-12,
                'skyrmion_radius': 10e-9,
                'easy_axis': [0, 0, 1]
            }
        else:
            # Generic parameters
            return {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'exchange_constant': 20e-12,
                'polarization': 0.7
            }

    def validate_parameters(
        self,
        device_type: str,
        device_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and complete device parameters.
        
        Args:
            device_type: Device type
            device_params: Parameters to validate
            
        Returns:
            Validated and completed parameters
        """
        device_type = device_type.lower()

        # Get default parameters and update with provided ones
        default_params = self.get_default_parameters(device_type)
        validated_params = default_params.copy()
        validated_params.update(device_params)

        # Device-specific validation
        if device_type == 'stt_mram':
            self._validate_stt_mram_params(validated_params)
        elif device_type == 'sot_mram':
            self._validate_sot_mram_params(validated_params)
        elif device_type == 'vcma_mram':
            self._validate_vcma_mram_params(validated_params)
        elif device_type == 'skyrmion':
            self._validate_skyrmion_params(validated_params)

        return validated_params

    def _validate_stt_mram_params(self, params: Dict[str, Any]) -> None:
        """Validate STT-MRAM specific parameters."""
        required = ['volume', 'saturation_magnetization', 'damping',
                   'uniaxial_anisotropy', 'polarization']

        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter for STT-MRAM: {param}")

        # Validate ranges
        if params['volume'] <= 0:
            raise ValueError("Volume must be positive")
        if params['saturation_magnetization'] <= 0:
            raise ValueError("Saturation magnetization must be positive")
        if not 0 <= params['damping'] <= 1:
            raise ValueError("Damping must be between 0 and 1")
        if not 0 <= params['polarization'] <= 1:
            raise ValueError("Polarization must be between 0 and 1")

    def _validate_sot_mram_params(self, params: Dict[str, Any]) -> None:
        """Validate SOT-MRAM specific parameters."""
        # Similar validation for SOT-MRAM
        pass

    def _validate_vcma_mram_params(self, params: Dict[str, Any]) -> None:
        """Validate VCMA-MRAM specific parameters."""
        # Similar validation for VCMA-MRAM
        pass

    def _validate_skyrmion_params(self, params: Dict[str, Any]) -> None:
        """Validate skyrmion device specific parameters."""
        # Similar validation for skyrmion devices
        pass


# Global device factory instance
device_factory = DeviceFactory()
