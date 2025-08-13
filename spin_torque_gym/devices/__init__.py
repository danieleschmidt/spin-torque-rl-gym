"""Spintronic device models for RL training.

This module implements various spintronic device models including:
- STT-MRAM (Spin-Transfer Torque MRAM)
- SOT-MRAM (Spin-Orbit Torque MRAM)
- VCMA-MRAM (Voltage-Controlled Magnetic Anisotropy MRAM)
- Skyrmion devices
"""

from .base_device import BaseSpintronicDevice
from .device_factory import DeviceFactory, device_factory
from .skyrmion_device import SkyrmionDevice
from .sot_mram import SOTMRAMDevice
from .stt_mram import STTMRAMDevice
from .vcma_mram import VCMAMRAMDevice


def create_device(device_type: str, device_params=None):
    """Create a spintronic device instance.
    
    Args:
        device_type: Type of device to create ('stt_mram', 'sot_mram', 'vcma_mram', 'skyrmion')
        device_params: Optional parameters dict, uses defaults if None
        
    Returns:
        Device instance
    """
    if device_params is None:
        return device_factory.create_default_device(device_type)
    return device_factory.create_device(device_type, device_params)


__all__ = [
    "BaseSpintronicDevice",
    "STTMRAMDevice",
    "SOTMRAMDevice",
    "VCMAMRAMDevice",
    "SkyrmionDevice",
    "DeviceFactory",
    "create_device"
]
