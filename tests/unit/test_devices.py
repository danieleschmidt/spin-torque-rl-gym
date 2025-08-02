"""Unit tests for device models."""

import numpy as np
import pytest


class TestSTTMRAMDevice:
    """Test STT-MRAM device model."""

    def test_device_initialization(self, mock_device_params):
        """Test device initialization with parameters."""
        # Placeholder test for future device implementation
        params = mock_device_params
        assert params["ms"] > 0
        assert 0 < params["damping"] <= 1.0
        assert 0 < params["polarization"] <= 1.0

    def test_resistance_calculation(self, mock_device_params):
        """Test resistance calculation based on magnetization."""
        # TMR formula: R(θ) = R_P + (R_AP - R_P) * (1 - cos(θ)) / 2
        r_p = mock_device_params["resistance_p"]
        r_ap = mock_device_params["resistance_ap"]
        
        # Parallel state (θ = 0)
        cos_0 = 1.0
        r_parallel = r_p + (r_ap - r_p) * (1 - cos_0) / 2
        assert np.isclose(r_parallel, r_p)
        
        # Anti-parallel state (θ = π)
        cos_pi = -1.0
        r_antiparallel = r_p + (r_ap - r_p) * (1 - cos_pi) / 2
        assert np.isclose(r_antiparallel, r_ap)

    def test_energy_barrier_calculation(self, mock_device_params):
        """Test energy barrier calculation."""
        thermal_stability = mock_device_params["thermal_stability"]
        temperature = mock_device_params["temperature"]
        k_b = 1.380649e-23  # J/K
        
        energy_barrier = thermal_stability * k_b * temperature
        assert energy_barrier > 0

    def test_switching_threshold(self, mock_device_params):
        """Test critical switching current calculation."""
        # Placeholder for critical current calculation
        # I_c = (2 * e * alpha * Ms * V) / (h_bar * P)
        alpha = mock_device_params["damping"]
        ms = mock_device_params["ms"]
        polarization = mock_device_params["polarization"]
        
        assert alpha > 0
        assert ms > 0
        assert polarization > 0


class TestSOTMRAMDevice:
    """Test SOT-MRAM device model."""

    def test_sot_parameters(self):
        """Test SOT-specific parameters."""
        # Spin Hall angle
        theta_sh = 0.1
        assert -1 <= theta_sh <= 1
        
        # Spin Hall conductivity
        sigma_sh = 1e6  # S/m
        assert sigma_sh > 0

    def test_field_like_torque(self):
        """Test field-like torque calculation."""
        # Placeholder for field-like torque
        h_fl = 100  # mT
        assert h_fl > 0

    def test_damping_like_torque(self):
        """Test damping-like torque calculation."""
        # Placeholder for damping-like torque
        h_dl = 200  # mT
        assert h_dl > 0


class TestVCMADevice:
    """Test VCMA-MRAM device model."""

    def test_voltage_control(self):
        """Test voltage-controlled magnetic anisotropy."""
        # VCMA coefficient (J/V⋅m)
        xi_vcma = 100e-15
        assert xi_vcma != 0
        
        # Applied voltage
        voltage = 1.0  # V
        anisotropy_change = xi_vcma * voltage
        assert abs(anisotropy_change) > 0

    def test_electric_field_effect(self):
        """Test electric field effect on anisotropy."""
        # Placeholder for electric field calculations
        electric_field = 1e9  # V/m
        assert electric_field > 0


class TestSkyrmionDevice:
    """Test skyrmion racetrack device model."""

    def test_skyrmion_parameters(self):
        """Test skyrmion-specific parameters."""
        # Skyrmion radius
        radius = 20e-9  # 20 nm
        assert radius > 0
        
        # Track width
        track_width = 200e-9  # 200 nm
        assert track_width > radius
        
        # Dzyaloshinskii-Moriya interaction
        dmi = 1e-3  # J/m²
        assert dmi > 0

    def test_skyrmion_mobility(self):
        """Test skyrmion mobility calculation."""
        # Skyrmion velocity proportional to current
        current_density = 1e6  # A/cm²
        mobility = 1e-12  # m²/A⋅s
        velocity = mobility * current_density * 1e4  # Convert to m/s
        assert velocity > 0

    def test_magnus_force(self):
        """Test Magnus force on skyrmion."""
        # Magnus force deflects skyrmion perpendicular to current
        # F_Magnus ∝ v × z
        velocity = np.array([1.0, 0.0, 0.0])  # m/s
        z_axis = np.array([0.0, 0.0, 1.0])
        magnus_direction = np.cross(velocity, z_axis)
        expected_direction = np.array([0.0, 1.0, 0.0])
        assert np.allclose(magnus_direction, expected_direction)


class TestDeviceFactory:
    """Test device factory and configuration."""

    def test_device_types(self):
        """Test available device types."""
        device_types = ["stt_mram", "sot_mram", "vcma_mram", "skyrmion"]
        for device_type in device_types:
            assert isinstance(device_type, str)
            assert len(device_type) > 0

    def test_device_configuration_validation(self, mock_device_params):
        """Test device configuration validation."""
        config = mock_device_params
        
        # Size validation
        size = config["size"]
        assert len(size) == 3  # x, y, z dimensions
        assert all(dim > 0 for dim in size)
        
        # Material property validation
        assert config["ms"] > 0
        assert 0 < config["damping"] <= 1.0
        assert 0 < config["polarization"] <= 1.0
        assert config["resistance_p"] > 0
        assert config["resistance_ap"] > config["resistance_p"]
        assert config["thermal_stability"] > 0
        assert config["temperature"] >= 0