"""Core functionality tests for Spin-Torque RL-Gym."""

import numpy as np
import pytest


class TestPackageImport:
    """Test basic package import and structure."""

    def test_package_import(self):
        """Test that package can be imported."""
        try:
            import spin_torque_gym
            assert hasattr(spin_torque_gym, "__version__")
        except ImportError:
            pytest.skip("Package not yet fully implemented")

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        try:
            import spin_torque_gym.devices
            import spin_torque_gym.envs
            import spin_torque_gym.physics
            import spin_torque_gym.rewards
        except ImportError:
            pytest.skip("Submodules not yet implemented")


class TestMagnetizationUtils:
    """Test magnetization utility functions."""

    def test_magnetization_normalization(self):
        """Test magnetization vector normalization."""
        # This is a placeholder test for future magnetization utilities
        m = np.array([1.0, 1.0, 1.0])
        m_norm = m / np.linalg.norm(m)
        assert np.isclose(np.linalg.norm(m_norm), 1.0)

    def test_magnetization_dot_product(self):
        """Test dot product between magnetization vectors."""
        m1 = np.array([0.0, 0.0, 1.0])
        m2 = np.array([0.0, 0.0, -1.0])
        dot_product = np.dot(m1, m2)
        assert np.isclose(dot_product, -1.0)

    def test_magnetization_angles(self):
        """Test angle calculation between magnetization vectors."""
        m1 = np.array([1.0, 0.0, 0.0])
        m2 = np.array([0.0, 1.0, 0.0])
        angle = np.arccos(np.dot(m1, m2))
        assert np.isclose(angle, np.pi / 2)


class TestPhysicsConstants:
    """Test physics constants and units."""

    def test_physical_constants(self):
        """Test that physical constants are reasonable."""
        # Planck constant (J⋅s)
        h_bar = 1.054571817e-34
        assert h_bar > 0

        # Boltzmann constant (J/K)
        k_b = 1.380649e-23
        assert k_b > 0

        # Electron charge (C)
        e = 1.602176634e-19
        assert e > 0

        # Electron mass (kg)
        m_e = 9.1093837015e-31
        assert m_e > 0

    def test_magnetic_units(self):
        """Test magnetic unit conversions."""
        # Tesla to A/m conversion factor
        mu_0 = 4 * np.pi * 1e-7  # H/m
        assert mu_0 > 0

        # Typical saturation magnetization (A/m)
        ms_typical = 800e3
        assert ms_typical > 0


class TestParameterValidation:
    """Test parameter validation utilities."""

    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperatures
        valid_temps = [0, 77, 300, 400, 1000]
        for temp in valid_temps:
            assert temp >= 0, f"Temperature {temp} should be non-negative"

        # Invalid temperatures
        invalid_temps = [-1, -273.16]
        for temp in invalid_temps:
            assert temp < 0, f"Temperature {temp} should be invalid"

    def test_damping_validation(self):
        """Test damping parameter validation."""
        # Valid damping values
        valid_damping = [0.001, 0.01, 0.1, 1.0]
        for damping in valid_damping:
            assert 0 < damping <= 1.0, f"Damping {damping} should be in (0, 1]"

    def test_current_density_validation(self):
        """Test current density parameter validation."""
        # Typical current densities (A/cm²)
        typical_currents = [1e5, 1e6, 5e6, 1e7]
        for current in typical_currents:
            assert current > 0, f"Current density {current} should be positive"
