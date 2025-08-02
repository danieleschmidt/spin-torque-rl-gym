"""Pytest configuration and shared fixtures for Spin-Torque RL-Gym tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def mock_device_params():
    """Mock device parameters for testing."""
    return {
        "size": (50e-9, 100e-9, 2e-9),  # 50×100×2 nm³
        "ms": 800e3,  # Saturation magnetization (A/m)
        "damping": 0.01,  # Gilbert damping
        "polarization": 0.7,  # Spin polarization
        "resistance_ap": 10e3,  # Anti-parallel resistance (Ω)
        "resistance_p": 5e3,   # Parallel resistance (Ω)
        "thermal_stability": 60,  # E_b/k_B T
        "temperature": 300,  # Kelvin
    }


@pytest.fixture
def sample_magnetization():
    """Sample magnetization vector for testing."""
    return np.array([0.0, 0.0, 1.0])  # Z-axis aligned


@pytest.fixture
def target_magnetization():
    """Target magnetization vector for testing."""
    return np.array([0.0, 0.0, -1.0])  # Opposite to sample


@pytest.fixture
def mock_action():
    """Mock action for testing."""
    return {
        "current": 1e6,  # A/cm²
        "duration": 1e-9,  # 1 ns
    }


@pytest.fixture
def physics_config():
    """Physics simulation configuration for testing."""
    return {
        "timestep": 1e-12,  # 1 ps
        "max_time": 10e-9,  # 10 ns
        "temperature": 300,  # K
        "include_thermal": False,  # Disable for deterministic tests
        "include_dipolar": False,
        "solver": "rk45",
        "atol": 1e-9,
        "rtol": 1e-6,
    }


@pytest.fixture
def env_config():
    """Environment configuration for testing."""
    return {
        "device_type": "stt_mram",
        "max_steps": 100,
        "success_threshold": 0.95,
        "energy_weight": 0.1,
        "reward_type": "composite",
        "render_mode": None,
    }


# Skip marks for different test categories
pytest_slow = pytest.mark.slow
pytest_physics = pytest.mark.physics
pytest_integration = pytest.mark.integration
pytest_benchmark = pytest.mark.benchmark
pytest_gpu = pytest.mark.gpu


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "physics: marks tests as physics simulation tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test path."""
    for item in items:
        # Add slow marker to integration and benchmark tests
        if "integration" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add physics marker to physics tests
        if "physics" in item.nodeid:
            item.add_marker(pytest.mark.physics)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add benchmark marker to benchmark tests
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["ENABLE_VISUALIZATION"] = "false"
    yield
    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)
    os.environ.pop("ENABLE_VISUALIZATION", None)