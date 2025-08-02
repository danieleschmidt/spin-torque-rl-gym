# Testing Guide for Spin-Torque RL-Gym

This document provides comprehensive information about testing in the Spin-Torque RL-Gym project.

## Test Structure

The test suite is organized into several categories:

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions
├── e2e/                 # End-to-end tests for complete workflows
├── benchmarks/          # Performance and benchmark tests
├── fixtures/            # Test data, configurations, and utilities
├── conftest.py          # Pytest configuration and shared fixtures
└── __init__.py
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **`test_core.py`**: Basic functionality and imports
- **`test_devices.py`**: Device model components
- **`test_physics.py`**: Physics simulation components
- **`test_envs.py`**: Environment components (when implemented)
- **`test_rewards.py`**: Reward function components (when implemented)

### Integration Tests (`tests/integration/`)

Test component interactions:

- **`test_environment.py`**: Full environment integration
- **`test_training.py`**: Training loop integration (when implemented)
- **`test_multi_device.py`**: Multi-device system integration (when implemented)

### End-to-End Tests (`tests/e2e/`)

Test complete workflows:

- **`test_training_workflow.py`**: Complete RL training workflows
- **`test_benchmark_comparison.py`**: Experimental validation (when implemented)

### Benchmark Tests (`tests/benchmarks/`)

Performance and scalability tests:

- **`test_performance.py`**: Performance benchmarks
- **`test_scalability.py`**: Scalability tests (when implemented)

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::TestPackageImport::test_package_import
```

### Test Categories with Markers

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only physics-related tests
pytest -m physics

# Run benchmark tests
pytest -m benchmark --benchmark-only
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=spin_torque_gym

# Generate HTML coverage report
pytest --cov=spin_torque_gym --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Parallel Test Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with specific number of processes
pytest -n 4
```

## Test Configuration

### Pytest Configuration

Configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=spin_torque_gym --cov-report=term-missing"
testpaths = ["tests"]
```

### Test Markers

Available test markers:

- `slow`: Long-running tests (typically >5 seconds)
- `physics`: Physics simulation tests
- `integration`: Integration tests  
- `benchmark`: Performance benchmark tests
- `gpu`: Tests requiring GPU acceleration

### Environment Variables

Set during testing (via `conftest.py`):

- `TESTING=true`: Indicates test environment
- `LOG_LEVEL=DEBUG`: Enable debug logging
- `ENABLE_VISUALIZATION=false`: Disable visualization during tests

## Test Fixtures

### Shared Fixtures (`conftest.py`)

Common fixtures available to all tests:

- `test_data_dir`: Path to test data directory
- `temp_dir`: Temporary directory for test files
- `mock_device_params`: Mock device parameters
- `sample_magnetization`: Sample magnetization vector
- `target_magnetization`: Target magnetization vector
- `mock_action`: Mock RL action
- `physics_config`: Physics simulation configuration
- `env_config`: Environment configuration

### Device Configuration Fixtures (`fixtures/device_configs.py`)

Predefined device configurations:

```python
from tests.fixtures.device_configs import get_device_config

# Get standard STT-MRAM configuration
config = get_device_config("stt_mram", "standard")

# Get low-damping variant
config = get_device_config("stt_mram", "low_damping")
```

### Sample Data Fixtures (`fixtures/sample_data.py`)

Generate test data:

```python
from tests.fixtures.sample_data import generate_magnetization_trajectory

# Generate sample trajectory
trajectory = generate_magnetization_trajectory(
    initial=np.array([0, 0, 1]),
    target=np.array([0, 0, -1]),
    num_steps=100
)
```

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Unit Test

```python
import numpy as np
import pytest

class TestMagnetizationUtils:
    """Test magnetization utility functions."""

    def test_magnetization_normalization(self):
        """Test magnetization vector normalization."""
        m = np.array([1.0, 1.0, 1.0])
        m_norm = m / np.linalg.norm(m)
        assert np.isclose(np.linalg.norm(m_norm), 1.0)

    def test_invalid_magnetization(self):
        """Test handling of invalid magnetization."""
        with pytest.raises(ValueError):
            # Test code that should raise ValueError
            pass
```

### Example Integration Test

```python
@pytest.mark.integration
class TestEnvironmentIntegration:
    """Test environment integration."""

    def test_environment_step_cycle(self, env_config):
        """Test complete environment step cycle."""
        # Test would create environment, run step, verify results
        pass
```

### Example Parametric Test

```python
@pytest.mark.parametrize("device_type,expected_resistance", [
    ("stt_mram", 7500),
    ("sot_mram", 15000),
    ("vcma_mram", 22500),
])
def test_device_resistance(device_type, expected_resistance):
    """Test device resistance calculation."""
    # Test implementation
    pass
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:

- Pull requests
- Pushes to main branch
- Scheduled runs (when CI is set up)

### Test Matrix

Tests run on multiple configurations:

- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: Minimum and latest versions

## Performance Testing

### Benchmark Tests

Use `pytest-benchmark` for performance tests:

```python
def test_physics_simulation_speed(benchmark):
    """Benchmark physics simulation speed."""
    
    def simulate_step():
        # Physics simulation code
        return result
    
    result = benchmark(simulate_step)
    assert result is not None
```

### Performance Requirements

Target performance metrics:

- Single device simulation: >1000 steps/second
- 4x4 array simulation: >100 steps/second  
- Memory usage: <100 MB for typical simulations
- Startup time: <2 seconds

## Test Data Management

### Mock Data

Use fixtures for consistent test data:

```python
@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    return {
        "observations": np.random.randn(1000, 10),
        "actions": np.random.randn(1000, 2),
        "rewards": np.random.randn(1000)
    }
```

### External Data

For tests requiring external data:

- Store small datasets in `tests/data/`
- Use data generation functions for larger datasets
- Mock external API calls

## Debugging Tests

### Common Issues

1. **Import errors**: Ensure package is installed in development mode
2. **Path issues**: Use fixtures for file paths
3. **Random failures**: Set random seeds for reproducibility
4. **Slow tests**: Use `@pytest.mark.slow` and run with `-m "not slow"`

### Debugging Commands

```bash
# Run single test with detailed output
pytest tests/unit/test_core.py::test_function -v -s

# Drop into debugger on failure
pytest --pdb

# Run with warnings enabled
pytest -W error

# Show local variables on failure
pytest -l
```

## Test Coverage Goals

### Coverage Targets

- Overall coverage: >90%
- Core modules: >95%
- New features: 100%
- Critical paths: 100%

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=spin_torque_gym --cov-report=html

# Check coverage of specific module
pytest --cov=spin_torque_gym.physics --cov-report=term-missing
```

## Contributing Tests

### Test Requirements for New Features

1. Unit tests for all new functions/classes
2. Integration tests for component interactions
3. Performance tests for computationally intensive code
4. Documentation tests for public APIs

### Test Review Checklist

- [ ] Tests cover all new functionality
- [ ] Tests include edge cases and error conditions  
- [ ] Tests are properly categorized with markers
- [ ] Tests run quickly (or marked as slow)
- [ ] Tests are deterministic and reproducible
- [ ] Test names clearly describe what is being tested

## Advanced Testing Patterns

### Property-Based Testing

Consider using Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=-1, max_value=1, width=32))
def test_magnetization_bounds(value):
    """Test magnetization component bounds."""
    assert -1 <= value <= 1
```

### Mock External Dependencies

Use `unittest.mock` for external dependencies:

```python
from unittest.mock import patch, Mock

@patch('spin_torque_gym.external_lib.function')
def test_with_mocked_dependency(mock_function):
    """Test with mocked external function."""
    mock_function.return_value = "mocked_result"
    # Test implementation
```

### Testing Async Code

For future async functionality:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    result = await async_function()
    assert result is not None
```