"""Performance benchmark tests."""

import numpy as np
import pytest


@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_single_device_simulation_speed(self, benchmark, mock_device_params):
        """Benchmark single device simulation speed."""

        def simulate_device_steps():
            # Mock simulation of 1000 physics steps
            dt = 1e-12  # s
            num_steps = 1000

            # Simulate magnetization evolution
            m = np.array([0.0, 0.0, 1.0])
            for step in range(num_steps):
                # Simple precession (mock physics)
                omega = 1e9  # rad/s
                theta = omega * dt * step
                m = np.array([np.sin(theta) * 0.1, 0, np.cos(theta)])
                m = m / np.linalg.norm(m)

            return m

        # Benchmark the simulation
        result = benchmark(simulate_device_steps)

        # Verify result is normalized magnetization
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_array_device_simulation_speed(self, benchmark):
        """Benchmark array device simulation speed."""

        def simulate_device_array():
            # Mock simulation of 4x4 device array
            array_size = (4, 4)
            num_steps = 100

            # Initialize array
            magnetizations = np.zeros((*array_size, 3))
            magnetizations[:, :, 2] = 1.0  # All pointing up initially

            # Simulate evolution
            for step in range(num_steps):
                # Mock physics with some coupling
                for i in range(array_size[0]):
                    for j in range(array_size[1]):
                        # Simple rotation
                        angle = 0.1 * step / num_steps
                        magnetizations[i, j] = [
                            np.sin(angle),
                            0,
                            np.cos(angle)
                        ]

            return magnetizations

        result = benchmark(simulate_device_array)
        assert result.shape == (4, 4, 3)

    def test_reward_calculation_speed(self, benchmark, sample_magnetization, target_magnetization):
        """Benchmark reward calculation speed."""

        def calculate_composite_reward():
            m_current = sample_magnetization
            m_target = target_magnetization

            # Multiple reward components
            alignment = np.dot(m_current, m_target)
            alignment_reward = 10.0 if alignment > 0.95 else 0.0

            energy_consumed = 1e-12  # J
            energy_penalty = -0.1 * energy_consumed / 1e-12

            switching_time = 1e-9  # s
            speed_bonus = 1.0 / (1.0 + switching_time / 1e-9)

            return alignment_reward + energy_penalty + speed_bonus

        result = benchmark(calculate_composite_reward)
        assert isinstance(result, float)

    def test_magnetization_operations_speed(self, benchmark):
        """Benchmark magnetization vector operations."""

        def magnetization_operations():
            # Generate random magnetization vectors
            num_vectors = 10000
            vectors = np.random.normal(0, 1, (num_vectors, 3))

            # Normalize all vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / norms

            # Calculate all pairwise dot products
            dot_products = np.dot(normalized, normalized.T)

            # Find closest pairs
            np.fill_diagonal(dot_products, -2)  # Exclude self-pairs
            max_alignments = np.max(dot_products, axis=1)

            return max_alignments

        result = benchmark(magnetization_operations)
        assert len(result) == 10000
        assert np.all(result <= 1.0)  # Dot products ≤ 1

    def test_physics_integration_speed(self, benchmark, physics_config):
        """Benchmark physics integration performance."""

        def integrate_llgs_equation():
            # Mock LLGS integration
            m = np.array([0.0, 0.0, 1.0])
            dt = physics_config["timestep"]
            num_steps = 1000

            # Integration parameters
            gamma = 2.21e5  # m/A⋅s
            alpha = 0.01

            for step in range(num_steps):
                # Mock effective field
                h_eff = np.array([0.1, 0.0, -0.1])  # Simple field

                # LLG equation terms
                cross1 = np.cross(m, h_eff)
                cross2 = np.cross(m, cross1)

                # Derivative
                dmdt = -gamma * cross1 + alpha * gamma * cross2

                # Euler integration (simplified)
                m = m + dt * dmdt
                m = m / np.linalg.norm(m)

            return m

        result = benchmark(integrate_llgs_equation)
        assert np.isclose(np.linalg.norm(result), 1.0)


@pytest.mark.benchmark
class TestMemoryUsage:
    """Memory usage benchmark tests."""

    def test_single_device_memory_usage(self):
        """Test memory usage for single device simulation."""
        # Monitor memory usage during simulation setup
        initial_arrays = []

        # Device parameters
        device_size = (50e-9, 100e-9, 2e-9)
        initial_arrays.append(np.array(device_size))

        # Magnetization state
        magnetization = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        initial_arrays.append(magnetization)

        # Physics arrays
        num_steps = 10000
        trajectory = np.zeros((num_steps, 3), dtype=np.float64)
        time_array = np.linspace(0, 10e-9, num_steps)

        # Estimate memory usage (rough)
        total_bytes = sum(arr.nbytes for arr in [trajectory, time_array])
        total_mb = total_bytes / 1024**2

        # Should be reasonable for single device
        assert total_mb < 10  # Less than 10 MB

    def test_array_simulation_memory_scaling(self):
        """Test memory scaling for array simulations."""
        array_sizes = [(2, 2), (4, 4), (8, 8)]
        memory_usage = []

        for size in array_sizes:
            num_devices = size[0] * size[1]

            # Magnetization arrays
            magnetizations = np.zeros((*size, 3), dtype=np.float64)

            # History arrays (assuming 1000 steps)
            num_steps = 1000
            history = np.zeros((num_steps, *size, 3), dtype=np.float64)

            # Calculate memory usage
            total_bytes = magnetizations.nbytes + history.nbytes
            total_mb = total_bytes / 1024**2
            memory_usage.append(total_mb)

            # Memory should scale reasonably
            expected_scaling = num_devices

        # Check that memory scales approximately linearly
        ratios = [memory_usage[i+1] / memory_usage[i] for i in range(len(memory_usage)-1)]
        for ratio in ratios:
            assert 3 < ratio < 5  # Approximately 4x scaling


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""

    def test_device_count_scaling(self, benchmark):
        """Test performance scaling with device count."""

        def simulate_n_devices(n_devices):
            # Simulate n independent devices
            results = []
            for i in range(n_devices):
                # Mock device simulation
                m = np.array([0.0, 0.0, 1.0])
                for step in range(100):  # Reduced steps for scaling test
                    # Simple evolution
                    theta = 0.01 * step
                    m = np.array([np.sin(theta), 0, np.cos(theta)])
                    m = m / np.linalg.norm(m)
                results.append(m)
            return results

        # Test different device counts
        device_counts = [1, 4, 16]
        times = []

        for n in device_counts:
            result = benchmark(simulate_n_devices, n)
            # Record timing (would be captured by benchmark fixture)
            assert len(result) == n

        # Note: Actual timing analysis would be done by pytest-benchmark

    def test_timestep_scaling(self, benchmark):
        """Test performance scaling with number of timesteps."""

        def simulate_n_steps(n_steps):
            m = np.array([0.0, 0.0, 1.0])
            dt = 1e-12

            for step in range(n_steps):
                # Mock physics step
                theta = 1e9 * dt * step  # 1 GHz precession
                m = np.array([
                    np.sin(theta) * 0.1,
                    0,
                    np.cos(theta)
                ])
                m = m / np.linalg.norm(m)

            return m

        # Test different step counts
        step_counts = [100, 1000, 10000]

        for n_steps in step_counts:
            result = benchmark(simulate_n_steps, n_steps)
            assert np.isclose(np.linalg.norm(result), 1.0)

    def test_parameter_sweep_scaling(self, benchmark):
        """Test performance for parameter sweeps."""

        def parameter_sweep():
            # Mock parameter sweep over current and duration
            currents = np.linspace(1e5, 5e6, 20)  # A/cm²
            durations = np.linspace(0.1e-9, 10e-9, 20)  # s

            results = np.zeros((len(currents), len(durations)))

            for i, current in enumerate(currents):
                for j, duration in enumerate(durations):
                    # Mock simulation result
                    # Higher current and longer duration = higher success
                    success_prob = np.tanh(current / 1e6) * np.tanh(duration / 1e-9)
                    results[i, j] = success_prob

            return results

        result = benchmark(parameter_sweep)
        assert result.shape == (20, 20)
        assert np.all(0 <= result) and np.all(result <= 1)
