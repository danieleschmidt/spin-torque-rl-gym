"""Comprehensive test suite for SpinTorque Gym.

This module provides extensive testing including unit tests,
integration tests, performance benchmarks, and safety validation.
"""

import numpy as np
import pytest
import gymnasium as gym
import time
from typing import Dict, List, Any
import warnings

# Suppress physics warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)

import spin_torque_gym
from spin_torque_gym.utils.performance import get_optimizer
from spin_torque_gym.utils.monitoring import EnvironmentMonitor


class TestEnvironmentCreation:
    """Test environment creation and basic functionality."""

    def test_environment_registration(self):
        """Test that environments are properly registered."""
        env_ids = ['SpinTorque-v0', 'SpinTorqueArray-v0', 'SkyrmionRacetrack-v0']
        
        for env_id in env_ids:
            env = gym.make(env_id)
            assert env is not None
            env.close()

    def test_basic_environment_cycle(self):
        """Test basic environment reset and step cycle."""
        env = gym.make('SpinTorque-v0')
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (12,), f"Expected obs shape (12,), got {obs.shape}"
        assert isinstance(info, dict)
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (12,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()

    def test_action_space_validation(self):
        """Test action space boundaries and validation."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        # Test valid actions
        valid_actions = [
            [1e6, 1e-9],   # Normal current and duration
            [0, 1e-12],    # Zero current
            [-1e6, 5e-9],  # Negative current
        ]
        
        for action in valid_actions:
            obs, reward, done, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
            assert np.isfinite(reward), "Reward is non-finite"
        
        env.close()

    def test_observation_space_validation(self):
        """Test observation space structure and values."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        # Test observation components
        magnetization = obs[:3]
        target = obs[3:6]
        resistance = obs[6]
        temperature = obs[7]
        steps_remaining = obs[8]
        energy = obs[9]
        
        # Validate magnetization is normalized
        mag_norm = np.linalg.norm(magnetization)
        assert abs(mag_norm - 1.0) < 0.1, f"Magnetization not normalized: {mag_norm}"
        
        # Validate target is normalized
        target_norm = np.linalg.norm(target)
        assert abs(target_norm - 1.0) < 0.1, f"Target not normalized: {target_norm}"
        
        # Validate other components are finite
        assert np.isfinite(resistance), "Resistance is non-finite"
        assert np.isfinite(temperature), "Temperature is non-finite"
        assert 0 <= steps_remaining <= 1, f"Steps remaining out of range: {steps_remaining}"
        
        env.close()


class TestPhysicsSimulation:
    """Test physics simulation accuracy and stability."""

    def test_magnetization_conservation(self):
        """Test that magnetization magnitude is conserved."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        initial_mag = obs[:3]
        initial_norm = np.linalg.norm(initial_mag)
        
        # Run multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            
            current_mag = obs[:3]
            current_norm = np.linalg.norm(current_mag)
            
            # Check conservation (within numerical tolerance)
            assert abs(current_norm - 1.0) < 0.1, f"Magnetization norm not conserved: {current_norm}"
            
            if done or trunc:
                break
        
        env.close()

    def test_energy_conservation_principles(self):
        """Test energy conservation and thermodynamics."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        # Test with zero current (should conserve energy better)
        initial_energy = info.get('total_energy', 0)
        
        # Apply zero current
        obs, reward, done, trunc, info = env.step([0, 1e-12])
        final_energy = info.get('total_energy', 0)
        
        # Energy should not increase significantly without input
        energy_change = abs(final_energy - initial_energy)
        assert energy_change < 1e-11, f"Energy not conserved with zero current: {energy_change}"
        
        env.close()

    def test_thermal_stability(self):
        """Test thermal stability calculations."""
        from spin_torque_gym.physics.thermal_model import ThermalFluctuations
        
        thermal = ThermalFluctuations(temperature=300.0)
        
        # Test stability factor calculation
        volume = 1e-24  # m³
        k_u = 1e6  # J/m³
        
        delta = thermal.compute_thermal_barrier(k_u, volume)
        assert delta > 0, "Thermal stability factor should be positive"
        assert np.isfinite(delta), "Thermal stability factor should be finite"
        
        # Test temperature dependence
        thermal.set_temperature(600.0)  # Double temperature
        delta_hot = thermal.compute_thermal_barrier(k_u, volume)
        assert delta_hot < delta, "Higher temperature should reduce stability"


class TestPerformanceOptimization:
    """Test performance optimization features."""

    def test_caching_efficiency(self):
        """Test that caching improves performance."""
        optimizer = get_optimizer()
        
        # Clear cache
        optimizer.cache.clear()
        
        # Define a slow computation
        def slow_computation(x: float, y: float) -> float:
            time.sleep(0.01)  # Simulate slow operation
            return x * y + np.sin(x)
        
        params = {'x': 1.5, 'y': 2.0}
        
        # First call (cache miss)
        start_time = time.time()
        result1 = optimizer.cached_computation(slow_computation, params, "test")
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = optimizer.cached_computation(slow_computation, params, "test")
        second_call_time = time.time() - start_time
        
        # Verify results are identical
        assert abs(result1 - result2) < 1e-10, "Cached result differs from original"
        
        # Verify caching improves performance
        assert second_call_time < first_call_time * 0.5, "Caching did not improve performance"
        
        # Check cache statistics
        stats = optimizer.cache.get_stats()
        assert stats['hits'] >= 1, "Cache should have at least one hit"

    def test_vectorized_operations(self):
        """Test vectorized operations performance."""
        from spin_torque_gym.utils.vectorized_operations import VectorizedPhysics
        
        vectorized = VectorizedPhysics()
        
        # Test batch normalization
        batch_size = 100
        vectors = np.random.normal(0, 1, (batch_size, 3))
        
        start_time = time.time()
        normalized = vectorized.batch_normalize(vectors)
        vectorized_time = time.time() - start_time
        
        # Test individual normalization
        start_time = time.time()
        normalized_individual = np.array([
            v / np.linalg.norm(v) for v in vectors
        ])
        individual_time = time.time() - start_time
        
        # Verify results are equivalent
        np.testing.assert_allclose(normalized, normalized_individual, rtol=1e-10)
        
        # Verify vectorized is faster for large batches
        if batch_size >= 10:
            assert vectorized_time < individual_time, "Vectorized operation should be faster"

    def test_memory_optimization(self):
        """Test memory-efficient operations."""
        from spin_torque_gym.utils.vectorized_operations import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_gb=0.1)  # Small limit for testing
        
        # Create large data array
        large_data = np.random.normal(0, 1, (10000, 3))
        
        # Test chunked operation
        def normalize_operation(chunk):
            return chunk / np.linalg.norm(chunk, axis=1, keepdims=True)
        
        result = optimizer.chunked_batch_operation(large_data, normalize_operation)
        
        # Verify shape and normalization
        assert result.shape == large_data.shape
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""

    def test_physics_error_recovery(self):
        """Test physics error recovery mechanisms."""
        # Test with invalid initial condition
        env = gym.make('SpinTorque-v0')
        
        # Force an error condition by creating extreme action
        obs, info = env.reset()
        
        # Extreme action that might cause numerical issues
        extreme_action = [1e15, 1e-15]  # Very high current, very short duration
        
        # This should either work or fail gracefully
        try:
            obs, reward, done, trunc, info = env.step(extreme_action)
            
            # If it works, verify observation is valid
            assert np.all(np.isfinite(obs)), "Observation should be finite"
            assert np.isfinite(reward), "Reward should be finite"
            
        except Exception as e:
            # If it fails, it should be a controlled failure
            assert isinstance(e, (ValueError, RuntimeError)), f"Unexpected exception type: {type(e)}"
        
        env.close()

    def test_safety_constraints(self):
        """Test safety constraint enforcement."""
        from spin_torque_gym.utils.error_handling import SafetyWrapper
        from spin_torque_gym.utils.monitoring import EnvironmentMonitor
        
        monitor = EnvironmentMonitor()
        safety = SafetyWrapper(monitor)
        
        # Test action validation
        safe_action = safety.validate_action([1e6, 1e-9])
        assert np.all(np.isfinite(safe_action)), "Safe action should be finite"
        
        # Test extreme action clamping
        extreme_action = [1e20, -1e-20]
        clamped_action = safety.validate_action(extreme_action)
        assert abs(clamped_action[0]) <= 1e9, "Current should be clamped to safe range"
        assert clamped_action[1] >= 1e-15, "Duration should be clamped to minimum"

    def test_error_statistics(self):
        """Test error counting and statistics."""
        from spin_torque_gym.utils.error_handling import ErrorHandler
        
        handler = ErrorHandler()
        stats = handler.get_error_stats()
        
        assert isinstance(stats, dict)
        assert 'total_errors' in stats
        assert 'error_types' in stats
        assert isinstance(stats['total_errors'], int)


class TestScalability:
    """Test scalability and performance under load."""

    def test_parallel_environment_creation(self):
        """Test creating multiple environments simultaneously."""
        n_envs = 5
        envs = []
        
        try:
            # Create multiple environments
            for i in range(n_envs):
                env = gym.make('SpinTorque-v0')
                envs.append(env)
            
            # Test parallel reset
            observations = []
            for env in envs:
                obs, info = env.reset()
                observations.append(obs)
            
            # Verify all observations are valid and different
            for obs in observations:
                assert obs.shape == (12,)
                assert np.all(np.isfinite(obs))
            
            # Test parallel stepping
            actions = [[1e6, 1e-9] for _ in range(n_envs)]
            rewards = []
            
            for env, action in zip(envs, actions):
                obs, reward, done, trunc, info = env.step(action)
                rewards.append(reward)
            
            # Verify all rewards are finite
            assert all(np.isfinite(r) for r in rewards)
            
        finally:
            # Clean up
            for env in envs:
                env.close()

    def test_memory_usage_stability(self):
        """Test memory usage remains stable during extended operation."""
        env = gym.make('SpinTorque-v0')
        
        # Run extended episode
        obs, info = env.reset()
        initial_obs_mean = np.mean(obs)
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            
            # Verify no memory corruption (observations should remain reasonable)
            assert np.all(np.isfinite(obs)), f"Non-finite observation at step {step}"
            assert abs(np.mean(obs)) < 1000, f"Observation values too large at step {step}"
            
            if done or trunc:
                obs, info = env.reset()
        
        env.close()

    def test_performance_benchmarks(self):
        """Benchmark environment performance."""
        env = gym.make('SpinTorque-v0')
        
        # Warm up
        obs, info = env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        
        # Benchmark reset time
        reset_times = []
        for _ in range(10):
            start_time = time.time()
            obs, info = env.reset()
            reset_times.append(time.time() - start_time)
        
        avg_reset_time = np.mean(reset_times)
        assert avg_reset_time < 0.1, f"Reset too slow: {avg_reset_time:.3f}s"
        
        # Benchmark step time
        step_times = []
        for _ in range(50):
            action = env.action_space.sample()
            start_time = time.time()
            obs, reward, done, trunc, info = env.step(action)
            step_times.append(time.time() - start_time)
            
            if done or trunc:
                env.reset()
        
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.05, f"Step too slow: {avg_step_time:.3f}s"
        
        print(f"Performance: Reset {avg_reset_time:.4f}s, Step {avg_step_time:.4f}s")
        
        env.close()


class TestPhysicsAccuracy:
    """Test physics simulation accuracy and correctness."""

    def test_llgs_conservation_laws(self):
        """Test LLGS equation conservation properties."""
        from spin_torque_gym.physics.simple_solver import SimpleLLGSSolver
        
        solver = SimpleLLGSSolver()
        
        # Test magnetization magnitude conservation
        m_initial = np.array([0.1, 0.2, 0.97])  # Nearly z-aligned
        m_initial = m_initial / np.linalg.norm(m_initial)
        
        device_params = {
            'damping': 0.01,
            'saturation_magnetization': 800e3,
            'uniaxial_anisotropy': 1e6,
            'volume': 1e-24,
            'easy_axis': np.array([0, 0, 1])
        }
        
        result = solver.solve(
            m_initial,
            (0, 1e-10),
            device_params
        )
        
        assert result['success'], f"Solver failed: {result['message']}"
        
        # Check magnitude conservation
        magnitudes = np.linalg.norm(result['m'], axis=1)
        assert np.all(np.abs(magnitudes - 1.0) < 0.01), "Magnetization magnitude not conserved"

    def test_thermal_fluctuations(self):
        """Test thermal fluctuation models."""
        from spin_torque_gym.physics.thermal_model import ThermalFluctuations
        
        thermal = ThermalFluctuations(temperature=300.0, seed=42)
        
        # Test noise generation
        damping = 0.01
        ms = 800e3
        volume = 1e-24
        dt = 1e-12
        
        # Generate thermal fields
        fields = []
        for _ in range(1000):
            field = thermal.generate_thermal_field(damping, ms, volume, dt)
            fields.append(field)
        
        fields = np.array(fields)
        
        # Statistical tests
        mean_field = np.mean(fields, axis=0)
        std_field = np.std(fields, axis=0)
        
        # Mean should be near zero (unbiased noise)
        assert np.all(np.abs(mean_field) < 0.1 * np.mean(std_field)), "Thermal noise is biased"
        
        # Standard deviation should match theoretical expectation
        theoretical_std = thermal.compute_noise_strength(damping, ms, volume)
        assert np.all(np.abs(std_field - theoretical_std) < 0.2 * theoretical_std), "Thermal noise strength incorrect"

    def test_anisotropy_fields(self):
        """Test magnetic anisotropy field calculations."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        # Test alignment with easy axis should give maximum anisotropy energy
        # This is indirectly tested through environment behavior
        
        # Apply action that should align with easy axis
        align_action = [5e6, 2e-9]  # Strong positive current
        
        initial_alignment = obs[2]  # z-component of magnetization
        
        obs, reward, done, trunc, info = env.step(align_action)
        final_alignment = obs[2]
        
        # Check that alignment changed (physics is working)
        alignment_change = abs(final_alignment - initial_alignment)
        assert alignment_change > 1e-6, "Magnetization did not respond to current"
        
        env.close()


class TestSafetyAndSecurity:
    """Test safety constraints and security measures."""

    def test_input_sanitization(self):
        """Test input sanitization and validation."""
        from spin_torque_gym.utils.security import get_security_manager
        
        security = get_security_manager()
        
        # Test numeric sanitization
        safe_num = security.sanitizer.sanitize_numeric("1e6", min_value=0, max_value=1e8)
        assert safe_num == 1e6
        
        # Test out-of-range clamping
        clamped = security.sanitizer.sanitize_numeric(1e10, min_value=0, max_value=1e8)
        assert clamped == 1e8
        
        # Test array sanitization
        safe_array = security.sanitizer.sanitize_array([1, 2, 3], max_size=10)
        assert safe_array.shape == (3,)
        assert np.all(np.isfinite(safe_array))

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        from spin_torque_gym.utils.security import get_security_manager
        
        security = get_security_manager()
        
        # Reset rate limiter for test
        security.rate_limiter.calls.clear()
        
        # Test normal operation within limits
        for i in range(50):
            allowed = security.rate_limiter.is_allowed("test_client")
            assert allowed, f"Request {i} should be allowed"
        
        # Test rate limit enforcement with high request rate
        security.rate_limiter.calls["test_client2"] = deque([time.time()] * 1000)
        blocked = security.rate_limiter.is_allowed("test_client2")
        assert not blocked, "Request should be blocked due to rate limit"

    def test_parameter_bounds(self):
        """Test parameter boundary enforcement."""
        env = gym.make('SpinTorque-v0')
        obs, info = env.reset()
        
        # Test with parameters at boundary conditions
        boundary_actions = [
            [1e9, 1e-15],    # Maximum current, minimum duration
            [-1e9, 10e-9],   # Minimum current, maximum duration
            [0, 1e-9],       # Zero current
        ]
        
        for action in boundary_actions:
            obs, reward, done, trunc, info = env.step(action)
            
            # Should not crash and should return valid values
            assert np.all(np.isfinite(obs)), f"Invalid observation with action {action}"
            assert np.isfinite(reward), f"Invalid reward with action {action}"
            
            if done or trunc:
                env.reset()
        
        env.close()


class TestIntegration:
    """Integration tests for complete system functionality."""

    def test_full_training_episode(self):
        """Test complete training episode from start to finish."""
        env = gym.make('SpinTorque-v0')
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            # Simple policy: apply current proportional to misalignment
            magnetization = obs[:3]
            target = obs[3:6]
            alignment = np.dot(magnetization, target)
            
            # Apply corrective current
            current = 2e6 * (1 - alignment)  # Stronger current for larger misalignment
            duration = 1e-9
            action = [current, duration]
            
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Verify step is valid
            assert np.all(np.isfinite(obs)), f"Invalid observation at step {steps}"
            assert np.isfinite(reward), f"Invalid reward at step {steps}"
            
            if done:
                success = info.get('is_success', False)
                if success:
                    print(f"✓ Training episode completed successfully in {steps} steps")
                break
            
            if trunc:
                print(f"Episode truncated after {steps} steps")
                break
        
        env.close()
        
        # Verify reasonable total reward
        assert total_reward > -1000, f"Total reward too negative: {total_reward}"

    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        monitor = EnvironmentMonitor()
        
        env = gym.make('SpinTorque-v0')
        
        # Monitor environment lifecycle
        monitor.on_environment_created(env, {})
        obs, info = env.reset()
        monitor.on_episode_start(env, obs, info)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            monitor.on_step(env, action, obs, reward, done, trunc, info)
            
            if done or trunc:
                monitor.on_episode_end(env, obs, reward, done, trunc, info)
                break
        
        # Get monitoring statistics
        stats = monitor.get_statistics()
        assert isinstance(stats, dict)
        assert 'episode_count' in stats
        
        env.close()


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running comprehensive SpinTorque Gym test suite...")
    
    # Use pytest to run tests
    import subprocess
    import sys
    
    # Run pytest on this file
    result = subprocess.run([
        sys.executable, '-m', 'pytest', __file__, '-v'
    ], capture_output=True, text=True)
    
    print("Test Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Setup test environment
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_comprehensive_tests()
    
    if success:
        print("✓ All comprehensive tests passed!")
    else:
        print("✗ Some tests failed!")
        exit(1)