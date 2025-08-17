#!/usr/bin/env python3
"""
Coverage Enhancement Test Suite
Targeted tests to boost test coverage for research excellence.
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

def test_environment_factory():
    """Test environment factory and registration."""
    try:
        # Test gymnasium registration
        env = gym.make('SpinTorque-v0')
        assert env is not None
        env.close()
    except:
        # If not registered, test direct import
        from spin_torque_gym.envs.spin_torque_env import SpinTorqueEnv
        env = SpinTorqueEnv()
        assert env is not None


def test_device_creation():
    """Test device creation and validation."""
    from spin_torque_gym.devices import DeviceFactory
    
    # Test STT-MRAM device
    device = DeviceFactory.create_device('stt_mram')
    assert device is not None
    assert hasattr(device, 'volume')
    assert hasattr(device, 'saturation_magnetization')
    
    # Test device parameter validation
    params = {
        'volume': 1e-24,
        'saturation_magnetization': 800e3,
        'damping': 0.01
    }
    device_with_params = DeviceFactory.create_device('stt_mram', params)
    assert device_with_params.volume == 1e-24


def test_physics_integration():
    """Test physics solver integration."""
    from spin_torque_gym.physics.llgs_solver import LLGSSolver
    from spin_torque_gym.physics.simple_solver import SimpleLLGSSolver
    
    # Test simple solver
    solver = SimpleLLGSSolver(dt=1e-12)
    assert solver is not None
    
    # Test evolution step
    m = np.array([0.1, 0.1, 0.99])
    m = m / np.linalg.norm(m)
    
    current = 1e6  # A/m²
    dt = 1e-12     # s
    
    m_new = solver.step(m, current, dt)
    assert len(m_new) == 3
    assert np.allclose(np.linalg.norm(m_new), 1.0, atol=1e-6)


def test_reward_calculations():
    """Test reward function calculations."""
    from spin_torque_gym.rewards.base_reward import BaseReward
    from spin_torque_gym.rewards.composite_reward import CompositeReward
    
    # Test base reward
    reward_func = BaseReward()
    
    state = {
        'magnetization': np.array([0.9, 0.1, 0.3]),
        'target': np.array([1.0, 0.0, 0.0]),
        'energy': 10e-12,
        'time': 1e-9
    }
    
    reward = reward_func.calculate(state)
    assert isinstance(reward, float)
    
    # Test composite reward
    components = {
        'alignment': {'weight': 1.0, 'type': 'alignment'},
        'energy': {'weight': -0.1, 'type': 'energy'},
    }
    
    composite = CompositeReward(components)
    composite_reward = composite.calculate(state)
    assert isinstance(composite_reward, float)


def test_materials_database():
    """Test materials database access."""
    from spin_torque_gym.physics.materials import MaterialDatabase
    
    db = MaterialDatabase()
    
    # Test material retrieval
    cofeb = db.get_material('cofeb')
    assert 'saturation_magnetization' in cofeb
    assert 'exchange_constant' in cofeb
    
    # Test custom material
    custom_params = {
        'saturation_magnetization': 1000e3,
        'exchange_constant': 15e-12,
        'damping_constant': 0.008
    }
    db.add_material('custom_material', custom_params)
    custom = db.get_material('custom_material')
    assert custom['saturation_magnetization'] == 1000e3


def test_monitoring_and_performance():
    """Test monitoring and performance tracking."""
    from spin_torque_gym.utils.monitoring import EnvironmentMonitor
    from spin_torque_gym.utils.performance import PerformanceProfiler
    
    # Test environment monitor
    monitor = EnvironmentMonitor()
    assert monitor is not None
    
    # Test performance profiler
    profiler = PerformanceProfiler()
    assert profiler is not None
    
    # Test basic metric recording
    with profiler.time_context('test_operation'):
        import time
        time.sleep(0.001)  # 1ms operation
    
    stats = profiler.get_stats()
    assert 'test_operation' in stats


def test_validation_utilities():
    """Test input/output validation utilities."""
    from spin_torque_gym.utils.validation import (
        validate_magnetization,
        validate_current_pulse,
        validate_device_parameters
    )
    
    # Test magnetization validation
    valid_m = np.array([0.6, 0.8, 0.0])
    assert validate_magnetization(valid_m)
    
    invalid_m = np.array([2.0, 0.0, 0.0])  # Too large
    assert not validate_magnetization(invalid_m)
    
    # Test current pulse validation
    valid_pulse = {'amplitude': 1e6, 'duration': 1e-9}
    assert validate_current_pulse(valid_pulse)
    
    # Test device parameter validation
    valid_params = {
        'volume': 1e-24,
        'saturation_magnetization': 800e3,
        'damping': 0.01
    }
    assert validate_device_parameters(valid_params)


def test_safety_and_error_handling():
    """Test safety wrappers and error handling."""
    from spin_torque_gym.utils.error_handling import SafeExecutionWrapper
    
    wrapper = SafeExecutionWrapper()
    
    # Test safe function execution
    def safe_function(x):
        return x * 2
    
    result = wrapper.execute_safely(safe_function, 5)
    assert result == 10
    
    # Test error handling
    def error_function():
        raise ValueError("Test error")
    
    result = wrapper.execute_safely(error_function, default_return=None)
    assert result is None


def test_caching_and_optimization():
    """Test caching and optimization utilities."""
    from spin_torque_gym.utils.cache import CacheManager
    
    cache = CacheManager(max_size=100)
    
    # Test cache operations
    cache.set('test_key', 'test_value')
    assert cache.get('test_key') == 'test_value'
    
    # Test cache statistics
    stats = cache.get_stats()
    assert 'hits' in stats
    assert 'misses' in stats


def test_quantum_enhancements():
    """Test quantum computation enhancements."""
    try:
        from spin_torque_gym.quantum.optimization import QuantumOptimizer
        from spin_torque_gym.quantum.error_correction import QuantumErrorCorrection
        
        # Test quantum optimizer
        optimizer = QuantumOptimizer()
        assert optimizer is not None
        
        # Test quantum error correction
        qec = QuantumErrorCorrection()
        assert qec is not None
        
    except ImportError:
        # Quantum features optional
        pytest.skip("Quantum features not available")


def test_advanced_solvers():
    """Test advanced physics solvers."""
    from spin_torque_gym.utils.robust_solver import RobustLLGSSolver
    
    solver = RobustLLGSSolver()
    assert solver is not None
    
    # Test solver methods
    methods = solver.get_available_methods()
    assert 'euler' in methods
    assert 'rk4' in methods


def test_thermal_effects():
    """Test thermal fluctuation modeling."""
    from spin_torque_gym.physics.thermal_model import ThermalFluctuations
    
    thermal = ThermalFluctuations(temperature=300.0)
    assert thermal is not None
    
    # Test thermal field generation
    thermal_field = thermal.generate_field(np.array([1.0, 0.0, 0.0]))
    assert len(thermal_field) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])