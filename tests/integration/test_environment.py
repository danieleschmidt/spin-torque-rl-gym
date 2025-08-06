"""Integration tests for Gymnasium environments."""

import numpy as np
import pytest
import gymnasium as gym

from spin_torque_gym.envs.spin_torque_env import SpinTorqueEnv
from spin_torque_gym.envs.array_env import SpinTorqueArrayEnv
from spin_torque_gym.envs.skyrmion_env import SkyrmionRacetrackEnv


@pytest.mark.integration
class TestSpinTorqueEnvironment:
    """Test basic Spin-Torque RL environment integration."""

    @pytest.fixture
    def env_config(self):
        """Basic environment configuration."""
        return {
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1]),
                'reference_magnetization': np.array([0, 0, 1]),
                'resistance_parallel': 1e3,
                'resistance_antiparallel': 2e3
            },
            'max_steps': 100,
            'reward_type': 'alignment'
        }
    
    @pytest.fixture
    def env(self, env_config):
        """Create environment instance."""
        return SpinTorqueEnv(**env_config)
    
    def test_environment_creation(self, env):
        """Test environment can be created successfully."""
        assert isinstance(env, SpinTorqueEnv)
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
    
    def test_gymnasium_interface_compliance(self, env):
        """Test Gymnasium interface compliance."""
        # Check required attributes
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        
        # Check action space
        assert hasattr(env.action_space, 'shape')
        assert hasattr(env.action_space, 'sample')
        
        # Check observation space
        assert hasattr(env.observation_space, 'shape')
        assert hasattr(env.observation_space, 'sample')
        
        # Test reset
        obs, info = env.reset()
        assert obs in env.observation_space
        assert isinstance(info, dict)
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs in env.observation_space
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reset_functionality(self, env):
        """Test reset functionality."""
        # First reset
        obs1, info1 = env.reset(seed=42)
        assert obs1 in env.observation_space
        assert isinstance(info1, dict)
        
        # Take some steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        
        # Reset again with same seed
        obs2, info2 = env.reset(seed=42)
        
        # Should return to same initial state
        assert np.allclose(obs1, obs2, rtol=1e-6)
    
    def test_deterministic_behavior(self, env):
        """Test deterministic behavior with fixed seed."""
        # Set seed and take actions
        env.reset(seed=123)
        
        actions = []
        observations = []
        rewards = []
        
        for i in range(10):
            action = np.array([1e6 + i * 1e5, 1e-9])  # Deterministic actions
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            observations.append(obs)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Reset with same seed and repeat
        env.reset(seed=123)
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.allclose(observations[i], obs, rtol=1e-6)
            assert np.isclose(rewards[i], reward, rtol=1e-6)
            
            if terminated or truncated:
                break
    
    def test_episode_length_limits(self, env):
        """Test episode length limits."""
        env.reset()
        
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Episode should end within max_steps
            if terminated or truncated:
                break
            
            # Safety check
            assert steps <= env.max_steps + 1
        
        assert steps <= env.max_steps
    
    def test_reward_bounds(self, env):
        """Test reward bounds are reasonable."""
        env.reset()
        rewards = []
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            
            # Reward should be finite
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break
        
        # Rewards should be in reasonable range
        rewards = np.array(rewards)
        assert np.all(rewards >= -1000)  # Not too negative
        assert np.all(rewards <= 1000)   # Not too positive


@pytest.mark.integration
class TestMultipleDeviceTypes:
    """Test integration across different device types."""
    
    def test_stt_mram_environment(self):
        """Test STT-MRAM environment."""
        config = {
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 50
        }
        
        env = SpinTorqueEnv(**config)
        obs, info = env.reset()
        
        # Test a few steps
        for _ in range(5):
            action = np.array([5e6, 1e-9])  # Current density, duration
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break
    
    def test_sot_mram_environment(self):
        """Test SOT-MRAM environment."""
        config = {
            'device_type': 'sot_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'easy_axis': np.array([0, 0, 1]),
                'spin_hall_angle': 0.1
            },
            'max_steps': 50
        }
        
        env = SpinTorqueEnv(**config)
        obs, info = env.reset()
        
        # Test a few steps
        for _ in range(5):
            action = np.array([1e7, 1e-9])  # Higher current needed for SOT
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break
    
    def test_vcma_mram_environment(self):
        """Test VCMA-MRAM environment."""
        config = {
            'device_type': 'vcma_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'easy_axis': np.array([0, 0, 1]),
                'vcma_coefficient': 100e-6
            },
            'max_steps': 50
        }
        
        env = SpinTorqueEnv(**config)
        obs, info = env.reset()
        
        # Test a few steps
        for _ in range(5):
            action = np.array([1.5, 1e-9])  # Voltage, duration
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break


@pytest.mark.integration
class TestArrayEnvironment:
    """Test array environment integration."""
    
    @pytest.fixture
    def array_config(self):
        """Array environment configuration."""
        return {
            'array_size': (3, 3),
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 50,
            'coupling_strength': 1e-24
        }
    
    @pytest.fixture
    def array_env(self, array_config):
        """Create array environment instance."""
        return SpinTorqueArrayEnv(**array_config)
    
    def test_array_environment_creation(self, array_env):
        """Test array environment creation."""
        assert isinstance(array_env, SpinTorqueArrayEnv)
        assert array_env.array_size == (3, 3)
        assert array_env.num_devices == 9
    
    def test_array_observation_space(self, array_env):
        """Test array observation space."""
        obs, info = array_env.reset()
        
        # Check observation structure
        assert 'magnetizations' in obs
        assert 'target_pattern' in obs
        assert 'resistances' in obs
        
        # Check shapes
        assert obs['magnetizations'].shape == (3, 3, 3)  # (rows, cols, xyz)
        assert obs['target_pattern'].shape == (3, 3, 3)
        assert obs['resistances'].shape == (3, 3)
        
        # Check normalization
        mags = obs['magnetizations']
        norms = np.linalg.norm(mags, axis=-1)
        assert np.allclose(norms, 1.0, rtol=1e-6)
    
    def test_array_action_space(self, array_env):
        """Test array action space."""
        # Check action space structure
        assert hasattr(array_env.action_space, 'shape')
        
        # Test sampling
        action = array_env.action_space.sample()
        assert action in array_env.action_space
    
    def test_array_step_functionality(self, array_env):
        """Test array environment step."""
        obs, info = array_env.reset()
        
        # Test different action types
        # Row action
        action = {'type': 'row', 'index': 1, 'current': 5e6, 'duration': 1e-9}
        obs, reward, terminated, truncated, info = array_env.step(action)
        assert np.isfinite(reward)
        
        # Column action
        action = {'type': 'column', 'index': 2, 'current': 5e6, 'duration': 1e-9}
        obs, reward, terminated, truncated, info = array_env.step(action)
        assert np.isfinite(reward)
        
        # Individual device action
        action = {'type': 'individual', 'row': 0, 'col': 1, 'current': 5e6, 'duration': 1e-9}
        obs, reward, terminated, truncated, info = array_env.step(action)
        assert np.isfinite(reward)
    
    def test_coupling_effects(self, array_env):
        """Test inter-device coupling effects."""
        obs, info = array_env.reset()
        
        # Apply action to central device
        action = {'type': 'individual', 'row': 1, 'col': 1, 'current': 1e7, 'duration': 1e-9}
        obs_before = obs['magnetizations'].copy()
        
        obs_after, reward, terminated, truncated, info = array_env.step(action)
        
        # Central device should change more than edge devices
        change_center = np.linalg.norm(obs_after['magnetizations'][1, 1] - obs_before[1, 1])
        change_corner = np.linalg.norm(obs_after['magnetizations'][0, 0] - obs_before[0, 0])
        
        # With coupling, corner device should also change but less than center
        assert change_center > change_corner


@pytest.mark.integration
class TestSkyrmionEnvironment:
    """Test skyrmion racetrack environment integration."""
    
    @pytest.fixture
    def skyrmion_config(self):
        """Skyrmion environment configuration."""
        return {
            'track_length': 1e-6,  # 1 μm
            'track_width': 200e-9,  # 200 nm
            'num_skyrmions': 3,
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'exchange_constant': 2e-11,
                'dmi_constant': 3e-3,
                'skyrmion_radius': 20e-9,
                'track_width': 200e-9,
                'spin_hall_angle': 0.1
            },
            'max_steps': 50
        }
    
    @pytest.fixture
    def skyrmion_env(self, skyrmion_config):
        """Create skyrmion environment instance."""
        return SkyrmionRacetrackEnv(**skyrmion_config)
    
    def test_skyrmion_environment_creation(self, skyrmion_env):
        """Test skyrmion environment creation."""
        assert isinstance(skyrmion_env, SkyrmionRacetrackEnv)
        assert skyrmion_env.track_length == 1e-6
        assert skyrmion_env.track_width == 200e-9
        assert skyrmion_env.num_skyrmions == 3
    
    def test_skyrmion_observation_space(self, skyrmion_env):
        """Test skyrmion observation space."""
        obs, info = skyrmion_env.reset()
        
        # Check observation structure
        assert 'skyrmion_positions' in obs
        assert 'track_field' in obs
        assert 'target_positions' in obs
        
        # Check shapes
        assert obs['skyrmion_positions'].shape == (3, 2)  # 3 skyrmions, (x,y) positions
        assert obs['target_positions'].shape == (3, 2)
        
        # Check position bounds
        positions = obs['skyrmion_positions']
        assert np.all(positions[:, 0] >= 0)  # x >= 0
        assert np.all(positions[:, 0] <= skyrmion_env.track_length)  # x <= track_length
        assert np.all(positions[:, 1] >= 0)  # y >= 0
        assert np.all(positions[:, 1] <= skyrmion_env.track_width)  # y <= track_width
    
    def test_skyrmion_action_space(self, skyrmion_env):
        """Test skyrmion action space."""
        # Test sampling
        action = skyrmion_env.action_space.sample()
        assert action in skyrmion_env.action_space
        
        # Action should be current density and duration
        assert len(action) == 2
        assert isinstance(action[0], (int, float))  # current density
        assert isinstance(action[1], (int, float))  # duration
    
    def test_skyrmion_movement(self, skyrmion_env):
        """Test skyrmion movement under current."""
        obs, info = skyrmion_env.reset()
        initial_positions = obs['skyrmion_positions'].copy()
        
        # Apply current
        action = np.array([1e6, 1e-9])  # 1 MA/m², 1 ns
        obs_after, reward, terminated, truncated, info = skyrmion_env.step(action)
        final_positions = obs_after['skyrmion_positions']
        
        # Skyrmions should have moved
        movement = np.linalg.norm(final_positions - initial_positions, axis=1)
        assert np.any(movement > 1e-12)  # At least some movement
    
    def test_boundary_conditions(self, skyrmion_env):
        """Test skyrmion boundary conditions."""
        obs, info = skyrmion_env.reset()
        
        # Apply strong current to try to move skyrmions out of bounds
        for _ in range(10):
            action = np.array([1e7, 1e-9])  # Strong current
            obs, reward, terminated, truncated, info = skyrmion_env.step(action)
            
            positions = obs['skyrmion_positions']
            
            # Check bounds
            assert np.all(positions[:, 0] >= 0)
            assert np.all(positions[:, 0] <= skyrmion_env.track_length)
            assert np.all(positions[:, 1] >= 0)
            assert np.all(positions[:, 1] <= skyrmion_env.track_width)
            
            if terminated or truncated:
                break


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-critical integration scenarios."""
    
    def test_environment_step_performance(self):
        """Test environment step performance."""
        import time
        
        config = {
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 1000
        }
        
        env = SpinTorqueEnv(**config)
        env.reset()
        
        # Time 100 steps
        start_time = time.time()
        for _ in range(100):
            action = np.array([5e6, 1e-9])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        
        elapsed_time = time.time() - start_time
        steps_per_second = 100 / elapsed_time
        
        # Should achieve reasonable performance (>10 steps/second)
        assert steps_per_second > 10, f"Performance too slow: {steps_per_second:.1f} steps/second"
    
    def test_array_environment_performance(self):
        """Test array environment performance."""
        import time
        
        config = {
            'array_size': (4, 4),
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 100
        }
        
        env = SpinTorqueArrayEnv(**config)
        env.reset()
        
        # Time 50 steps
        start_time = time.time()
        for _ in range(50):
            action = {'type': 'row', 'index': 0, 'current': 5e6, 'duration': 1e-9}
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        
        elapsed_time = time.time() - start_time
        steps_per_second = 50 / elapsed_time
        
        # Array environment will be slower due to multiple devices
        assert steps_per_second > 1, f"Array performance too slow: {steps_per_second:.1f} steps/second"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""
    
    def test_invalid_device_type(self):
        """Test handling of invalid device type."""
        config = {
            'device_type': 'invalid_device',
            'device_params': {},
            'max_steps': 50
        }
        
        with pytest.raises(ValueError, match="Unknown device type"):
            SpinTorqueEnv(**config)
    
    def test_missing_device_parameters(self):
        """Test handling of missing device parameters."""
        config = {
            'device_type': 'stt_mram',
            'device_params': {
                # Missing required parameters
                'volume': 1e-24
            },
            'max_steps': 50
        }
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            SpinTorqueEnv(**config)
    
    def test_invalid_action_bounds(self):
        """Test handling of out-of-bounds actions."""
        config = {
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 50
        }
        
        env = SpinTorqueEnv(**config)
        env.reset()
        
        # Try extreme actions - should be handled gracefully
        extreme_actions = [
            np.array([1e20, 1e-9]),   # Very high current
            np.array([1e6, -1e-9]),   # Negative duration
            np.array([0, 1e-3]),      # Very long duration
        ]
        
        for action in extreme_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            # Should not crash, reward should be finite
            assert np.isfinite(reward)
    
    def test_array_invalid_actions(self):
        """Test array environment with invalid actions."""
        config = {
            'array_size': (2, 2),
            'device_type': 'stt_mram',
            'device_params': {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7,
                'easy_axis': np.array([0, 0, 1])
            },
            'max_steps': 50
        }
        
        env = SpinTorqueArrayEnv(**config)
        env.reset()
        
        # Invalid actions
        invalid_actions = [
            {'type': 'row', 'index': 5, 'current': 1e6, 'duration': 1e-9},  # Out of bounds index
            {'type': 'column', 'index': -1, 'current': 1e6, 'duration': 1e-9},  # Negative index
            {'type': 'individual', 'row': 3, 'col': 0, 'current': 1e6, 'duration': 1e-9},  # Out of bounds
            {'type': 'invalid', 'index': 0, 'current': 1e6, 'duration': 1e-9},  # Invalid type
        ]
        
        for action in invalid_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            # Should handle gracefully without crashing
            assert np.isfinite(reward)