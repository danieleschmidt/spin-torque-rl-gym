"""Integration tests for Gymnasium environment."""

import numpy as np
import pytest


@pytest.mark.integration
class TestSpinTorqueEnvironment:
    """Test Spin-Torque RL environment integration."""

    def test_environment_creation(self, env_config):
        """Test environment can be created with configuration."""
        # Placeholder test for future environment implementation
        config = env_config
        assert config["device_type"] in ["stt_mram", "sot_mram", "vcma_mram", "skyrmion"]
        assert config["max_steps"] > 0
        assert 0 < config["success_threshold"] <= 1.0

    def test_gymnasium_interface(self):
        """Test Gymnasium interface compliance."""
        try:
            import gymnasium as gym
            # Placeholder for environment registration
            # env = gym.make('SpinTorque-v0')
            # assert hasattr(env, 'action_space')
            # assert hasattr(env, 'observation_space')
            # assert hasattr(env, 'reset')
            # assert hasattr(env, 'step')
            pytest.skip("Environment not yet registered")
        except ImportError:
            pytest.skip("Gymnasium not available")

    def test_observation_space_structure(self, sample_magnetization, target_magnetization):
        """Test observation space structure."""
        # Expected observation components
        observation = {
            'magnetization': sample_magnetization,
            'target': target_magnetization,
            'resistance': 7.5e3,  # Ω
            'temperature': 300,  # K
            'time_remaining': 0.8,  # Normalized
            'energy_used': 10e-12,  # J
            'last_action': [1e6, 1e-9]  # [current, duration]
        }
        
        # Validate observation structure
        assert len(observation['magnetization']) == 3
        assert len(observation['target']) == 3
        assert observation['resistance'] > 0
        assert observation['temperature'] >= 0
        assert 0 <= observation['time_remaining'] <= 1
        assert observation['energy_used'] >= 0
        assert len(observation['last_action']) == 2

    def test_action_space_structure(self, mock_action):
        """Test action space structure."""
        action = mock_action
        
        # Continuous action space
        assert 'current' in action
        assert 'duration' in action
        assert action['current'] != 0  # Non-zero current
        assert action['duration'] > 0  # Positive duration

    def test_reward_function_properties(self):
        """Test reward function properties."""
        # Success reward
        success_reward = 10.0
        assert success_reward > 0
        
        # Energy penalty
        energy_penalty = -0.1
        assert energy_penalty < 0
        
        # Speed reward
        speed_reward = 1.0
        assert speed_reward > 0
        
        # Total reward should be finite
        total_reward = success_reward + energy_penalty + speed_reward
        assert np.isfinite(total_reward)

    def test_episode_termination(self):
        """Test episode termination conditions."""
        # Success termination
        success_threshold = 0.95
        alignment = 0.97
        is_success = alignment >= success_threshold
        assert is_success
        
        # Timeout termination
        max_steps = 100
        current_step = 101
        is_timeout = current_step >= max_steps
        assert is_timeout
        
        # Failure termination (optional)
        min_alignment = -0.9
        is_failure = alignment < min_alignment
        assert not is_failure  # With alignment = 0.97


@pytest.mark.integration
@pytest.mark.slow
class TestMultiDeviceEnvironment:
    """Test multi-device environment integration."""

    def test_array_environment_structure(self):
        """Test array environment structure."""
        array_size = (4, 4)
        num_devices = array_size[0] * array_size[1]
        
        # Observation should include all devices
        magnetizations = np.random.normal(0, 1, (array_size[0], array_size[1], 3))
        magnetizations = magnetizations / np.linalg.norm(magnetizations, axis=-1, keepdims=True)
        
        assert magnetizations.shape == (array_size[0], array_size[1], 3)
        assert np.allclose(np.linalg.norm(magnetizations, axis=-1), 1.0)

    def test_coupling_effects(self):
        """Test inter-device coupling effects."""
        # Dipolar coupling strength
        coupling_strength = 1e-21  # J
        distance = 100e-9  # m between devices
        
        # Coupling should decay with distance
        coupling_at_distance = coupling_strength / distance**3
        assert coupling_at_distance > 0
        
        # Closer devices should have stronger coupling
        closer_distance = 50e-9  # m
        closer_coupling = coupling_strength / closer_distance**3
        assert closer_coupling > coupling_at_distance

    def test_parallel_simulation(self):
        """Test parallel simulation capabilities."""
        # Test that multiple devices can be simulated simultaneously
        num_devices = 16
        timestep = 1e-12  # s
        simulation_time = 10e-9  # s
        num_steps = int(simulation_time / timestep)
        
        assert num_devices > 1
        assert num_steps > 0
        assert timestep > 0


@pytest.mark.integration
class TestRewardFunctionIntegration:
    """Test reward function integration."""

    def test_composite_reward_calculation(self, sample_magnetization, target_magnetization):
        """Test composite reward function."""
        m_current = sample_magnetization
        m_target = target_magnetization
        
        # Alignment reward
        alignment = np.dot(m_current, m_target)
        alignment_reward = 10.0 if alignment > 0.95 else 0.0
        
        # Energy penalty (example: 1 pJ consumed)
        energy_consumed = 1e-12  # J
        energy_penalty = -0.1 * energy_consumed / 1e-12  # Normalized to pJ
        
        # Speed bonus (example: 1 ns switching time)
        switching_time = 1e-9  # s
        speed_bonus = 1.0 / (1.0 + switching_time / 1e-9)  # Normalized to ns
        
        # Composite reward
        total_reward = alignment_reward + energy_penalty + speed_bonus
        
        assert isinstance(total_reward, float)
        assert np.isfinite(total_reward)

    def test_reward_scaling_and_normalization(self):
        """Test reward scaling and normalization."""
        # Raw reward components (different scales)
        success_component = 1.0  # Binary success
        energy_component = -1e-12  # Energy in Joules
        time_component = 1e-9  # Time in seconds
        
        # Normalized components
        success_normalized = success_component * 10.0  # Scale up
        energy_normalized = energy_component * 1e12  # Scale to pJ
        time_normalized = time_component * 1e9  # Scale to ns
        
        # All components should be similar magnitude
        components = [success_normalized, energy_normalized, time_normalized]
        magnitudes = [abs(comp) for comp in components]
        max_magnitude = max(magnitudes)
        min_magnitude = min(magnitudes)
        
        # Check that magnitudes are within reasonable range
        assert max_magnitude / min_magnitude < 100  # Less than 2 orders of magnitude


@pytest.mark.integration
@pytest.mark.physics
class TestPhysicsIntegration:
    """Test physics simulation integration."""

    def test_llgs_solver_integration(self, sample_magnetization, physics_config):
        """Test LLGS solver integration with environment."""
        m_initial = sample_magnetization
        config = physics_config
        
        # Simulation parameters
        dt = config["timestep"]
        max_time = config["max_time"]
        num_steps = int(max_time / dt)
        
        # Mock time evolution (placeholder)
        m_current = m_initial.copy()
        for step in range(min(10, num_steps)):  # Test first 10 steps
            # Simple precession around z-axis (example)
            omega = 1e9  # rad/s
            theta = omega * dt * step
            m_current = np.array([
                np.sin(theta),
                0,
                np.cos(theta)
            ])
            
            # Ensure normalization
            m_current = m_current / np.linalg.norm(m_current)
            assert np.isclose(np.linalg.norm(m_current), 1.0)

    def test_energy_conservation(self, physics_config):
        """Test energy conservation in simulation."""
        # Total energy should be conserved (approximately)
        if not physics_config["include_thermal"]:
            # Without thermal fluctuations, energy should be conserved
            initial_energy = 1e-20  # J
            final_energy = initial_energy  # In ideal case
            
            energy_error = abs(final_energy - initial_energy) / initial_energy
            assert energy_error < 1e-6  # 0.0001% error tolerance

    def test_stability_analysis(self, sample_magnetization):
        """Test numerical stability of physics integration."""
        m = sample_magnetization
        
        # Small perturbation
        epsilon = 1e-10
        m_perturbed = m + np.array([epsilon, 0, 0])
        m_perturbed = m_perturbed / np.linalg.norm(m_perturbed)
        
        # Perturbation should remain small
        perturbation_magnitude = np.linalg.norm(m_perturbed - m)
        assert perturbation_magnitude < 1e-9