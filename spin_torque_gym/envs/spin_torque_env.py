"""Main Gymnasium environment for spintronic device control.

This module implements the core RL environment where agents learn to control
magnetization switching in spintronic devices through current pulses.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..devices import DeviceFactory
from ..physics import MaterialDatabase, ThermalFluctuations
from ..physics.simple_solver import SimpleLLGSSolver
from ..rewards import CompositeReward
from ..utils import (
    EnvironmentMonitor,
    PerformanceProfiler,
    SafetyWrapper,
    get_optimizer,
)


class SpinTorqueEnv(gym.Env):
    """Gymnasium environment for spintronic device control via RL.
    
    The agent learns to apply current pulses to switch magnetization
    from initial to target states while minimizing energy consumption
    and maximizing switching reliability.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        device_type: str = 'stt_mram',
        device_params: Optional[Dict[str, Any]] = None,
        target_states: Optional[List[np.ndarray]] = None,
        max_steps: int = 100,
        max_current: float = 2e6,  # A/m²
        max_duration: float = 5e-9,  # 5 ns
        temperature: float = 300.0,  # K
        include_thermal_fluctuations: bool = True,
        reward_components: Optional[Dict[str, Dict]] = None,
        action_mode: str = 'continuous',
        observation_mode: str = 'vector',
        success_threshold: float = 0.9,
        energy_penalty_weight: float = 0.1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize SpinTorque environment.
        
        Args:
            device_type: Type of device ('stt_mram', 'sot_mram', 'vcma_mram')
            device_params: Custom device parameters
            target_states: List of possible target magnetization states
            max_steps: Maximum steps per episode
            max_current: Maximum current density magnitude (A/m²)
            max_duration: Maximum pulse duration (s)
            temperature: Operating temperature (K)
            include_thermal_fluctuations: Whether to include thermal noise
            reward_components: Custom reward function components
            action_mode: 'continuous' or 'discrete'
            observation_mode: 'vector' or 'dict'
            success_threshold: Threshold for successful switching (dot product)
            energy_penalty_weight: Weight for energy penalty in reward
            render_mode: Rendering mode ('human', 'rgb_array', None)
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Environment configuration
        self.device_type = device_type
        self.max_steps = max_steps
        self.max_current = max_current
        self.max_duration = max_duration
        self.temperature = temperature
        self.include_thermal = include_thermal_fluctuations
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.success_threshold = success_threshold
        self.energy_penalty_weight = energy_penalty_weight
        self.render_mode = render_mode

        # Initialize random number generator
        self._np_random = None
        self.seed(seed)

        # Initialize physics components
        self.solver = SimpleLLGSSolver(method='euler', rtol=1e-3, atol=1e-6, timeout=0.1)
        self.thermal_model = ThermalFluctuations(
            temperature=temperature,
            correlation_time=1e-12,
            seed=seed
        )
        self.material_db = MaterialDatabase()

        # Initialize device
        device_factory = DeviceFactory()
        if device_params is None:
            device_params = self._get_default_device_params()
        self.device = device_factory.create_device(device_type, device_params)

        # Set target states
        if target_states is None:
            self.target_states = [np.array([0, 0, 1]), np.array([0, 0, -1])]  # ±z
        else:
            self.target_states = [self.device.validate_magnetization(t) for t in target_states]

        # Initialize reward function
        if reward_components is None:
            reward_components = self._get_default_reward_components()
        self.reward_function = CompositeReward(reward_components)

        # Define action space
        self._setup_action_space()

        # Define observation space
        self._setup_observation_space()

        # Environment state
        self.current_magnetization = None
        self.target_magnetization = None
        self.step_count = 0
        self.total_energy = 0.0
        self.episode_history = []
        self.last_action = np.zeros(2)  # [current, duration]

        # Monitoring and safety
        self.monitor = EnvironmentMonitor(log_level='WARNING')
        self.safety = SafetyWrapper(self.monitor)

        # Performance optimization
        self.optimizer = get_optimizer()
        self.profiler = PerformanceProfiler()
        self.enable_caching = True
        self.cache_observations = True

        # Rendering
        self.renderer = None
        if render_mode == 'human':
            self._init_renderer()

    def _get_default_device_params(self) -> Dict[str, Any]:
        """Get default device parameters based on device type."""
        if self.device_type == 'stt_mram':
            return {
                'volume': 50e-9 * 100e-9 * 2e-9,  # 50×100×2 nm³
                'area': 50e-9 * 100e-9,
                'thickness': 2e-9,
                'aspect_ratio': 2.0,
                'saturation_magnetization': 800e3,  # A/m
                'damping': 0.01,
                'uniaxial_anisotropy': 1.2e6,  # J/m³
                'exchange_constant': 20e-12,  # J/m
                'polarization': 0.7,
                'resistance_parallel': 1e3,  # Ω
                'resistance_antiparallel': 2e3,  # Ω
                'easy_axis': np.array([0, 0, 1]),
                'reference_magnetization': np.array([0, 0, 1])
            }
        else:
            # Default parameters for other device types
            return {
                'volume': 1e-24,
                'saturation_magnetization': 800e3,
                'damping': 0.01,
                'uniaxial_anisotropy': 1e6,
                'polarization': 0.7
            }

    def _get_default_reward_components(self) -> Dict[str, Dict]:
        """Get default reward function components."""
        return {
            'success': {
                'weight': 10.0,
                'function': lambda obs, action, next_obs, info:
                    10.0 if info.get('is_success', False) else 0.0
            },
            'energy': {
                'weight': -self.energy_penalty_weight,
                'function': lambda obs, action, next_obs, info:
                    -info.get('step_energy', 0.0) / 1e-12  # Normalize to pJ
            },
            'progress': {
                'weight': 1.0,
                'function': lambda obs, action, next_obs, info:
                    info.get('alignment_improvement', 0.0)
            },
            'stability': {
                'weight': -2.0,
                'function': lambda obs, action, next_obs, info:
                    -max(0, np.linalg.norm(next_obs['magnetization']) - 1.1) if isinstance(next_obs, dict) else 0.0
            }
        }

    def _setup_action_space(self):
        """Setup action space based on action mode."""
        if self.action_mode == 'continuous':
            # Continuous: [current_density, pulse_duration]
            self.action_space = spaces.Box(
                low=np.array([-self.max_current, 0.0]),
                high=np.array([self.max_current, self.max_duration]),
                dtype=np.float32
            )
        elif self.action_mode == 'discrete':
            # Discrete: 5 current levels × 4 duration levels = 20 actions
            self.current_levels = np.linspace(-self.max_current, self.max_current, 5)
            self.duration_levels = np.array([0.1e-9, 0.5e-9, 1.0e-9, 2.0e-9])
            self.action_space = spaces.Discrete(len(self.current_levels) * len(self.duration_levels))
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")

    def _setup_observation_space(self):
        """Setup observation space based on observation mode."""
        if self.observation_mode == 'vector':
            # Vector observation: [mx, my, mz, tx, ty, tz, R/R0, T/T0, t_rem, E_norm, I_prev, dt_prev]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float32
            )
        elif self.observation_mode == 'dict':
            # Dictionary observation
            self.observation_space = spaces.Dict({
                'magnetization': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                'target': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                'resistance': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'temperature': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'steps_remaining': spaces.Box(0, self.max_steps, shape=(1,), dtype=int),
                'energy_consumed': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'last_action': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[np.ndarray, Dict], Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if options is None:
            options = {}

        # End previous episode if exists
        if hasattr(self, 'episode_history') and len(self.episode_history) > 0:
            final_alignment = np.dot(self.current_magnetization, self.target_magnetization) if self.current_magnetization is not None else 0.0
            total_reward = sum(h['reward'] for h in self.episode_history)
            success = final_alignment >= self.success_threshold
            self.monitor.end_episode(total_reward, success)

        # Start new episode monitoring
        self.monitor.start_episode()

        # Reset environment state
        self.step_count = 0
        self.total_energy = 0.0
        self.episode_history = []
        self.last_action = np.zeros(2)

        # Set initial magnetization
        if 'initial_state' in options:
            self.current_magnetization = self.device.validate_magnetization(options['initial_state'])
        else:
            # Random initial state
            initial_state = self._np_random.normal(0, 1, 3)
            self.current_magnetization = self.device.validate_magnetization(initial_state)

        # Set target magnetization
        if 'target_state' in options:
            self.target_magnetization = self.device.validate_magnetization(options['target_state'])
        else:
            # Random target from available states
            idx = self._np_random.integers(len(self.target_states))
            self.target_magnetization = self.target_states[idx].copy()

        # Update thermal model temperature if specified
        if 'temperature' in options:
            self.thermal_model.set_temperature(options['temperature'])

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: Union[np.ndarray, int]
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.current_magnetization is None:
            raise RuntimeError("Environment must be reset before calling step")

        # Start step monitoring
        self.monitor.start_step()

        try:
            # Validate action for safety
            action_array = np.array(action, dtype=np.float32) if not isinstance(action, np.ndarray) else action
            safe_action = self.safety.validate_action(action_array)

            # Parse action
            current_density, pulse_duration = self._parse_action(safe_action)
            self.last_action = np.array([current_density, pulse_duration])

            # Store previous state for reward calculation
            prev_magnetization = self.current_magnetization.copy()
            prev_alignment = np.dot(prev_magnetization, self.target_magnetization)

            # Simulate magnetization dynamics
            step_info = self._simulate_dynamics(current_density, pulse_duration)

            # Update state
            self.current_magnetization = step_info['final_magnetization']
            self.total_energy += step_info['energy_consumed']
            self.step_count += 1

            # Calculate reward
            current_alignment = np.dot(self.current_magnetization, self.target_magnetization)
            alignment_improvement = current_alignment - prev_alignment

            is_success = current_alignment >= self.success_threshold

            reward_info = {
                'is_success': is_success,
                'step_energy': step_info['energy_consumed'],
                'alignment_improvement': alignment_improvement,
                'current_alignment': current_alignment
            }

            # Optimized observation computation
            with self.profiler.time_operation("get_observation"):
                observation = self._get_observation()

            # Optimized reward computation
            with self.profiler.time_operation("compute_reward"):
                reward = self.reward_function.compute(None, action, observation, reward_info)

            # Check termination conditions
            terminated = is_success
            truncated = self.step_count >= self.max_steps

            # Store step in history
            self.episode_history.append({
                'step': self.step_count,
                'action': [current_density, pulse_duration],
                'magnetization': self.current_magnetization.copy(),
                'reward': reward,
                'energy': step_info['energy_consumed'],
                'alignment': current_alignment
            })

            info = self._get_info()
            info.update(step_info)
            info.update(reward_info)

            # Validate outputs for safety
            observation = self.safety.validate_observation(observation)
            reward = self.safety.validate_reward(reward)

            # End step monitoring
            self.monitor.end_step(reward, info)

            return observation, reward, terminated, truncated, info

        except Exception as e:
            self.monitor.log_error(e, "step_execution")
            # Return safe fallback values
            observation = self._get_observation()
            reward = -1.0  # Penalty for error
            terminated = False
            truncated = True  # Force episode end on error
            info = {'error': str(e), 'step_count': self.step_count}

            self.monitor.end_step(reward, info)
            return observation, reward, terminated, truncated, info

    def _parse_action(self, action: Union[np.ndarray, int]) -> Tuple[float, float]:
        """Parse action into current density and pulse duration."""
        if self.action_mode == 'continuous':
            if isinstance(action, (int, float)):
                # Single value - interpret as current with default duration
                current_density = float(action)
                pulse_duration = 1e-9  # Default 1 ns
            else:
                current_density = float(action[0])
                pulse_duration = float(action[1]) if len(action) > 1 else 1e-9
        elif self.action_mode == 'discrete':
            action_idx = int(action)
            current_idx = action_idx // len(self.duration_levels)
            duration_idx = action_idx % len(self.duration_levels)

            current_density = self.current_levels[current_idx]
            pulse_duration = self.duration_levels[duration_idx]
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")

        # Clip to valid ranges
        current_density = np.clip(current_density, -self.max_current, self.max_current)
        pulse_duration = np.clip(pulse_duration, 1e-12, self.max_duration)

        return current_density, pulse_duration

    def _simulate_dynamics(
        self,
        current_density: float,
        pulse_duration: float
    ) -> Dict[str, Any]:
        """Simulate magnetization dynamics for given current pulse."""
        # Define current function
        def current_func(t: float) -> float:
            return current_density if t <= pulse_duration else 0.0

        # Define field function (no external field by default)
        def field_func(t: float) -> np.ndarray:
            return np.zeros(3)

        # Simulate dynamics
        try:
            result = self.solver.solve(
                m_initial=self.current_magnetization,
                time_span=(0, pulse_duration),
                device_params=self.device.device_params,
                current_func=current_func,
                field_func=field_func,
                thermal_noise=self.include_thermal,
                temperature=self.temperature
            )

            if result['success']:
                final_magnetization = result['m'][-1]
                # Ensure normalization
                final_magnetization = final_magnetization / np.linalg.norm(final_magnetization)
            else:
                warnings.warn("Dynamics simulation failed, using initial state")
                final_magnetization = self.current_magnetization

        except Exception as e:
            warnings.warn(f"Error in dynamics simulation: {e}")
            final_magnetization = self.current_magnetization

        # Calculate energy consumed
        if abs(current_density) > 1e-12:
            resistance = self.device.compute_resistance(self.current_magnetization)
            area = self.device.get_parameter('area', 1e-14)
            voltage = current_density * resistance * area
            energy_consumed = voltage**2 / resistance * pulse_duration
        else:
            energy_consumed = 0.0

        return {
            'final_magnetization': final_magnetization,
            'energy_consumed': energy_consumed,
            'pulse_duration': pulse_duration,
            'current_density': current_density,
            'simulation_success': result.get('success', False) if 'result' in locals() else False
        }

    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """Get current observation."""
        if self.observation_mode == 'vector':
            # Cache key for observation computation
            if self.cache_observations:
                cache_key = f"obs_{hash(tuple(self.current_magnetization.round(6)))}"
                cached_obs = self.optimizer.cache.get(cache_key)
                if cached_obs is not None:
                    return cached_obs

            # Compute normalized values
            resistance = self.device.compute_resistance(self.current_magnetization)
            r0 = self.device.get_parameter('resistance_parallel', 1e3)

            temp_norm = self.temperature / 300.0
            steps_remaining_norm = (self.max_steps - self.step_count) / self.max_steps
            energy_norm = self.total_energy / 1e-12  # Normalize to pJ

            current_norm = self.last_action[0] / self.max_current if len(self.last_action) > 0 else 0.0
            duration_norm = self.last_action[1] / self.max_duration if len(self.last_action) > 1 else 0.0

            obs = np.array([
                *self.current_magnetization,  # mx, my, mz
                *self.target_magnetization,   # tx, ty, tz
                resistance / r0,              # Normalized resistance
                temp_norm,                    # Normalized temperature
                steps_remaining_norm,         # Normalized steps remaining
                energy_norm,                  # Normalized energy
                current_norm,                 # Previous current (normalized)
                duration_norm                 # Previous duration (normalized)
            ], dtype=np.float32)

            # Cache the observation
            if self.cache_observations:
                self.optimizer.cache.put(cache_key, obs)

        elif self.observation_mode == 'dict':
            resistance = self.device.compute_resistance(self.current_magnetization)

            obs = {
                'magnetization': self.current_magnetization.astype(np.float32),
                'target': self.target_magnetization.astype(np.float32),
                'resistance': np.array([resistance], dtype=np.float32),
                'temperature': np.array([self.temperature], dtype=np.float32),
                'steps_remaining': np.array([self.max_steps - self.step_count], dtype=int),
                'energy_consumed': np.array([self.total_energy], dtype=np.float32),
                'last_action': self.last_action.astype(np.float32)
            }

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        current_alignment = np.dot(self.current_magnetization, self.target_magnetization) if self.current_magnetization is not None else 0.0

        return {
            'step_count': self.step_count,
            'total_energy': self.total_energy,
            'current_alignment': current_alignment,
            'is_success': current_alignment >= self.success_threshold,
            'target_reached': current_alignment >= self.success_threshold,
            'magnetization_magnitude': np.linalg.norm(self.current_magnetization) if self.current_magnetization is not None else 0.0,
            'device_type': self.device_type,
            'episode_history': self.episode_history.copy()
        }

    def render(self, mode: Optional[str] = None):
        """Render the environment."""
        if mode is None:
            mode = self.render_mode

        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        elif mode is None:
            return
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _init_renderer(self):
        """Initialize renderer for human mode."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            self.fig = plt.figure(figsize=(12, 8))
            self.ax_3d = self.fig.add_subplot(221, projection='3d')
            self.ax_energy = self.fig.add_subplot(222)
            self.ax_alignment = self.fig.add_subplot(223)
            self.ax_action = self.fig.add_subplot(224)

            plt.ion()  # Interactive mode
            self.renderer = True

        except ImportError:
            warnings.warn("Matplotlib not available, rendering disabled")
            self.renderer = None

    def _render_human(self):
        """Render for human viewing."""
        if self.renderer is None:
            return

        try:
            import matplotlib.pyplot as plt

            # Clear axes
            self.ax_3d.clear()
            self.ax_energy.clear()
            self.ax_alignment.clear()
            self.ax_action.clear()

            # 3D magnetization visualization
            self.ax_3d.quiver(0, 0, 0, *self.current_magnetization, color='red', label='Current', arrow_length_ratio=0.1)
            self.ax_3d.quiver(0, 0, 0, *self.target_magnetization, color='blue', label='Target', arrow_length_ratio=0.1)

            # Draw unit sphere
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

            self.ax_3d.set_xlim([-1.5, 1.5])
            self.ax_3d.set_ylim([-1.5, 1.5])
            self.ax_3d.set_zlim([-1.5, 1.5])
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('Z')
            self.ax_3d.legend()
            self.ax_3d.set_title('Magnetization State')

            # Plot energy and alignment history if available
            if self.episode_history:
                steps = [h['step'] for h in self.episode_history]
                energies = [h['energy'] for h in self.episode_history]
                alignments = [h['alignment'] for h in self.episode_history]
                currents = [h['action'][0] for h in self.episode_history]

                self.ax_energy.plot(steps, energies, 'g-')
                self.ax_energy.set_xlabel('Step')
                self.ax_energy.set_ylabel('Energy (J)')
                self.ax_energy.set_title('Energy Consumption')

                self.ax_alignment.plot(steps, alignments, 'b-')
                self.ax_alignment.axhline(y=self.success_threshold, color='r', linestyle='--', label='Success threshold')
                self.ax_alignment.set_xlabel('Step')
                self.ax_alignment.set_ylabel('Alignment')
                self.ax_alignment.set_title('Target Alignment')
                self.ax_alignment.legend()

                self.ax_action.plot(steps, currents, 'orange')
                self.ax_action.set_xlabel('Step')
                self.ax_action.set_ylabel('Current (A/m²)')
                self.ax_action.set_title('Applied Current')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        except Exception as e:
            warnings.warn(f"Rendering error: {e}")

    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array."""
        # Simplified rendering - return a placeholder image
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot magnetization state
        ax.quiver(0, 0, self.current_magnetization[0], self.current_magnetization[1],
                 color='red', scale=1, label='Current')
        ax.quiver(0, 0, self.target_magnetization[0], self.target_magnetization[1],
                 color='blue', scale=1, label='Target')

        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Step {self.step_count}: Alignment = {np.dot(self.current_magnetization, self.target_magnetization):.3f}')

        # Convert to RGB array
        fig.canvas.draw()
        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return rgb_array

    def close(self):
        """Close environment and cleanup."""
        if hasattr(self, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(self.fig)

        self.renderer = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed."""
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return self.device.get_device_info()

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive environment health report."""
        return self.monitor.get_health_report()

    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver performance information."""
        return self.solver.get_solver_info()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'profiler': self.profiler.get_stats(),
            'optimizer': self.optimizer.get_performance_stats(),
            'solver': self.get_solver_info(),
            'health': self.get_health_report()
        }

    def analyze_episode(self) -> Dict[str, Any]:
        """Analyze completed episode."""
        if not self.episode_history:
            return {}

        total_energy = sum(h['energy'] for h in self.episode_history)
        final_alignment = self.episode_history[-1]['alignment']
        success = final_alignment >= self.success_threshold

        # Calculate switching time (first time alignment exceeds threshold)
        switching_step = None
        for i, h in enumerate(self.episode_history):
            if h['alignment'] >= self.success_threshold:
                switching_step = i + 1
                break

        return {
            'episode_length': len(self.episode_history),
            'total_energy': total_energy,
            'final_alignment': final_alignment,
            'success': success,
            'switching_step': switching_step,
            'average_reward': np.mean([h['reward'] for h in self.episode_history]),
            'energy_efficiency': final_alignment / total_energy if total_energy > 0 else 0,
            'history': self.episode_history.copy()
        }
