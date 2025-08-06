"""Multi-device array environment for spintronic crossbar arrays.

This module implements RL environments for controlling arrays of spintronic
devices, enabling training of agents for crossbar array manipulation and
pattern programming applications.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings

from .spin_torque_env import SpinTorqueEnv
from ..physics import LLGSSolver, ThermalFluctuations
from ..devices import DeviceFactory
from ..rewards import CompositeReward


class SpinTorqueArrayEnv(gym.Env):
    """Gymnasium environment for controlling arrays of spintronic devices.
    
    This environment enables training RL agents to control crossbar arrays
    of spintronic devices for applications like neuromorphic computing,
    in-memory computing, and pattern storage.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(
        self,
        array_size: Tuple[int, int] = (4, 4),
        device_type: str = 'stt_mram',
        device_params: Optional[Dict[str, Any]] = None,
        target_pattern: Optional[np.ndarray] = None,
        max_steps: int = 200,
        max_current: float = 2e6,  # A/m²
        max_duration: float = 5e-9,  # 5 ns
        temperature: float = 300.0,  # K
        include_thermal_fluctuations: bool = True,
        include_coupling: bool = True,
        coupling_strength: float = 0.1,
        coupling_type: str = 'dipolar',
        reward_components: Optional[Dict[str, Dict]] = None,
        action_mode: str = 'individual',  # 'individual', 'row', 'column', 'global'
        observation_mode: str = 'array',
        success_threshold: float = 0.9,
        energy_penalty_weight: float = 0.1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize SpinTorqueArray environment.
        
        Args:
            array_size: (rows, cols) size of device array
            device_type: Type of devices in array
            device_params: Parameters for individual devices
            target_pattern: Target magnetization pattern
            max_steps: Maximum steps per episode
            max_current: Maximum current density magnitude (A/m²)
            max_duration: Maximum pulse duration (s)
            temperature: Operating temperature (K)
            include_thermal_fluctuations: Whether to include thermal noise
            include_coupling: Whether to include inter-device coupling
            coupling_strength: Strength of coupling between devices
            coupling_type: Type of coupling ('dipolar', 'exchange', 'stray_field')
            reward_components: Custom reward function components
            action_mode: How actions are applied ('individual', 'row', 'column', 'global')
            observation_mode: Observation format ('array', 'vector', 'dict')
            success_threshold: Threshold for successful pattern programming
            energy_penalty_weight: Weight for energy penalty in reward
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__()
        
        # Array configuration
        self.array_size = array_size
        self.n_rows, self.n_cols = array_size
        self.n_devices = self.n_rows * self.n_cols
        self.device_type = device_type
        
        # Environment parameters
        self.max_steps = max_steps
        self.max_current = max_current
        self.max_duration = max_duration
        self.temperature = temperature
        self.include_thermal = include_thermal_fluctuations
        self.include_coupling = include_coupling
        self.coupling_strength = coupling_strength
        self.coupling_type = coupling_type
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.success_threshold = success_threshold
        self.energy_penalty_weight = energy_penalty_weight
        self.render_mode = render_mode
        
        # Initialize random number generator
        self._np_random = None
        self.seed(seed)
        
        # Initialize physics components
        self.solver = LLGSSolver(method='RK45', rtol=1e-6, atol=1e-9)
        self.thermal_model = ThermalFluctuations(
            temperature=temperature,
            correlation_time=1e-12,
            seed=seed
        )
        
        # Initialize device array
        device_factory = DeviceFactory()
        if device_params is None:
            device_params = self._get_default_device_params()
        
        self.devices = []
        for i in range(self.n_devices):
            device = device_factory.create_device(device_type, device_params)
            self.devices.append(device)
        
        # Set target pattern
        if target_pattern is None:
            # Default checkerboard pattern
            self.target_pattern = self._generate_checkerboard_pattern()
        else:
            if target_pattern.shape != (self.n_rows, self.n_cols, 3):
                raise ValueError(f"Target pattern shape must be {(self.n_rows, self.n_cols, 3)}")
            self.target_pattern = target_pattern.copy()
        
        # Initialize reward function
        if reward_components is None:
            reward_components = self._get_default_reward_components()
        self.reward_function = CompositeReward(reward_components)
        
        # Setup action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()
        
        # Environment state
        self.current_pattern = None
        self.step_count = 0
        self.total_energy = 0.0
        self.episode_history = []
        
        # Coupling matrix for inter-device interactions
        if self.include_coupling:
            self.coupling_matrix = self._compute_coupling_matrix()
        
        # Rendering
        self.renderer = None
        if render_mode == 'human':
            self._init_renderer()
    
    def _get_default_device_params(self) -> Dict[str, Any]:
        """Get default device parameters."""
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
    
    def _generate_checkerboard_pattern(self) -> np.ndarray:
        """Generate default checkerboard magnetization pattern."""
        pattern = np.zeros((self.n_rows, self.n_cols, 3))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (i + j) % 2 == 0:
                    pattern[i, j] = [0, 0, 1]  # +z
                else:
                    pattern[i, j] = [0, 0, -1]  # -z
        return pattern
    
    def _get_default_reward_components(self) -> Dict[str, Dict]:
        """Get default reward function components."""
        return {
            'pattern_match': {
                'weight': 10.0,
                'function': self._pattern_match_reward
            },
            'energy': {
                'weight': -self.energy_penalty_weight,
                'function': lambda obs, action, next_obs, info: 
                    -info.get('step_energy', 0.0) / 1e-12
            },
            'progress': {
                'weight': 1.0,
                'function': lambda obs, action, next_obs, info:
                    info.get('pattern_improvement', 0.0)
            },
            'uniformity': {
                'weight': 2.0,
                'function': self._uniformity_reward
            }
        }
    
    def _pattern_match_reward(self, obs, action, next_obs, info) -> float:
        """Reward for matching target pattern."""
        if info.get('is_success', False):
            return 10.0
        else:
            # Partial reward based on pattern similarity
            similarity = info.get('pattern_similarity', 0.0)
            return similarity * 5.0
    
    def _uniformity_reward(self, obs, action, next_obs, info) -> float:
        """Reward for uniform magnetization magnitudes."""
        if self.current_pattern is None:
            return 0.0
        
        magnitudes = np.linalg.norm(self.current_pattern, axis=2)
        uniformity = 1.0 - np.std(magnitudes)
        return max(0, uniformity)
    
    def _setup_action_space(self):
        """Setup action space based on action mode."""
        if self.action_mode == 'individual':
            # Control each device individually: [device_idx, current, duration]
            self.action_space = spaces.Box(
                low=np.array([0, -self.max_current, 0]),
                high=np.array([self.n_devices - 1, self.max_current, self.max_duration]),
                dtype=np.float32
            )
        elif self.action_mode == 'row':
            # Control entire rows: [row_idx, current, duration]
            self.action_space = spaces.Box(
                low=np.array([0, -self.max_current, 0]),
                high=np.array([self.n_rows - 1, self.max_current, self.max_duration]),
                dtype=np.float32
            )
        elif self.action_mode == 'column':
            # Control entire columns: [col_idx, current, duration]
            self.action_space = spaces.Box(
                low=np.array([0, -self.max_current, 0]),
                high=np.array([self.n_cols - 1, self.max_current, self.max_duration]),
                dtype=np.float32
            )
        elif self.action_mode == 'global':
            # Apply same current to all devices: [current, duration]
            self.action_space = spaces.Box(
                low=np.array([-self.max_current, 0]),
                high=np.array([self.max_current, self.max_duration]),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")
    
    def _setup_observation_space(self):
        """Setup observation space based on observation mode."""
        if self.observation_mode == 'array':
            # Array observation: magnetization array + target pattern
            self.observation_space = spaces.Box(
                low=-1, high=1,
                shape=(self.n_rows, self.n_cols, 6),  # [mx,my,mz,tx,ty,tz] per device
                dtype=np.float32
            )
        elif self.observation_mode == 'vector':
            # Flattened vector observation
            obs_size = self.n_devices * 6 + 4  # magnetizations + targets + global info
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
        elif self.observation_mode == 'dict':
            # Dictionary observation
            self.observation_space = spaces.Dict({
                'current_pattern': spaces.Box(
                    -1, 1, shape=(self.n_rows, self.n_cols, 3), dtype=np.float32
                ),
                'target_pattern': spaces.Box(
                    -1, 1, shape=(self.n_rows, self.n_cols, 3), dtype=np.float32
                ),
                'pattern_similarity': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                'steps_remaining': spaces.Box(0, self.max_steps, shape=(1,), dtype=int),
                'total_energy': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
    
    def _compute_coupling_matrix(self) -> np.ndarray:
        """Compute coupling matrix between devices based on coupling type."""
        coupling_matrix = np.zeros((self.n_devices, self.n_devices))
        
        for i in range(self.n_devices):
            for j in range(self.n_devices):
                if i == j:
                    continue
                
                # Get device positions
                i_row, i_col = divmod(i, self.n_cols)
                j_row, j_col = divmod(j, self.n_cols)
                
                # Calculate distance
                distance = np.sqrt((i_row - j_row)**2 + (i_col - j_col)**2)
                
                if self.coupling_type == 'dipolar':
                    # Dipolar coupling: 1/r³
                    if distance > 0:
                        coupling_matrix[i, j] = self.coupling_strength / (distance**3)
                elif self.coupling_type == 'exchange':
                    # Exchange coupling: only nearest neighbors
                    if distance == 1:
                        coupling_matrix[i, j] = self.coupling_strength
                elif self.coupling_type == 'stray_field':
                    # Stray field coupling: 1/r²
                    if distance > 0:
                        coupling_matrix[i, j] = self.coupling_strength / (distance**2)
        
        return coupling_matrix
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[np.ndarray, Dict], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if options is None:
            options = {}
        
        # Reset environment state
        self.step_count = 0
        self.total_energy = 0.0
        self.episode_history = []
        
        # Set initial pattern
        if 'initial_pattern' in options:
            self.current_pattern = options['initial_pattern'].copy()
        else:
            # Random initial pattern
            self.current_pattern = np.zeros((self.n_rows, self.n_cols, 3))
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    # Random magnetization direction
                    random_m = self._np_random.normal(0, 1, 3)
                    random_m = random_m / np.linalg.norm(random_m)
                    self.current_pattern[i, j] = random_m
        
        # Set target pattern
        if 'target_pattern' in options:
            self.target_pattern = options['target_pattern'].copy()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: Union[np.ndarray, int]
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.current_pattern is None:
            raise RuntimeError("Environment must be reset before calling step")
        
        # Store previous state for reward calculation
        prev_pattern = self.current_pattern.copy()
        prev_similarity = self._compute_pattern_similarity(prev_pattern)
        
        # Parse and apply action
        step_info = self._apply_action(action)
        
        # Update state
        self.total_energy += step_info['energy_consumed']
        self.step_count += 1
        
        # Calculate reward
        current_similarity = self._compute_pattern_similarity(self.current_pattern)
        pattern_improvement = current_similarity - prev_similarity
        
        is_success = current_similarity >= self.success_threshold
        
        reward_info = {
            'is_success': is_success,
            'step_energy': step_info['energy_consumed'],
            'pattern_improvement': pattern_improvement,
            'pattern_similarity': current_similarity
        }
        
        observation = self._get_observation()
        reward = self.reward_function.compute(None, action, observation, reward_info)
        
        # Check termination conditions
        terminated = is_success
        truncated = self.step_count >= self.max_steps
        
        # Store step in history
        self.episode_history.append({
            'step': self.step_count,
            'action': action.copy() if hasattr(action, 'copy') else action,
            'pattern': self.current_pattern.copy(),
            'reward': reward,
            'energy': step_info['energy_consumed'],
            'similarity': current_similarity
        })
        
        info = self._get_info()
        info.update(step_info)
        info.update(reward_info)
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: Union[np.ndarray, int]) -> Dict[str, Any]:
        """Apply action to the device array."""
        current_density = float(action[1]) if len(action) > 1 else 0.0
        pulse_duration = float(action[2]) if len(action) > 2 else 1e-9
        
        # Clip to valid ranges
        current_density = np.clip(current_density, -self.max_current, self.max_current)
        pulse_duration = np.clip(pulse_duration, 1e-12, self.max_duration)
        
        total_energy = 0.0
        affected_devices = []
        
        if self.action_mode == 'individual':
            # Apply to single device
            device_idx = int(np.clip(action[0], 0, self.n_devices - 1))
            affected_devices = [device_idx]
            
        elif self.action_mode == 'row':
            # Apply to entire row
            row_idx = int(np.clip(action[0], 0, self.n_rows - 1))
            affected_devices = list(range(row_idx * self.n_cols, (row_idx + 1) * self.n_cols))
            
        elif self.action_mode == 'column':
            # Apply to entire column
            col_idx = int(np.clip(action[0], 0, self.n_cols - 1))
            affected_devices = list(range(col_idx, self.n_devices, self.n_cols))
            
        elif self.action_mode == 'global':
            # Apply to all devices
            affected_devices = list(range(self.n_devices))
        
        # Apply dynamics to affected devices
        for device_idx in affected_devices:
            row_idx, col_idx = divmod(device_idx, self.n_cols)
            current_m = self.current_pattern[row_idx, col_idx]
            
            # Compute effective field including coupling
            h_eff = self._compute_effective_field(device_idx, current_m)
            
            # Simulate dynamics
            try:
                final_m = self._simulate_device_dynamics(
                    current_m, current_density, pulse_duration, h_eff
                )
                self.current_pattern[row_idx, col_idx] = final_m
                
                # Calculate energy consumed
                device = self.devices[device_idx]
                resistance = device.compute_resistance(current_m)
                area = device.get_parameter('area', 1e-14)
                if abs(current_density) > 1e-12:
                    voltage = current_density * resistance * area
                    energy = voltage**2 / resistance * pulse_duration
                    total_energy += energy
                    
            except Exception as e:
                warnings.warn(f"Error simulating device {device_idx}: {e}")
        
        return {
            'energy_consumed': total_energy,
            'affected_devices': affected_devices,
            'current_density': current_density,
            'pulse_duration': pulse_duration
        }
    
    def _compute_effective_field(self, device_idx: int, magnetization: np.ndarray) -> np.ndarray:
        """Compute effective field for a device including coupling effects."""
        device = self.devices[device_idx]
        
        # Intrinsic effective field (anisotropy, thermal, etc.)
        h_intrinsic = device.compute_effective_field(magnetization, np.zeros(3))
        
        # Coupling field from other devices
        h_coupling = np.zeros(3)
        if self.include_coupling:
            for j in range(self.n_devices):
                if j != device_idx:
                    j_row, j_col = divmod(j, self.n_cols)
                    j_magnetization = self.current_pattern[j_row, j_col]
                    coupling_strength = self.coupling_matrix[device_idx, j]
                    h_coupling += coupling_strength * j_magnetization
        
        return h_intrinsic + h_coupling
    
    def _simulate_device_dynamics(
        self,
        m_initial: np.ndarray,
        current_density: float,
        pulse_duration: float,
        h_effective: np.ndarray
    ) -> np.ndarray:
        """Simulate dynamics for a single device."""
        # Simplified dynamics simulation
        # In practice, this would use the full LLGS solver
        
        # Apply torque from current
        if abs(current_density) > 1e-12:
            # Simplified STT torque
            p_hat = np.array([0, 0, 1])  # Reference direction
            tau_stt = 0.1 * current_density * np.cross(m_initial, np.cross(m_initial, p_hat))
            
            # Simple integration
            alpha = 0.01  # Damping
            gamma = 2.21e5  # Gyromagnetic ratio
            
            dm_dt = -gamma * np.cross(m_initial, h_effective)
            dm_dt += alpha * np.cross(m_initial, dm_dt)
            dm_dt += tau_stt
            
            # Euler integration (simplified)
            dt = pulse_duration / 10  # Sub-steps
            m = m_initial.copy()
            for _ in range(10):
                m += dm_dt * dt
                m = m / np.linalg.norm(m)  # Normalize
            
            return m
        else:
            return m_initial
    
    def _compute_pattern_similarity(self, pattern: np.ndarray) -> float:
        """Compute similarity between current and target patterns."""
        similarities = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                dot_product = np.dot(pattern[i, j], self.target_pattern[i, j])
                similarities.append(dot_product)
        
        return np.mean(similarities)
    
    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """Get current observation."""
        if self.observation_mode == 'array':
            # Concatenate current and target patterns
            obs = np.concatenate([
                self.current_pattern,
                self.target_pattern
            ], axis=2)
            return obs.astype(np.float32)
            
        elif self.observation_mode == 'vector':
            # Flatten everything into a vector
            current_flat = self.current_pattern.flatten()
            target_flat = self.target_pattern.flatten()
            
            similarity = self._compute_pattern_similarity(self.current_pattern)
            steps_remaining_norm = (self.max_steps - self.step_count) / self.max_steps
            energy_norm = self.total_energy / 1e-12
            
            obs = np.concatenate([
                current_flat,
                target_flat,
                [similarity, steps_remaining_norm, energy_norm, self.temperature / 300.0]
            ])
            return obs.astype(np.float32)
            
        elif self.observation_mode == 'dict':
            similarity = self._compute_pattern_similarity(self.current_pattern)
            
            return {
                'current_pattern': self.current_pattern.astype(np.float32),
                'target_pattern': self.target_pattern.astype(np.float32),
                'pattern_similarity': np.array([similarity], dtype=np.float32),
                'steps_remaining': np.array([self.max_steps - self.step_count], dtype=int),
                'total_energy': np.array([self.total_energy], dtype=np.float32)
            }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        similarity = self._compute_pattern_similarity(self.current_pattern)
        
        return {
            'step_count': self.step_count,
            'total_energy': self.total_energy,
            'pattern_similarity': similarity,
            'is_success': similarity >= self.success_threshold,
            'array_size': self.array_size,
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
            
            self.fig, ((self.ax_current, self.ax_target), 
                      (self.ax_similarity, self.ax_energy)) = plt.subplots(2, 2, figsize=(12, 8))
            plt.ion()
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
            self.ax_current.clear()
            self.ax_target.clear()
            self.ax_similarity.clear()
            self.ax_energy.clear()
            
            # Plot current pattern (z-component)
            current_z = self.current_pattern[:, :, 2]
            im1 = self.ax_current.imshow(current_z, cmap='RdBu', vmin=-1, vmax=1)
            self.ax_current.set_title('Current Pattern (Mz)')
            self.ax_current.set_xlabel('Column')
            self.ax_current.set_ylabel('Row')
            plt.colorbar(im1, ax=self.ax_current)
            
            # Plot target pattern (z-component)
            target_z = self.target_pattern[:, :, 2]
            im2 = self.ax_target.imshow(target_z, cmap='RdBu', vmin=-1, vmax=1)
            self.ax_target.set_title('Target Pattern (Mz)')
            self.ax_target.set_xlabel('Column')
            self.ax_target.set_ylabel('Row')
            plt.colorbar(im2, ax=self.ax_target)
            
            # Plot similarity history
            if self.episode_history:
                steps = [h['step'] for h in self.episode_history]
                similarities = [h['similarity'] for h in self.episode_history]
                energies = [h['energy'] for h in self.episode_history]
                
                self.ax_similarity.plot(steps, similarities, 'b-', label='Similarity')
                self.ax_similarity.axhline(y=self.success_threshold, color='r', linestyle='--', label='Success threshold')
                self.ax_similarity.set_xlabel('Step')
                self.ax_similarity.set_ylabel('Pattern Similarity')
                self.ax_similarity.set_title('Pattern Similarity Progress')
                self.ax_similarity.legend()
                self.ax_similarity.set_ylim([0, 1])
                
                self.ax_energy.plot(steps, energies, 'g-')
                self.ax_energy.set_xlabel('Step')
                self.ax_energy.set_ylabel('Energy (J)')
                self.ax_energy.set_title('Energy Consumption per Step')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            warnings.warn(f"Rendering error: {e}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot current and target patterns
        current_z = self.current_pattern[:, :, 2]
        target_z = self.target_pattern[:, :, 2]
        
        im1 = ax1.imshow(current_z, cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('Current Pattern')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(target_z, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title('Target Pattern')
        plt.colorbar(im2, ax=ax2)
        
        similarity = self._compute_pattern_similarity(self.current_pattern)
        fig.suptitle(f'Step {self.step_count}: Similarity = {similarity:.3f}')
        
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
    
    def get_array_info(self) -> Dict[str, Any]:
        """Get array configuration information."""
        return {
            'array_size': self.array_size,
            'n_devices': self.n_devices,
            'device_type': self.device_type,
            'action_mode': self.action_mode,
            'coupling_enabled': self.include_coupling,
            'coupling_type': self.coupling_type,
            'coupling_strength': self.coupling_strength
        }
    
    def set_target_pattern(self, pattern: np.ndarray):
        """Set new target pattern."""
        if pattern.shape != (self.n_rows, self.n_cols, 3):
            raise ValueError(f"Pattern shape must be {(self.n_rows, self.n_cols, 3)}")
        self.target_pattern = pattern.copy()
    
    def analyze_episode(self) -> Dict[str, Any]:
        """Analyze completed episode."""
        if not self.episode_history:
            return {}
        
        total_energy = sum(h['energy'] for h in self.episode_history)
        final_similarity = self.episode_history[-1]['similarity']
        success = final_similarity >= self.success_threshold
        
        return {
            'episode_length': len(self.episode_history),
            'total_energy': total_energy,
            'final_similarity': final_similarity,
            'success': success,
            'average_reward': np.mean([h['reward'] for h in self.episode_history]),
            'energy_efficiency': final_similarity / total_energy if total_energy > 0 else 0,
            'history': self.episode_history.copy()
        }