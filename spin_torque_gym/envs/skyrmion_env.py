"""Skyrmion racetrack environment for skyrmion-based memory devices.

This module implements RL environments for controlling skyrmion motion
along racetracks for applications in racetrack memory and skyrmion-based
neuromorphic computing.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..devices import DeviceFactory
from ..physics import LLGSSolver, ThermalFluctuations
from ..rewards import CompositeReward


class SkyrmionRacetrackEnv(gym.Env):
    """Gymnasium environment for controlling skyrmions on racetracks.
    
    This environment enables training RL agents to manipulate skyrmions
    (topologically protected magnetic textures) along racetrack channels
    for memory and computing applications.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        track_length: float = 1000e-9,  # 1 μm
        track_width: float = 200e-9,   # 200 nm
        track_thickness: float = 2e-9,  # 2 nm
        n_skyrmions: int = 1,
        skyrmion_radius: float = 20e-9,  # 20 nm
        target_positions: Optional[List[float]] = None,
        max_steps: int = 150,
        max_current: float = 1e12,  # A/m²
        max_gradient: float = 1e18,  # A/m³
        temperature: float = 300.0,  # K
        include_thermal_fluctuations: bool = True,
        include_pinning: bool = True,
        pinning_strength: float = 0.1,
        reward_components: Optional[Dict[str, Dict]] = None,
        action_mode: str = 'continuous',
        observation_mode: str = 'vector',
        success_threshold: float = 10e-9,  # 10 nm position accuracy
        energy_penalty_weight: float = 0.1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Initialize SkyrmionRacetrack environment.
        
        Args:
            track_length: Length of racetrack (m)
            track_width: Width of racetrack (m)
            track_thickness: Thickness of magnetic layer (m)
            n_skyrmions: Number of skyrmions on track
            skyrmion_radius: Typical skyrmion radius (m)
            target_positions: Target positions along track (m)
            max_steps: Maximum steps per episode
            max_current: Maximum current density magnitude (A/m²)
            max_gradient: Maximum current gradient magnitude (A/m³)
            temperature: Operating temperature (K)
            include_thermal_fluctuations: Whether to include thermal noise
            include_pinning: Whether to include pinning sites
            pinning_strength: Strength of pinning potential
            reward_components: Custom reward function components
            action_mode: 'continuous' or 'discrete'
            observation_mode: 'vector' or 'dict'
            success_threshold: Threshold for successful positioning (m)
            energy_penalty_weight: Weight for energy penalty in reward
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__()

        # Track geometry
        self.track_length = track_length
        self.track_width = track_width
        self.track_thickness = track_thickness

        # Skyrmion parameters
        self.n_skyrmions = n_skyrmions
        self.skyrmion_radius = skyrmion_radius

        # Environment parameters
        self.max_steps = max_steps
        self.max_current = max_current
        self.max_gradient = max_gradient
        self.temperature = temperature
        self.include_thermal = include_thermal_fluctuations
        self.include_pinning = include_pinning
        self.pinning_strength = pinning_strength
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.success_threshold = success_threshold
        self.energy_penalty_weight = energy_penalty_weight
        self.render_mode = render_mode

        # Set target positions
        if target_positions is None:
            # Default: evenly spaced along track
            self.target_positions = np.linspace(
                track_length * 0.2, track_length * 0.8, n_skyrmions
            ).tolist()
        else:
            if len(target_positions) != n_skyrmions:
                raise ValueError("Number of target positions must match number of skyrmions")
            self.target_positions = target_positions

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

        # Initialize device (racetrack material)
        device_factory = DeviceFactory()
        device_params = self._get_default_racetrack_params()
        self.racetrack = device_factory.create_device('skyrmion_track', device_params)

        # Pinning sites (disorder)
        if self.include_pinning:
            self.pinning_sites = self._generate_pinning_sites()
        else:
            self.pinning_sites = []

        # Initialize reward function
        if reward_components is None:
            reward_components = self._get_default_reward_components()
        self.reward_function = CompositeReward(reward_components)

        # Setup action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()

        # Environment state
        self.skyrmion_positions = None
        self.skyrmion_velocities = None
        self.step_count = 0
        self.total_energy = 0.0
        self.episode_history = []

        # Rendering
        self.renderer = None
        if render_mode == 'human':
            self._init_renderer()

    def _get_default_racetrack_params(self) -> Dict[str, Any]:
        """Get default racetrack material parameters."""
        return {
            'length': self.track_length,
            'width': self.track_width,
            'thickness': self.track_thickness,
            'saturation_magnetization': 580e3,  # A/m (typical for Pt/Co/MgO)
            'exchange_constant': 15e-12,  # J/m
            'dmi_constant': 3e-3,  # J/m² (interfacial DMI)
            'anisotropy_constant': 0.8e6,  # J/m³
            'damping': 0.3,  # Higher damping in heavy metal layers
            'spin_hall_angle': 0.1,  # Pt spin Hall angle
            'resistivity': 2e-7,  # Ω⋅m
            'easy_axis': np.array([0, 0, 1])  # Perpendicular anisotropy
        }

    def _generate_pinning_sites(self) -> List[Dict[str, float]]:
        """Generate random pinning sites along the racetrack."""
        n_sites = int(self.track_length / (20 * self.skyrmion_radius))  # ~1 site per 20 radii
        sites = []

        for _ in range(n_sites):
            position = self._np_random.uniform(0, self.track_length)
            strength = self._np_random.uniform(0.5, 2.0) * self.pinning_strength
            sites.append({'position': position, 'strength': strength})

        return sites

    def _get_default_reward_components(self) -> Dict[str, Dict]:
        """Get default reward function components."""
        return {
            'positioning': {
                'weight': 10.0,
                'function': self._positioning_reward
            },
            'energy': {
                'weight': -self.energy_penalty_weight,
                'function': lambda obs, action, next_obs, info:
                    -info.get('step_energy', 0.0) / 1e-15  # Normalize to fJ
            },
            'velocity': {
                'weight': -1.0,
                'function': self._velocity_penalty
            },
            'stability': {
                'weight': 5.0,
                'function': self._stability_reward
            },
            'efficiency': {
                'weight': 2.0,
                'function': self._efficiency_reward
            }
        }

    def _positioning_reward(self, obs, action, next_obs, info) -> float:
        """Reward for accurate positioning."""
        total_reward = 0.0
        position_errors = info.get('position_errors', [])

        for error in position_errors:
            if error < self.success_threshold:
                total_reward += 10.0  # Success bonus
            else:
                # Distance-based reward
                normalized_error = error / (self.track_length * 0.1)
                total_reward += max(0, 5.0 * (1.0 - normalized_error))

        return total_reward / len(position_errors) if position_errors else 0.0

    def _velocity_penalty(self, obs, action, next_obs, info) -> float:
        """Penalty for excessive velocity."""
        velocities = info.get('velocities', [])
        total_penalty = 0.0

        for vel in velocities:
            # Penalty for velocities above reasonable limit
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > 100.0:  # m/s
                total_penalty += (vel_magnitude - 100.0) / 100.0

        return total_penalty

    def _stability_reward(self, obs, action, next_obs, info) -> float:
        """Reward for maintaining skyrmion stability."""
        stability_factors = info.get('stability_factors', [])
        return np.mean(stability_factors) if stability_factors else 0.0

    def _efficiency_reward(self, obs, action, next_obs, info) -> float:
        """Reward for energy-efficient motion."""
        displacement = info.get('total_displacement', 0.0)
        energy = info.get('step_energy', 1e-15)

        if energy > 0:
            efficiency = displacement / (energy / 1e-15)  # displacement per fJ
            return min(efficiency, 10.0)
        return 0.0

    def _setup_action_space(self):
        """Setup action space based on action mode."""
        if self.action_mode == 'continuous':
            # Continuous: [current_x, current_y, gradient_x, gradient_y, duration]
            self.action_space = spaces.Box(
                low=np.array([-self.max_current, -self.max_current,
                             -self.max_gradient, -self.max_gradient, 0.0]),
                high=np.array([self.max_current, self.max_current,
                              self.max_gradient, self.max_gradient, 2e-9]),
                dtype=np.float32
            )
        elif self.action_mode == 'discrete':
            # Discrete: 5 current directions × 3 gradient levels × 3 durations
            self.current_directions = np.array([
                [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]  # +x, -x, +y, -y, off
            ])
            self.gradient_levels = np.array([0, self.max_gradient * 0.5, self.max_gradient])
            self.duration_levels = np.array([0.1e-9, 0.5e-9, 1.0e-9])

            n_actions = len(self.current_directions) * len(self.gradient_levels) * len(self.duration_levels)
            self.action_space = spaces.Discrete(n_actions)
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")

    def _setup_observation_space(self):
        """Setup observation space based on observation mode."""
        if self.observation_mode == 'vector':
            # Vector: [pos_x, pos_y, vel_x, vel_y] per skyrmion + targets + global info
            obs_size = self.n_skyrmions * 4 + self.n_skyrmions * 2 + 4  # positions, velocities, targets, info
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
        elif self.observation_mode == 'dict':
            self.observation_space = spaces.Dict({
                'positions': spaces.Box(
                    0, self.track_length, shape=(self.n_skyrmions, 2), dtype=np.float32
                ),
                'velocities': spaces.Box(
                    -np.inf, np.inf, shape=(self.n_skyrmions, 2), dtype=np.float32
                ),
                'target_positions': spaces.Box(
                    0, self.track_length, shape=(self.n_skyrmions,), dtype=np.float32
                ),
                'position_errors': spaces.Box(
                    0, np.inf, shape=(self.n_skyrmions,), dtype=np.float32
                ),
                'steps_remaining': spaces.Box(0, self.max_steps, shape=(1,), dtype=int),
                'total_energy': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")

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

        # Set initial skyrmion positions
        if 'initial_positions' in options:
            self.skyrmion_positions = np.array(options['initial_positions'])
        else:
            # Random initial positions along track
            self.skyrmion_positions = np.zeros((self.n_skyrmions, 2))
            for i in range(self.n_skyrmions):
                # x-position: random along track
                x_pos = self._np_random.uniform(
                    self.skyrmion_radius,
                    self.track_length - self.skyrmion_radius
                )
                # y-position: centered in track width
                y_pos = self.track_width / 2
                self.skyrmion_positions[i] = [x_pos, y_pos]

        # Initialize velocities
        self.skyrmion_velocities = np.zeros((self.n_skyrmions, 2))

        # Set target positions
        if 'target_positions' in options:
            self.target_positions = options['target_positions']

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: Union[np.ndarray, int]
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.skyrmion_positions is None:
            raise RuntimeError("Environment must be reset before calling step")

        # Store previous state
        prev_positions = self.skyrmion_positions.copy()
        prev_errors = self._compute_position_errors()

        # Parse and apply action
        step_info = self._apply_skyrmion_dynamics(action)

        # Update state
        self.total_energy += step_info['energy_consumed']
        self.step_count += 1

        # Calculate reward
        current_errors = self._compute_position_errors()
        position_improvements = [prev_err - curr_err for prev_err, curr_err in zip(prev_errors, current_errors)]

        # Check success condition
        is_success = all(error < self.success_threshold for error in current_errors)

        reward_info = {
            'is_success': is_success,
            'step_energy': step_info['energy_consumed'],
            'position_errors': current_errors,
            'position_improvements': position_improvements,
            'velocities': self.skyrmion_velocities.tolist(),
            'stability_factors': step_info['stability_factors'],
            'total_displacement': step_info['total_displacement']
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
            'positions': self.skyrmion_positions.copy(),
            'velocities': self.skyrmion_velocities.copy(),
            'reward': reward,
            'energy': step_info['energy_consumed'],
            'position_errors': current_errors
        })

        info = self._get_info()
        info.update(step_info)
        info.update(reward_info)

        return observation, reward, terminated, truncated, info

    def _apply_skyrmion_dynamics(self, action: Union[np.ndarray, int]) -> Dict[str, Any]:
        """Apply skyrmion dynamics based on action."""
        # Parse action
        if self.action_mode == 'continuous':
            current_x = float(action[0])
            current_y = float(action[1])
            gradient_x = float(action[2]) if len(action) > 2 else 0.0
            gradient_y = float(action[3]) if len(action) > 3 else 0.0
            duration = float(action[4]) if len(action) > 4 else 1e-9
        elif self.action_mode == 'discrete':
            action_idx = int(action)
            direction_idx = action_idx // (len(self.gradient_levels) * len(self.duration_levels))
            gradient_idx = (action_idx // len(self.duration_levels)) % len(self.gradient_levels)
            duration_idx = action_idx % len(self.duration_levels)

            direction = self.current_directions[direction_idx]
            current_x = direction[0] * self.max_current * 0.5
            current_y = direction[1] * self.max_current * 0.5
            gradient_x = self.gradient_levels[gradient_idx]
            gradient_y = 0.0
            duration = self.duration_levels[duration_idx]
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")

        # Clip values
        current_x = np.clip(current_x, -self.max_current, self.max_current)
        current_y = np.clip(current_y, -self.max_current, self.max_current)
        gradient_x = np.clip(gradient_x, -self.max_gradient, self.max_gradient)
        gradient_y = np.clip(gradient_y, -self.max_gradient, self.max_gradient)
        duration = np.clip(duration, 1e-12, 2e-9)

        total_energy = 0.0
        total_displacement = 0.0
        stability_factors = []

        # Simulate each skyrmion
        for i in range(self.n_skyrmions):
            prev_pos = self.skyrmion_positions[i].copy()

            # Apply skyrmion dynamics
            new_pos, new_vel, stability, energy = self._simulate_skyrmion_motion(
                i, current_x, current_y, gradient_x, gradient_y, duration
            )

            # Update position and velocity
            self.skyrmion_positions[i] = new_pos
            self.skyrmion_velocities[i] = new_vel

            # Accumulate metrics
            displacement = np.linalg.norm(new_pos - prev_pos)
            total_displacement += displacement
            total_energy += energy
            stability_factors.append(stability)

        return {
            'energy_consumed': total_energy,
            'total_displacement': total_displacement,
            'stability_factors': stability_factors,
            'current_applied': [current_x, current_y],
            'gradient_applied': [gradient_x, gradient_y],
            'duration': duration
        }

    def _simulate_skyrmion_motion(
        self,
        skyrmion_idx: int,
        current_x: float,
        current_y: float,
        gradient_x: float,
        gradient_y: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Simulate motion of a single skyrmion."""
        current_pos = self.skyrmion_positions[skyrmion_idx]
        current_vel = self.skyrmion_velocities[skyrmion_idx]

        # Skyrmion Hall angle (typically ~10-30 degrees)
        hall_angle = np.radians(20)

        # Magnus force coefficient
        magnus_coeff = 4 * np.pi * self.racetrack.get_parameter('saturation_magnetization', 580e3)

        # Current-induced force (spin-orbit torque)
        j_vector = np.array([current_x, current_y])
        if np.linalg.norm(j_vector) > 0:
            # Force parallel to current (drive force)
            f_drive = self.racetrack.get_parameter('spin_hall_angle', 0.1) * np.linalg.norm(j_vector)
            drive_direction = j_vector / np.linalg.norm(j_vector)

            # Force perpendicular to current (Magnus force)
            perpendicular_direction = np.array([-drive_direction[1], drive_direction[0]])
            f_magnus = f_drive * np.tan(hall_angle)

            # Total force
            force = f_drive * drive_direction + f_magnus * perpendicular_direction
        else:
            force = np.array([0.0, 0.0])

        # Add force from current gradient (additional drive mechanism)
        if gradient_x != 0 or gradient_y != 0:
            gradient_force = np.array([gradient_x, gradient_y]) * 1e-24  # Scale factor
            force += gradient_force

        # Add pinning forces
        pinning_force = self._compute_pinning_force(current_pos)
        force += pinning_force

        # Add thermal fluctuations
        if self.include_thermal:
            thermal_force = self._compute_thermal_force()
            force += thermal_force

        # Integrate motion (simplified dynamics)
        damping = self.racetrack.get_parameter('damping', 0.3)
        mass_eff = magnus_coeff * self.skyrmion_radius**2  # Effective mass

        # Simple Euler integration
        dt = duration / 10  # Sub-steps
        pos = current_pos.copy()
        vel = current_vel.copy()

        for _ in range(10):
            # Update velocity with damping
            accel = force / mass_eff - damping * vel
            vel += accel * dt

            # Update position
            pos += vel * dt

            # Boundary conditions (keep skyrmion in track)
            pos[0] = np.clip(pos[0], self.skyrmion_radius,
                           self.track_length - self.skyrmion_radius)
            pos[1] = np.clip(pos[1], self.skyrmion_radius,
                           self.track_width - self.skyrmion_radius)

            # Reflect velocity if hitting boundaries
            if pos[0] <= self.skyrmion_radius or pos[0] >= self.track_length - self.skyrmion_radius:
                vel[0] *= -0.5  # Inelastic collision
            if pos[1] <= self.skyrmion_radius or pos[1] >= self.track_width - self.skyrmion_radius:
                vel[1] *= -0.5

        # Calculate stability factor (0 = unstable, 1 = stable)
        vel_magnitude = np.linalg.norm(vel)
        stability = np.exp(-vel_magnitude / 50.0)  # Decay with high velocity

        # Calculate energy consumption
        resistivity = self.racetrack.get_parameter('resistivity', 2e-7)
        current_magnitude = np.linalg.norm([current_x, current_y])
        if current_magnitude > 0:
            voltage = current_magnitude * resistivity * self.track_length / (self.track_width * self.track_thickness)
            energy = voltage**2 / resistivity * duration * self.track_width * self.track_thickness / self.track_length
        else:
            energy = 0.0

        return pos, vel, stability, energy

    def _compute_pinning_force(self, position: np.ndarray) -> np.ndarray:
        """Compute force from pinning sites."""
        if not self.include_pinning:
            return np.array([0.0, 0.0])

        total_force = np.array([0.0, 0.0])

        for site in self.pinning_sites:
            site_pos = np.array([site['position'], self.track_width / 2])
            distance_vec = position - site_pos
            distance = np.linalg.norm(distance_vec)

            if distance < 3 * self.skyrmion_radius:  # Pinning range
                # Attractive pinning force (toward site)
                force_magnitude = site['strength'] * np.exp(-distance / self.skyrmion_radius)
                if distance > 0:
                    force_direction = -distance_vec / distance
                    total_force += force_magnitude * force_direction

        return total_force

    def _compute_thermal_force(self) -> np.ndarray:
        """Compute random thermal force."""
        # Thermal force based on fluctuation-dissipation theorem
        kb = 1.38e-23  # Boltzmann constant
        thermal_energy = kb * self.temperature

        # Random force with appropriate magnitude
        force_magnitude = np.sqrt(2 * thermal_energy / (self.skyrmion_radius * 1e-9))
        force_direction = self._np_random.normal(0, 1, 2)
        force_direction = force_direction / np.linalg.norm(force_direction)

        return force_magnitude * force_direction

    def _compute_position_errors(self) -> List[float]:
        """Compute position errors for all skyrmions."""
        errors = []
        for i in range(self.n_skyrmions):
            target_pos = np.array([self.target_positions[i], self.track_width / 2])
            current_pos = self.skyrmion_positions[i]
            error = np.linalg.norm(current_pos - target_pos)
            errors.append(error)
        return errors

    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """Get current observation."""
        position_errors = self._compute_position_errors()

        if self.observation_mode == 'vector':
            # Flatten positions and velocities
            positions_flat = self.skyrmion_positions.flatten()
            velocities_flat = self.skyrmion_velocities.flatten()
            targets_flat = np.array([[pos, self.track_width / 2] for pos in self.target_positions]).flatten()

            # Global information
            steps_remaining_norm = (self.max_steps - self.step_count) / self.max_steps
            energy_norm = self.total_energy / 1e-15  # Normalize to fJ
            avg_error_norm = np.mean(position_errors) / (self.track_length * 0.1)
            avg_velocity_norm = np.mean([np.linalg.norm(vel) for vel in self.skyrmion_velocities]) / 100.0

            obs = np.concatenate([
                positions_flat / self.track_length,  # Normalize positions
                velocities_flat / 100.0,            # Normalize velocities
                targets_flat / self.track_length,   # Normalize targets
                [steps_remaining_norm, energy_norm, avg_error_norm, avg_velocity_norm]
            ])

            return obs.astype(np.float32)

        elif self.observation_mode == 'dict':
            return {
                'positions': self.skyrmion_positions.astype(np.float32),
                'velocities': self.skyrmion_velocities.astype(np.float32),
                'target_positions': np.array(self.target_positions, dtype=np.float32),
                'position_errors': np.array(position_errors, dtype=np.float32),
                'steps_remaining': np.array([self.max_steps - self.step_count], dtype=int),
                'total_energy': np.array([self.total_energy], dtype=np.float32)
            }

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        position_errors = self._compute_position_errors()

        return {
            'step_count': self.step_count,
            'total_energy': self.total_energy,
            'position_errors': position_errors,
            'average_error': np.mean(position_errors),
            'is_success': all(error < self.success_threshold for error in position_errors),
            'n_skyrmions': self.n_skyrmions,
            'track_length': self.track_length,
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

            self.fig, (self.ax_track, self.ax_metrics) = plt.subplots(2, 1, figsize=(12, 8))
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
            self.ax_track.clear()
            self.ax_metrics.clear()

            # Plot racetrack
            track_rect = plt.Rectangle(
                (0, 0), self.track_length * 1e9, self.track_width * 1e9,
                fill=False, edgecolor='black', linewidth=2
            )
            self.ax_track.add_patch(track_rect)

            # Plot skyrmions
            for i, pos in enumerate(self.skyrmion_positions):
                circle = plt.Circle(
                    (pos[0] * 1e9, pos[1] * 1e9),
                    self.skyrmion_radius * 1e9,
                    color='red', alpha=0.7, label=f'Skyrmion {i}' if i == 0 else ""
                )
                self.ax_track.add_patch(circle)

                # Velocity arrow
                if np.linalg.norm(self.skyrmion_velocities[i]) > 0:
                    self.ax_track.arrow(
                        pos[0] * 1e9, pos[1] * 1e9,
                        self.skyrmion_velocities[i][0] * 1e9 * 1e-6,  # Scale for visibility
                        self.skyrmion_velocities[i][1] * 1e9 * 1e-6,
                        head_width=5, head_length=10, fc='darkred', ec='darkred'
                    )

            # Plot target positions
            for i, target_x in enumerate(self.target_positions):
                target_y = self.track_width / 2
                self.ax_track.scatter(
                    target_x * 1e9, target_y * 1e9,
                    marker='x', s=100, color='blue',
                    label='Targets' if i == 0 else ""
                )

            # Plot pinning sites
            if self.include_pinning:
                for site in self.pinning_sites:
                    self.ax_track.scatter(
                        site['position'] * 1e9, self.track_width / 2 * 1e9,
                        marker='v', s=20, color='gray', alpha=0.5
                    )

            self.ax_track.set_xlim(0, self.track_length * 1e9)
            self.ax_track.set_ylim(-self.track_width * 1e9 * 0.1, self.track_width * 1e9 * 1.1)
            self.ax_track.set_xlabel('Position (nm)')
            self.ax_track.set_ylabel('Width (nm)')
            self.ax_track.set_title(f'Skyrmion Racetrack - Step {self.step_count}')
            self.ax_track.legend()
            self.ax_track.set_aspect('equal')

            # Plot metrics
            if self.episode_history:
                steps = [h['step'] for h in self.episode_history]
                avg_errors = [np.mean(h['position_errors']) for h in self.episode_history]
                energies = [h['energy'] for h in self.episode_history]

                self.ax_metrics.plot(steps, np.array(avg_errors) * 1e9, 'b-', label='Avg Error (nm)')
                self.ax_metrics.axhline(y=self.success_threshold * 1e9, color='r', linestyle='--', label='Success threshold')

                # Secondary y-axis for energy
                ax2 = self.ax_metrics.twinx()
                ax2.plot(steps, np.array(energies) * 1e15, 'g-', label='Energy (fJ)')
                ax2.set_ylabel('Energy (fJ)', color='g')

                self.ax_metrics.set_xlabel('Step')
                self.ax_metrics.set_ylabel('Position Error (nm)', color='b')
                self.ax_metrics.set_title('Training Progress')
                self.ax_metrics.legend(loc='upper left')
                ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        except Exception as e:
            warnings.warn(f"Rendering error: {e}")

    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot simplified racetrack view
        track_rect = plt.Rectangle(
            (0, 0), self.track_length * 1e9, self.track_width * 1e9,
            fill=False, edgecolor='black', linewidth=2
        )
        ax.add_patch(track_rect)

        # Plot skyrmions and targets
        for i, pos in enumerate(self.skyrmion_positions):
            ax.scatter(pos[0] * 1e9, pos[1] * 1e9, s=100, color='red', alpha=0.7)
            ax.scatter(self.target_positions[i] * 1e9, self.track_width / 2 * 1e9,
                      marker='x', s=100, color='blue')

        ax.set_xlim(0, self.track_length * 1e9)
        ax.set_ylim(-self.track_width * 1e9 * 0.1, self.track_width * 1e9 * 1.1)
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Width (nm)')
        ax.set_title(f'Step {self.step_count}: Avg Error = {np.mean(self._compute_position_errors()) * 1e9:.1f} nm')
        ax.set_aspect('equal')

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

    def get_racetrack_info(self) -> Dict[str, Any]:
        """Get racetrack configuration information."""
        return {
            'track_dimensions': [self.track_length, self.track_width, self.track_thickness],
            'n_skyrmions': self.n_skyrmions,
            'skyrmion_radius': self.skyrmion_radius,
            'target_positions': self.target_positions.copy(),
            'pinning_enabled': self.include_pinning,
            'n_pinning_sites': len(self.pinning_sites),
            'success_threshold': self.success_threshold
        }

    def set_target_positions(self, positions: List[float]):
        """Set new target positions."""
        if len(positions) != self.n_skyrmions:
            raise ValueError("Number of positions must match number of skyrmions")

        for pos in positions:
            if not (0 <= pos <= self.track_length):
                raise ValueError(f"Position {pos} outside track bounds")

        self.target_positions = positions

    def analyze_episode(self) -> Dict[str, Any]:
        """Analyze completed episode."""
        if not self.episode_history:
            return {}

        total_energy = sum(h['energy'] for h in self.episode_history)
        final_errors = self.episode_history[-1]['position_errors']
        success = all(error < self.success_threshold for error in final_errors)

        # Calculate average positioning accuracy over time
        error_progression = [np.mean(h['position_errors']) for h in self.episode_history]

        return {
            'episode_length': len(self.episode_history),
            'total_energy': total_energy,
            'final_errors': final_errors,
            'average_final_error': np.mean(final_errors),
            'success': success,
            'average_reward': np.mean([h['reward'] for h in self.episode_history]),
            'positioning_efficiency': (1.0 / np.mean(final_errors)) if np.mean(final_errors) > 0 else 0,
            'error_progression': error_progression,
            'history': self.episode_history.copy()
        }
