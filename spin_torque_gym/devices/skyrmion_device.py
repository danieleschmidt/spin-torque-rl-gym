"""Skyrmion-based spintronic device implementation.

This module implements a skyrmion device model for racetrack memory
applications, where skyrmions (topological magnetic textures) are
manipulated for information storage and processing.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import warnings

from .base_device import BaseSpintronicDevice


class SkyrmionDevice(BaseSpintronicDevice):
    """Skyrmion-based spintronic device model.
    
    This class implements the physics of skyrmion devices used in
    racetrack memory and other spintronic applications where topological
    magnetic textures are manipulated for information processing.
    """
    
    def __init__(self, device_params: Dict[str, Any]):
        """Initialize Skyrmion device.
        
        Args:
            device_params: Dictionary containing device parameters
        """
        super().__init__(device_params)
        
        # Validate required parameters
        self._validate_skyrmion_params()
        
        # Skyrmion-specific parameters
        self.dmi_constant = device_params.get('dmi_constant', 3e-3)  # J/m²
        self.skyrmion_radius = device_params.get('skyrmion_radius', 20e-9)  # m
        self.track_width = device_params.get('track_width', 200e-9)  # m
        self.spin_hall_angle = device_params.get('spin_hall_angle', 0.1)
        self.interface_transparency = device_params.get('interface_transparency', 0.5)
        self.pinning_strength = device_params.get('pinning_strength', 0.1)
        self.gilbert_damping = device_params.get('damping', 0.3)
        
        # Material properties specific to skyrmion systems
        self.perpendicular_anisotropy = device_params.get('perpendicular_anisotropy', True)
        self.heavy_metal_layer = device_params.get('heavy_metal_layer', 'Pt')
        self.ferromagnet_layer = device_params.get('ferromagnet_layer', 'Co')
        
        # Cache commonly used values
        self._update_cached_parameters()
    
    def _validate_skyrmion_params(self) -> None:
        """Validate skyrmion-specific parameters."""
        required_params = [
            'volume', 'saturation_magnetization', 'exchange_constant'
        ]
        
        for param in required_params:
            if param not in self.device_params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate that DMI is present (required for skyrmion stability)
        if self.device_params.get('dmi_constant', 0) == 0:
            warnings.warn("Zero DMI constant - skyrmions may not be stable")
        
        # Check skyrmion radius vs exchange length
        exchange_length = self._compute_exchange_length()
        if self.device_params.get('skyrmion_radius', 20e-9) < exchange_length:
            warnings.warn("Skyrmion radius smaller than exchange length - may be unstable")
    
    def _update_cached_parameters(self) -> None:
        """Update cached parameters for efficiency."""
        # Exchange length
        self.exchange_length = self._compute_exchange_length()
        
        # Skyrmion number (topological charge)
        self.skyrmion_charge = -1  # Néel skyrmion
        
        # Characteristic energies
        self.exchange_energy_scale = self.device_params['exchange_constant'] / self.exchange_length
        self.dmi_energy_scale = self.dmi_constant
        
        # Magnus force coefficient for skyrmion dynamics
        self.magnus_coefficient = 4 * np.pi * self.saturation_magnetization * self.thickness
        
        # Effective mass for skyrmion motion
        self.effective_mass = self.magnus_coefficient * self.skyrmion_radius**2
        
        # Mobility parameters
        self.mobility_factor = self.spin_hall_angle * self.interface_transparency
    
    def _compute_exchange_length(self) -> float:
        """Compute magnetic exchange length."""
        A = self.device_params['exchange_constant']
        K = self.device_params.get('uniaxial_anisotropy', 0)
        
        if K > 0:
            # With anisotropy
            exchange_length = np.sqrt(2 * A / K)
        else:
            # Without anisotropy (use demagnetization)
            mu0 = 4 * np.pi * 1e-7
            Ms = self.saturation_magnetization
            exchange_length = np.sqrt(2 * A / (mu0 * Ms**2))
        
        return exchange_length
    
    def compute_effective_field(
        self, 
        position: np.ndarray,
        magnetization_profile: Optional[np.ndarray] = None,
        applied_field: np.ndarray = np.zeros(3)
    ) -> np.ndarray:
        """Compute effective field for skyrmion at given position.
        
        Args:
            position: Skyrmion center position [x, y] (m)
            magnetization_profile: Spatially resolved magnetization (optional)
            applied_field: Applied external field (A/m)
            
        Returns:
            Effective magnetic field (A/m)
        """
        # For skyrmion dynamics, we typically work with collective coordinates
        # rather than detailed magnetization profiles
        
        # Zeeman field
        h_zeeman = applied_field
        
        # Anisotropy field (perpendicular anisotropy for skyrmion stability)
        K_u = self.device_params.get('uniaxial_anisotropy', 1e6)
        if self.perpendicular_anisotropy:
            h_anis = np.array([0, 0, 2 * K_u / (self.mu0 * self.saturation_magnetization)])
        else:
            h_anis = np.zeros(3)
        
        # Demagnetization field (depends on skyrmion configuration)
        h_demag = self._compute_skyrmion_demagnetization_field(position)
        
        # DMI field (stabilizes skyrmion texture)
        h_dmi = self._compute_dmi_field(position)
        
        return h_zeeman + h_anis + h_demag + h_dmi
    
    def _compute_skyrmion_demagnetization_field(self, position: np.ndarray) -> np.ndarray:
        """Compute demagnetization field for skyrmion configuration."""
        # Simplified: assume skyrmion creates local demagnetization field
        # In practice, this would require detailed micromagnetic calculation
        
        # Approximate demagnetization field at skyrmion center
        demag_strength = self.saturation_magnetization / 2  # Approximate
        return np.array([0, 0, -demag_strength])
    
    def _compute_dmi_field(self, position: np.ndarray) -> np.ndarray:
        """Compute DMI field contribution."""
        # For interfacial DMI, field depends on magnetization gradient
        # Simplified: constant DMI field for skyrmion texture
        
        dmi_field_strength = self.dmi_constant / (self.mu0 * self.saturation_magnetization * self.exchange_length)
        
        # DMI field direction depends on skyrmion handedness and position
        # Simplified uniform field
        return np.array([0, 0, dmi_field_strength])
    
    def compute_skyrmion_velocity(
        self,
        current_density: np.ndarray,
        position: np.ndarray,
        external_force: np.ndarray = np.zeros(2)
    ) -> np.ndarray:
        """Compute skyrmion velocity under applied current.
        
        Args:
            current_density: Applied current density [Jx, Jy] (A/m²)
            position: Current skyrmion position [x, y] (m)
            external_force: Additional external force [Fx, Fy] (N)
            
        Returns:
            Skyrmion velocity [vx, vy] (m/s)
        """
        if np.linalg.norm(current_density) < 1e-12:
            return np.zeros(2)
        
        # Current-induced forces
        # Drive force (along current direction)
        j_magnitude = np.linalg.norm(current_density)
        j_direction = current_density / j_magnitude
        
        # Spin-orbit torque efficiency
        force_magnitude = self.mobility_factor * j_magnitude * self.effective_mass
        
        # Drive force (parallel to current)
        f_drive = force_magnitude * j_direction
        
        # Magnus force (perpendicular to current - skyrmion Hall effect)
        hall_angle = self._compute_skyrmion_hall_angle()
        perpendicular_direction = np.array([-j_direction[1], j_direction[0]])
        f_magnus = force_magnitude * np.tan(hall_angle) * perpendicular_direction
        
        # Total current-induced force
        f_current = f_drive + f_magnus
        
        # Add pinning forces
        f_pinning = self._compute_pinning_force(position)
        
        # Total force
        f_total = f_current + external_force + f_pinning
        
        # Velocity from force (including damping)
        # v = F / (β * G) where β is damping coefficient, G is gyromagnetic coupling
        damping_coefficient = self.gilbert_damping * self.magnus_coefficient
        
        velocity = f_total / damping_coefficient
        
        return velocity
    
    def _compute_skyrmion_hall_angle(self) -> float:
        """Compute skyrmion Hall angle."""
        # Skyrmion Hall angle depends on damping parameter
        # Typical values: 10-30 degrees
        
        # Simplified empirical relation
        alpha = self.gilbert_damping
        hall_angle = np.arctan(alpha / 0.1)  # Empirical scaling
        
        # Clip to reasonable range
        hall_angle = np.clip(hall_angle, np.radians(5), np.radians(45))
        
        return hall_angle
    
    def _compute_pinning_force(self, position: np.ndarray) -> np.ndarray:
        """Compute force from pinning sites.
        
        Args:
            position: Skyrmion position [x, y] (m)
            
        Returns:
            Pinning force [Fx, Fy] (N)
        """
        # Simplified pinning model
        # In practice, this would include specific defect locations and strengths
        
        # Random pinning (thermal fluctuations and disorder)
        if self.pinning_strength > 0:
            # Simplified: random force with characteristic strength
            np.random.seed(int(position[0] * 1e12) % 2**32)  # Position-dependent seed
            random_direction = np.random.normal(0, 1, 2)
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            pinning_force_magnitude = self.pinning_strength * self.exchange_energy_scale
            return -pinning_force_magnitude * random_direction
        
        return np.zeros(2)
    
    def compute_skyrmion_stability(
        self,
        position: np.ndarray,
        temperature: float = 300.0
    ) -> float:
        """Compute skyrmion stability factor.
        
        Args:
            position: Skyrmion position
            temperature: Operating temperature (K)
            
        Returns:
            Stability factor (0 = unstable, 1 = stable)
        """
        # Skyrmion energy
        E_skyrmion = self._compute_skyrmion_energy()
        
        # Thermal energy
        kb = 1.38e-23
        E_thermal = kb * temperature
        
        # Stability ratio
        if E_thermal > 0:
            stability = min(1.0, E_skyrmion / (40 * E_thermal))  # 40 kT threshold
        else:
            stability = 1.0
        
        # Additional stability factors
        # Check if skyrmion is within track boundaries
        if hasattr(self, 'track_width'):
            boundary_factor = 1.0
            if position[1] < self.skyrmion_radius or position[1] > self.track_width - self.skyrmion_radius:
                boundary_factor = 0.5  # Reduced stability near edges
            stability *= boundary_factor
        
        return stability
    
    def _compute_skyrmion_energy(self) -> float:
        """Compute total energy of skyrmion configuration."""
        # Exchange energy
        A = self.device_params['exchange_constant']
        E_exchange = 8 * np.pi * A
        
        # DMI energy  
        E_dmi = -4 * np.pi * self.dmi_constant * self.skyrmion_radius
        
        # Anisotropy energy
        K = self.device_params.get('uniaxial_anisotropy', 0)
        E_anisotropy = np.pi * K * self.skyrmion_radius**2 * self.thickness
        
        # Demagnetization energy (approximate)
        mu0 = 4 * np.pi * 1e-7
        Ms = self.saturation_magnetization
        E_demag = mu0 * Ms**2 * self.skyrmion_radius**2 * self.thickness / 2
        
        total_energy = E_exchange + E_dmi + E_anisotropy + E_demag
        
        return total_energy
    
    def compute_resistance(self, skyrmion_positions: List[np.ndarray]) -> float:
        """Compute device resistance based on skyrmion configuration.
        
        Args:
            skyrmion_positions: List of skyrmion center positions
            
        Returns:
            Device resistance (Ω)
        """
        # Base resistance
        base_resistance = self.device_params.get('base_resistance', 1e3)
        
        # Skyrmion contribution (topological Hall effect, anisotropic magnetoresistance)
        skyrmion_resistance_change = 0.0
        
        for pos in skyrmion_positions:
            # Each skyrmion contributes to resistance change
            # This depends on specific measurement geometry
            
            # Simplified: resistance change proportional to skyrmion number
            delta_r = self.device_params.get('skyrmion_resistance_factor', 0.1) * base_resistance
            skyrmion_resistance_change += abs(self.skyrmion_charge) * delta_r
        
        total_resistance = base_resistance + skyrmion_resistance_change
        
        return max(total_resistance, 1.0)
    
    def compute_power_consumption(
        self,
        current_density: np.ndarray,
        pulse_duration: float,
        skyrmion_positions: List[np.ndarray]
    ) -> float:
        """Compute power consumption for skyrmion manipulation.
        
        Args:
            current_density: Applied current density (A/m²)
            pulse_duration: Current pulse duration (s)
            skyrmion_positions: Skyrmion positions
            
        Returns:
            Energy consumed (J)
        """
        if np.linalg.norm(current_density) < 1e-12:
            return 0.0
        
        # Ohmic power consumption
        resistivity = self.device_params.get('resistivity', 2e-7)  # Heavy metal layer
        track_length = self.device_params.get('length', 1e-6)
        
        current_magnitude = np.linalg.norm(current_density)
        current = current_magnitude * self.track_width * self.thickness
        voltage = current * resistivity * track_length / (self.track_width * self.thickness)
        power = voltage * current
        energy = power * pulse_duration
        
        return energy
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device_type': 'Skyrmion Device',
            'geometry': {
                'volume': self.volume,
                'thickness': self.thickness,
                'skyrmion_radius': self.skyrmion_radius,
                'track_width': getattr(self, 'track_width', None),
                'exchange_length': self.exchange_length
            },
            'magnetic_properties': {
                'saturation_magnetization': self.saturation_magnetization,
                'exchange_constant': self.device_params['exchange_constant'],
                'dmi_constant': self.dmi_constant,
                'damping': self.gilbert_damping,
                'perpendicular_anisotropy': self.perpendicular_anisotropy
            },
            'skyrmion_properties': {
                'skyrmion_charge': self.skyrmion_charge,
                'magnus_coefficient': self.magnus_coefficient,
                'effective_mass': self.effective_mass,
                'mobility_factor': self.mobility_factor,
                'pinning_strength': self.pinning_strength
            },
            'material_stack': {
                'heavy_metal_layer': self.heavy_metal_layer,
                'ferromagnet_layer': self.ferromagnet_layer,
                'spin_hall_angle': self.spin_hall_angle,
                'interface_transparency': self.interface_transparency
            }
        }
        
        # Add computed energies
        info['energies'] = {
            'skyrmion_energy': self._compute_skyrmion_energy(),
            'exchange_energy_scale': self.exchange_energy_scale,
            'dmi_energy_scale': self.dmi_energy_scale
        }
        
        return info
    
    def validate_position(self, position: np.ndarray) -> np.ndarray:
        """Validate skyrmion position.
        
        Args:
            position: Input position [x, y]
            
        Returns:
            Validated position
            
        Raises:
            ValueError: If position is invalid
        """
        if position.shape != (2,):
            raise ValueError("Position must be a 2D vector [x, y]")
        
        # Check track boundaries if defined
        if hasattr(self, 'track_width'):
            if position[1] < 0 or position[1] > self.track_width:
                raise ValueError(f"Position y={position[1]} outside track width")
        
        return position.copy()
    
    def estimate_motion_time(
        self,
        initial_position: np.ndarray,
        target_position: np.ndarray,
        current_density: float
    ) -> float:
        """Estimate time for skyrmion to move between positions.
        
        Args:
            initial_position: Starting position
            target_position: Target position  
            current_density: Applied current density magnitude
            
        Returns:
            Estimated motion time (s)
        """
        if current_density < 1e-6:
            return np.inf
        
        # Distance to travel
        distance = np.linalg.norm(target_position - initial_position)
        
        # Average velocity (simplified)
        current_vec = np.array([current_density, 0])  # Along x-direction
        velocity = self.compute_skyrmion_velocity(current_vec, initial_position)
        avg_speed = np.linalg.norm(velocity)
        
        if avg_speed < 1e-12:
            return np.inf
        
        return distance / avg_speed
    
    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """Get device parameter value."""
        return self.device_params.get(param_name, default)
    
    def __repr__(self) -> str:
        """String representation of device."""
        return (f"SkyrmionDevice(radius={self.skyrmion_radius:.1e} m, "
                f"DMI={self.dmi_constant:.2e} J/m², "
                f"Ms={self.saturation_magnetization:.0f} A/m)")