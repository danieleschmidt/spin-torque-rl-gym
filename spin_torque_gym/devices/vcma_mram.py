"""Voltage-Controlled Magnetic Anisotropy MRAM device implementation.

This module implements a VCMA-MRAM device model that uses electric field
to modulate magnetic anisotropy for low-power magnetization switching.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

from .base_device import BaseSpintronicDevice


class VCMAMRAMDevice(BaseSpintronicDevice):
    """Voltage-Controlled Magnetic Anisotropy MRAM device model.
    
    This class implements the physics of VCMA-MRAM devices where magnetization
    switching is achieved by electrically modulating the magnetic anisotropy
    through voltage-controlled magnetic anisotropy effect.
    """

    def __init__(self, device_params: Dict[str, Any]):
        """Initialize VCMA-MRAM device.
        
        Args:
            device_params: Dictionary containing device parameters
        """
        super().__init__(device_params)

        # Validate required parameters
        self._validate_vcma_params()

        # VCMA-specific parameters
        self.vcma_coefficient = device_params.get('vcma_coefficient', 100e-6)  # J/(V⋅m)
        self.dielectric_thickness = device_params.get('dielectric_thickness', 1e-9)  # MgO thickness
        self.dielectric_constant = device_params.get('dielectric_constant', 25.0)  # MgO εᵣ
        self.breakdown_voltage = device_params.get('breakdown_voltage', 2.0)  # V
        self.leakage_resistance = device_params.get('leakage_resistance', 1e12)  # Ω
        self.capacitance_per_area = device_params.get('capacitance_per_area', None)

        # Cache commonly used values
        self._update_cached_parameters()

    def _validate_vcma_params(self) -> None:
        """Validate VCMA-specific parameters."""
        required_params = [
            'volume', 'saturation_magnetization', 'damping',
            'uniaxial_anisotropy', 'easy_axis'
        ]

        for param in required_params:
            if param not in self.device_params:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate parameter ranges
        if self.device_params.get('vcma_coefficient', 100e-6) < 0:
            warnings.warn("Negative VCMA coefficient indicates inverted VCMA effect")

    def _update_cached_parameters(self) -> None:
        """Update cached parameters for efficiency."""
        # Calculate device capacitance
        epsilon_0 = 8.854e-12  # F/m
        self.area = self.device_params.get('area', self.volume / self.thickness)

        if self.capacitance_per_area is None:
            self.capacitance = (epsilon_0 * self.dielectric_constant * self.area /
                              self.dielectric_thickness)
        else:
            self.capacitance = self.capacitance_per_area * self.area

        # Base anisotropy (without voltage)
        self.base_anisotropy = self.device_params['uniaxial_anisotropy']

        # Maximum electric field
        self.max_electric_field = self.breakdown_voltage / self.dielectric_thickness

        # Thermal stability factor
        kb = 1.38e-23
        temperature = self.device_params.get('temperature', 300.0)
        self.thermal_energy = kb * temperature

        # Cache for efficiency
        self.energy_scale = self.base_anisotropy * self.volume

    def compute_effective_field(
        self,
        magnetization: np.ndarray,
        applied_field: np.ndarray,
        applied_voltage: float = 0.0
    ) -> np.ndarray:
        """Compute effective magnetic field including voltage-modified anisotropy.
        
        Args:
            magnetization: Current magnetization vector (normalized)
            applied_field: Applied external magnetic field (A/m)
            applied_voltage: Applied voltage across dielectric (V)
            
        Returns:
            Effective magnetic field (A/m)
        """
        # Voltage-modified anisotropy
        effective_anisotropy = self._compute_effective_anisotropy(applied_voltage)

        # Uniaxial anisotropy field with VCMA modification
        easy_axis = self.device_params['easy_axis']
        h_anis = (2 * effective_anisotropy / (self.mu0 * self.saturation_magnetization)) * np.dot(magnetization, easy_axis) * easy_axis

        # Exchange field (simplified - assuming uniform magnetization)
        exchange_constant = self.device_params.get('exchange_constant', 0.0)
        h_exchange = np.zeros(3)  # For uniform state

        # Demagnetization field (shape-dependent)
        h_demag = self._compute_demagnetization_field(magnetization)

        # Thermal field (if temperature is specified)
        temperature = self.device_params.get('temperature', 0.0)
        h_thermal = self._compute_thermal_field(temperature) if temperature > 0 else np.zeros(3)

        return applied_field + h_anis + h_exchange + h_demag + h_thermal

    def _compute_effective_anisotropy(self, voltage: float) -> float:
        """Compute effective anisotropy with VCMA effect.
        
        Args:
            voltage: Applied voltage (V)
            
        Returns:
            Effective anisotropy constant (J/m³)
        """
        # Clip voltage to prevent breakdown
        voltage = np.clip(voltage, -self.breakdown_voltage, self.breakdown_voltage)

        # Electric field
        electric_field = voltage / self.dielectric_thickness

        # VCMA effect: ΔK = ξ × E where ξ is VCMA coefficient (J/(V⋅m))
        # For switching, VCMA should reduce anisotropy - use negative sign
        anisotropy_change = -self.vcma_coefficient * abs(voltage) / (self.dielectric_thickness**2)

        effective_anisotropy = self.base_anisotropy + anisotropy_change

        # Ensure anisotropy doesn't become too negative (would flip easy axis)
        min_anisotropy = -0.5 * self.base_anisotropy
        effective_anisotropy = max(effective_anisotropy, min_anisotropy)

        return effective_anisotropy

    def _compute_demagnetization_field(self, magnetization: np.ndarray) -> np.ndarray:
        """Compute demagnetization field based on device geometry."""
        # Simplified demagnetization field for thin film geometry
        aspect_ratio = self.device_params.get('aspect_ratio', 1.0)

        # Demagnetization factors (approximate for elliptical geometry)
        if aspect_ratio >= 1.0:
            n_x = 1.0 / (1.0 + aspect_ratio)
            n_y = aspect_ratio / (1.0 + aspect_ratio)
        else:
            n_x = aspect_ratio / (1.0 + aspect_ratio)
            n_y = 1.0 / (1.0 + aspect_ratio)
        n_z = 1.0 - n_x - n_y

        n_tensor = np.array([n_x, n_y, n_z])
        h_demag = -self.saturation_magnetization * n_tensor * magnetization

        return h_demag

    def _compute_thermal_field(self, temperature: float) -> np.ndarray:
        """Compute thermal fluctuation field."""
        if temperature <= 0:
            return np.zeros(3)

        # Thermal field magnitude based on fluctuation-dissipation theorem
        kb = 1.38e-23  # Boltzmann constant
        alpha = self.device_params.get('damping', 0.01)
        gamma = 2.21e5  # Gyromagnetic ratio (m/(A⋅s))

        # Thermal field strength
        h_thermal_strength = np.sqrt(
            2 * alpha * kb * temperature /
            (self.mu0 * self.saturation_magnetization * self.volume * gamma)
        )

        # Random field direction (simplified - should be random in practice)
        return h_thermal_strength * np.array([0.0, 0.0, 0.0])

    def compute_switching_probability(
        self,
        voltage: float,
        pulse_duration: float,
        temperature: float = 300.0,
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        """Compute switching probability for voltage-assisted switching.
        
        Args:
            voltage: Applied voltage (V)
            pulse_duration: Voltage pulse duration (s)
            temperature: Operating temperature (K)
            initial_state: Initial magnetization state
            
        Returns:
            Switching probability (0 to 1)
        """
        if initial_state is None:
            # Assume initially in easy axis direction
            initial_state = self.device_params['easy_axis'].copy()

        # Compute effective anisotropy with voltage
        k_eff = self._compute_effective_anisotropy(voltage)

        # Energy barrier between states
        energy_barrier = k_eff * self.volume

        # Thermal energy
        kb = 1.38e-23
        thermal_energy = kb * temperature

        if energy_barrier <= 0:
            # Anisotropy eliminated or reversed - deterministic switching
            return 1.0

        if thermal_energy <= 0:
            # Zero temperature - no thermal switching
            return 0.0

        # Switching rate (Arrhenius law)
        attempt_frequency = 1e9  # Typical attempt frequency (Hz)
        switching_rate = attempt_frequency * np.exp(-energy_barrier / thermal_energy)

        # Switching probability over pulse duration
        probability = 1.0 - np.exp(-switching_rate * pulse_duration)

        return min(probability, 1.0)

    def compute_resistance(self, magnetization: np.ndarray) -> float:
        """Compute device resistance based on magnetization state.
        
        Args:
            magnetization: Current magnetization vector
            
        Returns:
            Device resistance (Ω)
        """
        # MTJ resistance based on TMR effect
        reference_mag = self.device_params.get('reference_magnetization', np.array([0, 0, 1]))
        reference_mag = reference_mag / np.linalg.norm(reference_mag)

        cos_theta = np.dot(magnetization, reference_mag)

        r_p = self.device_params.get('resistance_parallel', 1e3)
        r_ap = self.device_params.get('resistance_antiparallel', 2e3)

        # TMR effect
        resistance = r_p + (r_ap - r_p) * (1 - cos_theta) / 2

        return max(resistance, 1.0)  # Minimum resistance

    def compute_power_consumption(
        self,
        voltage: float,
        pulse_duration: float,
        magnetization: Optional[np.ndarray] = None
    ) -> float:
        """Compute power consumption for VCMA switching operation.
        
        Args:
            voltage: Applied voltage (V)
            pulse_duration: Pulse duration (s)
            magnetization: Current magnetization state
            
        Returns:
            Energy consumed (J)
        """
        if abs(voltage) < 1e-12:
            return 0.0

        # Power consumption dominated by capacitive charging and leakage
        # Capacitive energy: E_cap = 0.5 * C * V²
        capacitive_energy = 0.5 * self.capacitance * voltage**2

        # Leakage current energy: E_leak = V² * t / R_leak
        leakage_energy = voltage**2 * pulse_duration / self.leakage_resistance

        total_energy = capacitive_energy + leakage_energy

        return total_energy

    def get_switching_threshold(self) -> Dict[str, float]:
        """Get critical voltage for magnetization switching.
        
        Returns:
            Dictionary with threshold parameters
        """
        # Critical voltage where anisotropy goes to zero
        v_critical = abs(self.base_anisotropy * self.thickness / self.vcma_coefficient)

        # Clip to breakdown voltage
        v_critical = min(v_critical, self.breakdown_voltage)

        # Thermal switching voltage (lower threshold)
        kb = 1.38e-23
        temperature = self.device_params.get('temperature', 300.0)
        thermal_energy = kb * temperature

        # Voltage needed to reduce barrier to ~40 kT (practical switching)
        barrier_reduction_needed = self.energy_scale - 40 * thermal_energy
        if barrier_reduction_needed > 0:
            v_thermal = (barrier_reduction_needed / (self.vcma_coefficient * self.area)) * self.dielectric_thickness
        else:
            v_thermal = 0.0

        v_thermal = min(v_thermal, self.breakdown_voltage)

        return {
            'critical_voltage': v_critical,
            'thermal_switching_voltage': v_thermal,
            'breakdown_voltage': self.breakdown_voltage,
            'vcma_coefficient': self.vcma_coefficient
        }

    def validate_magnetization(self, magnetization: np.ndarray) -> np.ndarray:
        """Validate and normalize magnetization vector.
        
        Args:
            magnetization: Input magnetization vector
            
        Returns:
            Normalized magnetization vector
            
        Raises:
            ValueError: If magnetization vector is invalid
        """
        if magnetization.shape != (3,):
            raise ValueError("Magnetization must be a 3D vector")

        magnitude = np.linalg.norm(magnetization)
        if magnitude < 1e-12:
            raise ValueError("Magnetization vector cannot have zero magnitude")

        # Normalize to unit vector
        return magnetization / magnitude

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information.
        
        Returns:
            Dictionary containing device parameters and computed values
        """
        info = {
            'device_type': 'VCMA-MRAM',
            'geometry': {
                'volume': self.volume,
                'thickness': self.thickness,
                'area': self.area,
                'aspect_ratio': self.device_params.get('aspect_ratio', 1.0)
            },
            'magnetic_properties': {
                'saturation_magnetization': self.saturation_magnetization,
                'damping': self.device_params['damping'],
                'base_anisotropy': self.base_anisotropy,
                'easy_axis': self.device_params['easy_axis'].tolist()
            },
            'vcma_properties': {
                'vcma_coefficient': self.vcma_coefficient,
                'dielectric_thickness': self.dielectric_thickness,
                'dielectric_constant': self.dielectric_constant,
                'breakdown_voltage': self.breakdown_voltage,
                'max_electric_field': self.max_electric_field
            },
            'electrical_properties': {
                'capacitance': self.capacitance,
                'leakage_resistance': self.leakage_resistance,
                'resistance_parallel': self.device_params.get('resistance_parallel', 1e3),
                'resistance_antiparallel': self.device_params.get('resistance_antiparallel', 2e3)
            },
            'computed_values': {
                'energy_scale': self.energy_scale,
                'thermal_energy': self.thermal_energy
            }
        }

        # Add switching threshold information
        info['switching_threshold'] = self.get_switching_threshold()

        return info

    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """Get device parameter value.
        
        Args:
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.device_params.get(param_name, default)

    def update_temperature(self, temperature: float) -> None:
        """Update operating temperature.
        
        Args:
            temperature: New temperature (K)
        """
        self.device_params['temperature'] = temperature

        # Update temperature-dependent parameters
        kb = 1.38e-23
        self.thermal_energy = kb * temperature

        if temperature > 400:
            warnings.warn(f"High temperature ({temperature} K) may affect VCMA efficiency")

    def compute_energy_barrier(
        self,
        magnetization: np.ndarray,
        voltage: float = 0.0
    ) -> float:
        """Compute energy barrier for magnetization switching.
        
        Args:
            magnetization: Current magnetization state
            voltage: Applied voltage
            
        Returns:
            Energy barrier (J)
        """
        # Effective anisotropy with voltage
        k_eff = self._compute_effective_anisotropy(voltage)

        # Energy barrier for uniaxial anisotropy
        easy_axis = self.device_params['easy_axis']

        # Angle with easy axis
        cos_theta = abs(np.dot(magnetization, easy_axis))

        # For switching, the relevant barrier is between stable states
        # For uniaxial anisotropy: E = K * V (full barrier between +z and -z states)
        energy_barrier = abs(k_eff) * self.volume

        return max(energy_barrier, 0.0)

    def estimate_switching_time(
        self,
        voltage: float,
        temperature: float = 300.0
    ) -> float:
        """Estimate switching time for given voltage and temperature.
        
        Args:
            voltage: Applied voltage (V)
            temperature: Operating temperature (K)
            
        Returns:
            Estimated switching time (s)
        """
        if abs(voltage) < 1e-6:
            return np.inf

        # Voltage-modified energy barrier
        k_eff = self._compute_effective_anisotropy(voltage)
        energy_barrier = k_eff * self.volume

        if energy_barrier <= 0:
            # Barrier eliminated - essentially instantaneous switching
            return 1e-12  # Picosecond scale

        # Thermally activated switching
        kb = 1.38e-23

        # Attempt frequency (typical ~GHz)
        f0 = 1e9

        # Switching time
        switching_time = (1 / f0) * np.exp(energy_barrier / (kb * temperature))

        return switching_time

    def compute_leakage_current(self, voltage: float) -> float:
        """Compute leakage current through dielectric.
        
        Args:
            voltage: Applied voltage (V)
            
        Returns:
            Leakage current (A)
        """
        if abs(voltage) < 1e-12:
            return 0.0

        # Simple ohmic leakage
        current = voltage / self.leakage_resistance

        # Add tunneling leakage (simplified exponential dependence)
        electric_field = abs(voltage) / self.dielectric_thickness

        # Fowler-Nordheim tunneling (simplified)
        if electric_field > 1e8:  # V/m
            tunneling_factor = np.exp(-3.5e9 / electric_field)  # Simplified
            tunneling_current = 1e-6 * electric_field * tunneling_factor * self.area
            current += tunneling_current

        return current

    def __repr__(self) -> str:
        """String representation of device."""
        return (f"VCMAMRAMDevice(volume={self.volume:.2e} m³, "
                f"Ms={self.saturation_magnetization:.0f} A/m, "
                f"ξ={self.vcma_coefficient:.2e} J/(V⋅m))")
