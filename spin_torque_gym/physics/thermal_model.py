"""Thermal fluctuation models for spintronic devices.

This module implements thermal noise models for magnetization dynamics,
including Brown's thermal fluctuation model and correlated noise generation.
"""

import numpy as np
from typing import Optional, Tuple
import scipy.stats as stats


class ThermalFluctuations:
    """Thermal fluctuation model for magnetization dynamics."""
    
    def __init__(
        self,
        temperature: float = 300.0,
        correlation_time: float = 1e-12,
        seed: Optional[int] = None
    ):
        """Initialize thermal fluctuation model.
        
        Args:
            temperature: Temperature in Kelvin
            correlation_time: Correlation time of thermal fluctuations (s)
            seed: Random number generator seed for reproducibility
        """
        self.temperature = temperature
        self.correlation_time = correlation_time
        
        # Physical constants
        self.k_b = 1.380649e-23  # Boltzmann constant (J/K)
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Internal state for correlated noise
        self._previous_noise = np.zeros(3)
        self._correlation_factor = 0.0
    
    def set_temperature(self, temperature: float) -> None:
        """Update temperature."""
        self.temperature = temperature
    
    def compute_noise_strength(
        self,
        damping: float,
        saturation_magnetization: float,
        volume: float,
        gamma: float = 2.21e5
    ) -> float:
        """Compute RMS strength of thermal noise field.
        
        Args:
            damping: Gilbert damping parameter
            saturation_magnetization: Saturation magnetization (A/m)
            volume: Device volume (m³)
            gamma: Gyromagnetic ratio (m/A·s)
            
        Returns:
            RMS thermal field strength (T)
        """
        if self.temperature <= 0:
            return 0.0
        
        # Brown's thermal field strength
        variance = (
            2 * damping * self.k_b * self.temperature /
            (gamma * self.mu_0 * saturation_magnetization * volume)
        )
        
        return np.sqrt(variance)
    
    def generate_thermal_field(
        self,
        damping: float,
        saturation_magnetization: float,
        volume: float,
        dt: float,
        gamma: float = 2.21e5,
        correlated: bool = True
    ) -> np.ndarray:
        """Generate thermal field vector for current timestep.
        
        Args:
            damping: Gilbert damping parameter
            saturation_magnetization: Saturation magnetization (A/m)
            volume: Device volume (m³)
            dt: Time step (s)
            gamma: Gyromagnetic ratio (m/A·s)
            correlated: Whether to include temporal correlation
            
        Returns:
            3D thermal field vector (T)
        """
        noise_strength = self.compute_noise_strength(
            damping, saturation_magnetization, volume, gamma
        )
        
        if noise_strength == 0:
            return np.zeros(3)
        
        if correlated and self.correlation_time > 0:
            return self._generate_correlated_noise(noise_strength, dt)
        else:
            return self._generate_white_noise(noise_strength)
    
    def _generate_white_noise(self, strength: float) -> np.ndarray:
        """Generate uncorrelated white noise."""
        return strength * self.rng.normal(0, 1, 3)
    
    def _generate_correlated_noise(
        self, 
        strength: float, 
        dt: float
    ) -> np.ndarray:
        """Generate temporally correlated noise using Ornstein-Uhlenbeck process."""
        # Correlation decay
        if self.correlation_time > 0:
            decay = np.exp(-dt / self.correlation_time)
        else:
            decay = 0.0
        
        # Generate new uncorrelated component
        white_noise = self.rng.normal(0, 1, 3)
        
        # Combine with previous noise for correlation
        correlated_noise = (
            decay * self._previous_noise + 
            np.sqrt(1 - decay**2) * white_noise
        )
        
        # Update previous noise state
        self._previous_noise = correlated_noise
        
        return strength * correlated_noise
    
    def compute_thermal_barrier(
        self,
        anisotropy_constant: float,
        volume: float
    ) -> float:
        """Compute thermal stability factor Δ = K_u V / k_B T.
        
        Args:
            anisotropy_constant: Uniaxial anisotropy constant (J/m³)
            volume: Device volume (m³)
            
        Returns:
            Thermal stability factor (dimensionless)
        """
        if self.temperature <= 0:
            return float('inf')
        
        return anisotropy_constant * volume / (self.k_b * self.temperature)
    
    def compute_switching_probability(
        self,
        energy_barrier: float,
        attempt_frequency: float = 1e9,
        measurement_time: float = 1e-9
    ) -> float:
        """Compute thermal switching probability using Néel-Brown model.
        
        Args:
            energy_barrier: Energy barrier for switching (J)
            attempt_frequency: Attempt frequency (Hz)
            measurement_time: Measurement time scale (s)
            
        Returns:
            Switching probability
        """
        if self.temperature <= 0:
            return 0.0
        
        # Thermal activation rate
        rate = attempt_frequency * np.exp(-energy_barrier / (self.k_b * self.temperature))
        
        # Probability over measurement time
        probability = 1 - np.exp(-rate * measurement_time)
        
        return min(probability, 1.0)
    
    def sample_switching_time(
        self,
        energy_barrier: float,
        attempt_frequency: float = 1e9
    ) -> float:
        """Sample thermal switching time from exponential distribution.
        
        Args:
            energy_barrier: Energy barrier for switching (J)
            attempt_frequency: Attempt frequency (Hz)
            
        Returns:
            Switching time (s)
        """
        if self.temperature <= 0:
            return float('inf')
        
        rate = attempt_frequency * np.exp(-energy_barrier / (self.k_b * self.temperature))
        
        if rate <= 0:
            return float('inf')
        
        return self.rng.exponential(1.0 / rate)
    
    def compute_retention_time(
        self,
        energy_barrier: float,
        failure_rate: float = 1e-9,
        attempt_frequency: float = 1e9
    ) -> float:
        """Compute data retention time for given failure rate.
        
        Args:
            energy_barrier: Energy barrier for switching (J)
            failure_rate: Acceptable failure rate
            attempt_frequency: Attempt frequency (Hz)
            
        Returns:
            Retention time (s)
        """
        if self.temperature <= 0 or failure_rate <= 0:
            return float('inf')
        
        # Time for failure_rate probability of switching
        thermal_factor = energy_barrier / (self.k_b * self.temperature)
        retention_time = -np.log(failure_rate) / (attempt_frequency * np.exp(-thermal_factor))
        
        return retention_time
    
    def analyze_thermal_stability(
        self,
        device_params: dict,
        time_scale: float = 10.0  # years
    ) -> dict:
        """Analyze thermal stability characteristics of device.
        
        Args:
            device_params: Device parameter dictionary
            time_scale: Analysis time scale (years)
            
        Returns:
            Dictionary with thermal stability analysis
        """
        volume = device_params.get('volume', 1e-24)  # m³
        k_u = device_params.get('uniaxial_anisotropy', 1e6)  # J/m³
        
        # Energy barrier
        energy_barrier = k_u * volume
        
        # Thermal stability factor
        delta = self.compute_thermal_barrier(k_u, volume)
        
        # Switching probability and retention
        time_seconds = time_scale * 365.25 * 24 * 3600  # Convert years to seconds
        switch_prob = self.compute_switching_probability(
            energy_barrier, measurement_time=time_seconds
        )
        retention_time = self.compute_retention_time(energy_barrier) / (365.25 * 24 * 3600)  # years
        
        return {
            'thermal_stability_factor': delta,
            'energy_barrier_J': energy_barrier,
            'energy_barrier_kT': energy_barrier / (self.k_b * self.temperature),
            'switching_probability': switch_prob,
            'retention_time_years': retention_time,
            'is_thermally_stable': delta > 40,  # Common stability criterion
            'temperature_K': self.temperature
        }
    
    def generate_temperature_sweep(
        self,
        temp_range: Tuple[float, float],
        device_params: dict,
        n_points: int = 100
    ) -> dict:
        """Generate thermal properties vs temperature.
        
        Args:
            temp_range: (T_min, T_max) temperature range (K)
            device_params: Device parameters
            n_points: Number of temperature points
            
        Returns:
            Dictionary with temperature-dependent properties
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], n_points)
        original_temp = self.temperature
        
        results = {
            'temperature': temperatures,
            'thermal_stability_factor': [],
            'switching_probability': [],
            'retention_time': [],
            'noise_strength': []
        }
        
        volume = device_params.get('volume', 1e-24)
        k_u = device_params.get('uniaxial_anisotropy', 1e6)
        damping = device_params.get('damping', 0.01)
        ms = device_params.get('saturation_magnetization', 800e3)
        
        for temp in temperatures:
            self.set_temperature(temp)
            
            # Thermal stability factor
            delta = self.compute_thermal_barrier(k_u, volume)
            results['thermal_stability_factor'].append(delta)
            
            # Switching probability (1 year)
            energy_barrier = k_u * volume
            switch_prob = self.compute_switching_probability(
                energy_barrier, measurement_time=365.25*24*3600
            )
            results['switching_probability'].append(switch_prob)
            
            # Retention time
            retention = self.compute_retention_time(energy_barrier)
            results['retention_time'].append(retention / (365.25*24*3600))  # years
            
            # Noise strength
            noise = self.compute_noise_strength(damping, ms, volume)
            results['noise_strength'].append(noise)
        
        # Restore original temperature
        self.set_temperature(original_temp)
        
        # Convert lists to arrays
        for key in ['thermal_stability_factor', 'switching_probability', 
                   'retention_time', 'noise_strength']:
            results[key] = np.array(results[key])
        
        return results