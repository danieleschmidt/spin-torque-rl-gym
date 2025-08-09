"""Energy landscape analysis for spintronic devices.

This module provides tools for analyzing the magnetic energy landscape,
finding stable states, computing energy barriers, and generating phase diagrams.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class EnergyLandscape:
    """Magnetic energy landscape analyzer."""

    def __init__(self, device_params: Dict[str, Any]):
        """Initialize energy landscape analyzer.
        
        Args:
            device_params: Device parameters dictionary
        """
        self.device_params = device_params

        # Physical constants
        self.mu_0 = 4 * np.pi * 1e-7  # H/m
        self.k_b = 1.380649e-23  # J/K

        # Extract key parameters
        self.ms = device_params.get('saturation_magnetization', 800e3)
        self.volume = device_params.get('volume', 1e-24)
        self.k_u = device_params.get('uniaxial_anisotropy', 1e6)
        self.easy_axis = device_params.get('easy_axis', np.array([0, 0, 1]))
        self.demag_factors = device_params.get('demag_factors', np.array([0, 0, 1]))

    def compute_energy(
        self,
        magnetization: np.ndarray,
        applied_field: np.ndarray = None,
        current: float = 0.0
    ) -> float:
        """Compute total magnetic energy for given magnetization state.
        
        Args:
            magnetization: Magnetization vector (unit vector)
            applied_field: Applied magnetic field (T)
            current: Applied current density (A/m²)
            
        Returns:
            Total energy (J)
        """
        if applied_field is None:
            applied_field = np.zeros(3)

        m = magnetization / np.linalg.norm(magnetization)  # Ensure unit vector

        # Zeeman energy (interaction with applied field)
        e_zeeman = -self.mu_0 * self.ms * self.volume * np.dot(m, applied_field)

        # Uniaxial anisotropy energy
        cos_theta = np.dot(m, self.easy_axis)
        e_anisotropy = -self.k_u * self.volume * cos_theta**2

        # Demagnetization energy (shape anisotropy)
        e_demag = 0.5 * self.mu_0 * self.ms**2 * self.volume * np.sum(self.demag_factors * m**2)

        # Exchange energy (simplified - constant for single domain)
        a_ex = self.device_params.get('exchange_constant', 20e-12)
        e_exchange = 0.0  # Uniform magnetization assumption

        return e_zeeman + e_anisotropy + e_demag + e_exchange

    def compute_energy_gradient(
        self,
        magnetization: np.ndarray,
        applied_field: np.ndarray = None
    ) -> np.ndarray:
        """Compute energy gradient (effective field) for given state.
        
        Args:
            magnetization: Magnetization vector
            applied_field: Applied magnetic field (T)
            
        Returns:
            Energy gradient vector (effective field in T)
        """
        if applied_field is None:
            applied_field = np.zeros(3)

        m = magnetization / np.linalg.norm(magnetization)

        # Applied field contribution
        h_eff = applied_field.copy()

        # Anisotropy field
        cos_theta = np.dot(m, self.easy_axis)
        h_anis = (2 * self.k_u / (self.mu_0 * self.ms)) * cos_theta * self.easy_axis
        h_eff += h_anis

        # Demagnetization field
        h_demag = -self.ms * self.demag_factors * m
        h_eff += h_demag

        return h_eff

    def find_stable_states(
        self,
        n_trials: int = 100,
        applied_field: np.ndarray = None,
        tolerance: float = 1e-6
    ) -> List[np.ndarray]:
        """Find stable magnetization states by energy minimization.
        
        Args:
            n_trials: Number of random initial conditions to try
            applied_field: Applied magnetic field (T)
            tolerance: Convergence tolerance for unique states
            
        Returns:
            List of stable magnetization states (unit vectors)
        """
        if applied_field is None:
            applied_field = np.zeros(3)

        stable_states = []

        def energy_function(m_spherical):
            """Energy as function of spherical coordinates (theta, phi)."""
            theta, phi = m_spherical
            m = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            return self.compute_energy(m, applied_field)

        for _ in range(n_trials):
            # Random initial state in spherical coordinates
            theta_init = np.random.uniform(0, np.pi)
            phi_init = np.random.uniform(0, 2*np.pi)

            try:
                result = minimize(
                    energy_function,
                    [theta_init, phi_init],
                    method='BFGS',
                    options={'ftol': 1e-12, 'gtol': 1e-12}
                )

                if result.success:
                    theta, phi = result.x
                    m_stable = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])

                    # Check if this is a new stable state
                    is_new = True
                    for existing_state in stable_states:
                        if np.linalg.norm(m_stable - existing_state) < tolerance:
                            is_new = False
                            break

                    if is_new:
                        stable_states.append(m_stable)

            except Exception:
                continue

        # Sort by energy
        if stable_states:
            energies = [self.compute_energy(m, applied_field) for m in stable_states]
            sorted_indices = np.argsort(energies)
            stable_states = [stable_states[i] for i in sorted_indices]

        return stable_states

    def compute_energy_barrier(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        applied_field: np.ndarray = None,
        n_intermediate: int = 50
    ) -> Tuple[float, np.ndarray]:
        """Compute energy barrier between two states using linear interpolation.
        
        Args:
            initial_state: Initial magnetization state
            final_state: Final magnetization state
            applied_field: Applied magnetic field (T)
            n_intermediate: Number of intermediate points
            
        Returns:
            Tuple of (barrier_height, energy_path)
        """
        if applied_field is None:
            applied_field = np.zeros(3)

        # Create path between states
        path_params = np.linspace(0, 1, n_intermediate)
        energy_path = []

        for t in path_params:
            # Linear interpolation in 3D space
            m_interp = (1 - t) * initial_state + t * final_state
            m_interp = m_interp / np.linalg.norm(m_interp)  # Normalize

            energy = self.compute_energy(m_interp, applied_field)
            energy_path.append(energy)

        energy_path = np.array(energy_path)

        # Find barrier height
        initial_energy = energy_path[0]
        final_energy = energy_path[-1]
        max_energy = np.max(energy_path)

        barrier_height = max_energy - min(initial_energy, final_energy)

        return barrier_height, energy_path

    def plot_energy_surface(
        self,
        applied_field: np.ndarray = None,
        theta_range: Tuple[float, float] = (0, np.pi),
        phi_range: Tuple[float, float] = (0, 2*np.pi),
        resolution: int = 50,
        save_path: Optional[str] = None
    ) -> None:
        """Plot 2D energy surface in spherical coordinates.
        
        Args:
            applied_field: Applied magnetic field (T)
            theta_range: Range of polar angle (radians)
            phi_range: Range of azimuthal angle (radians)
            resolution: Grid resolution
            save_path: Optional path to save figure
        """
        if applied_field is None:
            applied_field = np.zeros(3)

        theta_grid = np.linspace(theta_range[0], theta_range[1], resolution)
        phi_grid = np.linspace(phi_range[0], phi_range[1], resolution)

        Theta, Phi = np.meshgrid(theta_grid, phi_grid)
        Energy = np.zeros_like(Theta)

        for i in range(resolution):
            for j in range(resolution):
                m = np.array([
                    np.sin(Theta[i, j]) * np.cos(Phi[i, j]),
                    np.sin(Theta[i, j]) * np.sin(Phi[i, j]),
                    np.cos(Theta[i, j])
                ])
                Energy[i, j] = self.compute_energy(m, applied_field)

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(Phi, Theta, Energy, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Energy (J)')
        plt.xlabel('Azimuthal angle φ (rad)')
        plt.ylabel('Polar angle θ (rad)')
        plt.title('Magnetic Energy Landscape')

        # Mark stable states
        stable_states = self.find_stable_states(applied_field=applied_field)
        for state in stable_states:
            theta = np.arccos(state[2])
            phi = np.arctan2(state[1], state[0])
            if phi < 0:
                phi += 2 * np.pi
            plt.plot(phi, theta, 'ro', markersize=8, label='Stable state')

        if stable_states:
            plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_phase_diagram(
        self,
        current_range: Tuple[float, float],
        field_range: Tuple[float, float],
        resolution: int = 50,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Generate switching phase diagram.
        
        Args:
            current_range: (I_min, I_max) current density range (A/m²)
            field_range: (H_min, H_max) field range (T)
            resolution: Grid resolution
            save_path: Optional path to save figure
            
        Returns:
            Dictionary with phase diagram data
        """
        currents = np.linspace(current_range[0], current_range[1], resolution)
        fields = np.linspace(field_range[0], field_range[1], resolution)

        Current, Field = np.meshgrid(currents, fields)
        SwitchingField = np.zeros_like(Current)

        # Simple switching criterion (placeholder for actual dynamics)
        for i in range(resolution):
            for j in range(resolution):
                # Critical field for switching with given current
                # This is a simplified model - real dynamics would require time evolution
                I = Current[i, j]
                H = Field[i, j]

                # Effective switching field considering spin torque assistance
                beta = self.device_params.get('polarization', 0.7) * 2.21e5 / (2 * self.ms * self.volume)
                h_stt = abs(beta * I)

                # Critical field for switching
                h_k = 2 * self.k_u / (self.mu_0 * self.ms)  # Anisotropy field
                h_critical = h_k - h_stt

                SwitchingField[i, j] = 1 if abs(H) > h_critical else 0

        plt.figure(figsize=(10, 8))
        plt.contourf(Current/1e6, Field*1e3, SwitchingField, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.7)
        plt.colorbar(label='Switching probability')
        plt.xlabel('Current density (MA/cm²)')
        plt.ylabel('Applied field (mT)')
        plt.title('Switching Phase Diagram')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        return {
            'currents': currents,
            'fields': fields,
            'switching_probability': SwitchingField
        }

    def compute_thermal_stability_factor(self, temperature: float = 300.0) -> float:
        """Compute thermal stability factor Δ = E_barrier / k_B T.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Thermal stability factor
        """
        # Energy barrier height (simplified as anisotropy energy)
        energy_barrier = self.k_u * self.volume

        if temperature <= 0:
            return float('inf')

        return energy_barrier / (self.k_b * temperature)
