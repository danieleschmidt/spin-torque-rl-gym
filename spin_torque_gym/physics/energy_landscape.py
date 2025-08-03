"""Energy landscape analysis for spintronic devices.

This module provides tools for analyzing the magnetic energy landscape,
finding stable states, computing energy barriers, and generating phase diagrams.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            \"\"\"Energy as function of spherical coordinates (theta, phi).\"\"\"\n            theta, phi = m_spherical\n            m = np.array([\n                np.sin(theta) * np.cos(phi),\n                np.sin(theta) * np.sin(phi),\n                np.cos(theta)\n            ])\n            return self.compute_energy(m, applied_field)\n        \n        for _ in range(n_trials):\n            # Random initial state in spherical coordinates\n            theta_init = np.random.uniform(0, np.pi)\n            phi_init = np.random.uniform(0, 2*np.pi)\n            \n            try:\n                result = minimize(\n                    energy_function,\n                    [theta_init, phi_init],\n                    method='BFGS',\n                    options={'ftol': 1e-12, 'gtol': 1e-12}\n                )\n                \n                if result.success:\n                    theta, phi = result.x\n                    m_stable = np.array([\n                        np.sin(theta) * np.cos(phi),\n                        np.sin(theta) * np.sin(phi),\n                        np.cos(theta)\n                    ])\n                    \n                    # Check if this is a new stable state\n                    is_new = True\n                    for existing_state in stable_states:\n                        if np.linalg.norm(m_stable - existing_state) < tolerance:\n                            is_new = False\n                            break\n                    \n                    if is_new:\n                        stable_states.append(m_stable)\n                        \n            except Exception:\n                continue\n        \n        # Sort by energy\n        if stable_states:\n            energies = [self.compute_energy(m, applied_field) for m in stable_states]\n            sorted_indices = np.argsort(energies)\n            stable_states = [stable_states[i] for i in sorted_indices]\n        \n        return stable_states\n    \n    def compute_energy_barrier(\n        self,\n        initial_state: np.ndarray,\n        final_state: np.ndarray,\n        applied_field: np.ndarray = None,\n        n_intermediate: int = 50\n    ) -> Tuple[float, np.ndarray]:\n        \"\"\"Compute energy barrier between two states using linear interpolation.\n        \n        Args:\n            initial_state: Initial magnetization state\n            final_state: Final magnetization state\n            applied_field: Applied magnetic field (T)\n            n_intermediate: Number of intermediate points\n            \n        Returns:\n            Tuple of (barrier_height, energy_path)\n        \"\"\"\n        if applied_field is None:\n            applied_field = np.zeros(3)\n        \n        # Create path between states\n        path_params = np.linspace(0, 1, n_intermediate)\n        energy_path = []\n        \n        for t in path_params:\n            # Linear interpolation in 3D space\n            m_interp = (1 - t) * initial_state + t * final_state\n            m_interp = m_interp / np.linalg.norm(m_interp)  # Normalize\n            \n            energy = self.compute_energy(m_interp, applied_field)\n            energy_path.append(energy)\n        \n        energy_path = np.array(energy_path)\n        \n        # Find barrier height\n        initial_energy = energy_path[0]\n        final_energy = energy_path[-1]\n        max_energy = np.max(energy_path)\n        \n        barrier_height = max_energy - min(initial_energy, final_energy)\n        \n        return barrier_height, energy_path\n    \n    def minimum_energy_path(\n        self,\n        initial_state: np.ndarray,\n        final_state: np.ndarray,\n        applied_field: np.ndarray = None,\n        n_images: int = 20,\n        max_iterations: int = 100\n    ) -> Tuple[np.ndarray, np.ndarray]:\n        \"\"\"Find minimum energy path using nudged elastic band method.\n        \n        Args:\n            initial_state: Initial magnetization state\n            final_state: Final magnetization state\n            applied_field: Applied magnetic field (T)\n            n_images: Number of images in the band\n            max_iterations: Maximum optimization iterations\n            \n        Returns:\n            Tuple of (path_magnetizations, path_energies)\n        \"\"\"\n        if applied_field is None:\n            applied_field = np.zeros(3)\n        \n        # Initialize band with linear interpolation\n        band = []\n        for i in range(n_images):\n            t = i / (n_images - 1)\n            m_i = (1 - t) * initial_state + t * final_state\n            m_i = m_i / np.linalg.norm(m_i)\n            band.append(m_i)\n        \n        band = np.array(band)\n        \n        # NEB optimization (simplified)\n        spring_constant = 1.0\n        step_size = 0.01\n        \n        for iteration in range(max_iterations):\n            forces = np.zeros_like(band)\n            \n            for i in range(1, n_images - 1):  # Skip endpoints\n                # Compute true force (negative energy gradient)\n                h_eff = self.compute_energy_gradient(band[i], applied_field)\n                true_force = h_eff\n                \n                # Compute tangent\n                if i < n_images - 1:\n                    tangent = band[i+1] - band[i]\n                    tangent = tangent / np.linalg.norm(tangent)\n                else:\n                    tangent = band[i] - band[i-1]\n                    tangent = tangent / np.linalg.norm(tangent)\n                \n                # Remove parallel component of true force\n                parallel_component = np.dot(true_force, tangent) * tangent\n                perpendicular_force = true_force - parallel_component\n                \n                # Spring force along tangent\n                spring_force_left = spring_constant * (band[i-1] - band[i])\n                spring_force_right = spring_constant * (band[i+1] - band[i])\n                spring_force = spring_force_left + spring_force_right\n                spring_component = np.dot(spring_force, tangent) * tangent\n                \n                # Total NEB force\n                forces[i] = perpendicular_force + spring_component\n            \n            # Update band\n            band[1:-1] += step_size * forces[1:-1]\n            \n            # Renormalize\n            for i in range(n_images):\n                band[i] = band[i] / np.linalg.norm(band[i])\n        \n        # Compute energies along path\n        path_energies = np.array([self.compute_energy(m, applied_field) for m in band])\n        \n        return band, path_energies\n    \n    def plot_energy_surface(\n        self,\n        applied_field: np.ndarray = None,\n        theta_range: Tuple[float, float] = (0, np.pi),\n        phi_range: Tuple[float, float] = (0, 2*np.pi),\n        resolution: int = 50,\n        save_path: Optional[str] = None\n    ) -> None:\n        \"\"\"Plot 2D energy surface in spherical coordinates.\n        \n        Args:\n            applied_field: Applied magnetic field (T)\n            theta_range: Range of polar angle (radians)\n            phi_range: Range of azimuthal angle (radians)\n            resolution: Grid resolution\n            save_path: Optional path to save figure\n        \"\"\"\n        if applied_field is None:\n            applied_field = np.zeros(3)\n        \n        theta_grid = np.linspace(theta_range[0], theta_range[1], resolution)\n        phi_grid = np.linspace(phi_range[0], phi_range[1], resolution)\n        \n        Theta, Phi = np.meshgrid(theta_grid, phi_grid)\n        Energy = np.zeros_like(Theta)\n        \n        for i in range(resolution):\n            for j in range(resolution):\n                m = np.array([\n                    np.sin(Theta[i, j]) * np.cos(Phi[i, j]),\n                    np.sin(Theta[i, j]) * np.sin(Phi[i, j]),\n                    np.cos(Theta[i, j])\n                ])\n                Energy[i, j] = self.compute_energy(m, applied_field)\n        \n        plt.figure(figsize=(10, 8))\n        contour = plt.contourf(Phi, Theta, Energy, levels=50, cmap='viridis')\n        plt.colorbar(contour, label='Energy (J)')\n        plt.xlabel('Azimuthal angle φ (rad)')\n        plt.ylabel('Polar angle θ (rad)')\n        plt.title('Magnetic Energy Landscape')\n        \n        # Mark stable states\n        stable_states = self.find_stable_states(applied_field=applied_field)\n        for state in stable_states:\n            theta = np.arccos(state[2])\n            phi = np.arctan2(state[1], state[0])\n            if phi < 0:\n                phi += 2 * np.pi\n            plt.plot(phi, theta, 'ro', markersize=8, label='Stable state')\n        \n        if stable_states:\n            plt.legend()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        else:\n            plt.show()\n    \n    def generate_phase_diagram(\n        self,\n        current_range: Tuple[float, float],\n        field_range: Tuple[float, float],\n        resolution: int = 50,\n        save_path: Optional[str] = None\n    ) -> Dict[str, np.ndarray]:\n        \"\"\"Generate switching phase diagram.\n        \n        Args:\n            current_range: (I_min, I_max) current density range (A/m²)\n            field_range: (H_min, H_max) field range (T)\n            resolution: Grid resolution\n            save_path: Optional path to save figure\n            \n        Returns:\n            Dictionary with phase diagram data\n        \"\"\"\n        currents = np.linspace(current_range[0], current_range[1], resolution)\n        fields = np.linspace(field_range[0], field_range[1], resolution)\n        \n        Current, Field = np.meshgrid(currents, fields)\n        SwitchingField = np.zeros_like(Current)\n        \n        # Simple switching criterion (placeholder for actual dynamics)\n        for i in range(resolution):\n            for j in range(resolution):\n                # Critical field for switching with given current\n                # This is a simplified model - real dynamics would require time evolution\n                I = Current[i, j]\n                H = Field[i, j]\n                \n                # Effective switching field considering spin torque assistance\n                beta = self.device_params.get('polarization', 0.7) * 2.21e5 / (2 * self.ms * self.volume)\n                h_stt = abs(beta * I)\n                \n                # Critical field for switching\n                h_k = 2 * self.k_u / (self.mu_0 * self.ms)  # Anisotropy field\n                h_critical = h_k - h_stt\n                \n                SwitchingField[i, j] = 1 if abs(H) > h_critical else 0\n        \n        plt.figure(figsize=(10, 8))\n        plt.contourf(Current/1e6, Field*1e3, SwitchingField, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.7)\n        plt.colorbar(label='Switching probability')\n        plt.xlabel('Current density (MA/cm²)')\n        plt.ylabel('Applied field (mT)')\n        plt.title('Switching Phase Diagram')\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        else:\n            plt.show()\n        \n        return {\n            'currents': currents,\n            'fields': fields,\n            'switching_probability': SwitchingField\n        }\n    \n    def compute_thermal_stability_factor(self, temperature: float = 300.0) -> float:\n        \"\"\"Compute thermal stability factor Δ = E_barrier / k_B T.\n        \n        Args:\n            temperature: Temperature in Kelvin\n            \n        Returns:\n            Thermal stability factor\n        \"\"\"\n        # Energy barrier height (simplified as anisotropy energy)\n        energy_barrier = self.k_u * self.volume\n        \n        if temperature <= 0:\n            return float('inf')\n        \n        return energy_barrier / (self.k_b * temperature)