"""Landau-Lifshitz-Gilbert-Slonczewski equation solver for magnetization dynamics.

This module implements numerical integration of the LLGS equation:
dm/dt = -γ(m × H_eff) + α(m × dm/dt) + γ_STT * τ_STT + γ_FL * τ_FL

Where:
- m: magnetization vector (unit vector)
- H_eff: effective magnetic field
- α: Gilbert damping parameter
- τ_STT: spin-transfer torque term
- τ_FL: field-like torque term
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


class LLGSSolver:
    """Numerical solver for the Landau-Lifshitz-Gilbert-Slonczewski equation."""

    def __init__(
        self,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = 1e-12,
        gamma: float = 2.21e5,  # Gyromagnetic ratio (m/A·s)
    ):
        """Initialize LLGS solver.
        
        Args:
            method: Integration method ('RK45', 'DOP853', 'Radau', 'BDF', 'LSODA')
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration  
            max_step: Maximum integration step size (seconds)
            gamma: Gyromagnetic ratio (m/A·s)
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.gamma = gamma

        # Physical constants
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        self.k_b = 1.380649e-23  # Boltzmann constant (J/K)

    def solve(
        self,
        m_initial: np.ndarray,
        time_span: Tuple[float, float],
        device_params: Dict[str, Any],
        current_func: Callable[[float], float],
        field_func: Optional[Callable[[float], np.ndarray]] = None,
        thermal_noise: bool = True,
        temperature: float = 300.0,
    ) -> Dict[str, np.ndarray]:
        """Solve LLGS equation for given time span and parameters.
        
        Args:
            m_initial: Initial magnetization vector (3D unit vector)
            time_span: (t_start, t_end) integration time span
            device_params: Device parameters dictionary
            current_func: Function returning current density (A/m²) vs time
            field_func: Function returning applied field (T) vs time 
            thermal_noise: Whether to include thermal fluctuations
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary containing solution arrays: 't', 'm', 'energy', 'torques'
        """
        # Normalize initial magnetization
        m_initial = m_initial / np.linalg.norm(m_initial)

        # Extract device parameters
        alpha = device_params.get('damping', 0.01)
        ms = device_params.get('saturation_magnetization', 800e3)  # A/m
        volume = device_params.get('volume', 1e-24)  # m³
        polarization = device_params.get('polarization', 0.7)

        # Thermal noise strength
        thermal_strength = 0.0
        if thermal_noise:
            thermal_strength = np.sqrt(
                2 * alpha * self.k_b * temperature /
                (self.gamma * self.mu_0 * ms * volume)
            )

        def llgs_rhs(t: float, y: np.ndarray) -> np.ndarray:
            """Right-hand side of LLGS equation."""
            m = y[:3]

            # Ensure unit magnetization
            m_norm = np.linalg.norm(m)
            if m_norm > 1e-12:
                m = m / m_norm
            else:
                m = np.array([0, 0, 1])  # Default to +z

            # Applied current and field
            current = current_func(t)
            h_applied = field_func(t) if field_func else np.zeros(3)

            # Effective field calculation
            h_eff = self._compute_effective_field(m, h_applied, device_params)

            # Thermal field
            if thermal_noise:
                h_thermal = thermal_strength * np.random.normal(0, 1, 3)
                h_eff += h_thermal

            # Torque terms
            tau_stt, tau_fl = self._compute_spin_torques(
                m, current, polarization, volume, ms
            )

            # LLGS equation
            m_cross_h = np.cross(m, h_eff)
            dm_dt = -self.gamma * m_cross_h
            dm_dt += alpha * np.cross(m, dm_dt)  # Gilbert damping
            dm_dt += tau_stt + tau_fl  # Spin torque terms

            return dm_dt

        # Solve ODE
        try:
            sol = solve_ivp(
                llgs_rhs,
                time_span,
                m_initial,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
                dense_output=True
            )

            if not sol.success:
                warnings.warn(f"Integration failed: {sol.message}")

        except Exception as e:
            raise RuntimeError(f"LLGS integration failed: {e}")

        # Extract solution
        t_eval = sol.t
        m_traj = sol.y.T  # Shape: (n_steps, 3)

        # Normalize trajectory
        for i in range(len(m_traj)):
            m_traj[i] = m_traj[i] / np.linalg.norm(m_traj[i])

        # Compute energy and torques along trajectory
        energy_traj = []
        torque_traj = []

        for i, t in enumerate(t_eval):
            m = m_traj[i]
            current = current_func(t)
            h_applied = field_func(t) if field_func else np.zeros(3)

            # Energy calculation
            energy = self._compute_energy(m, h_applied, device_params)
            energy_traj.append(energy)

            # Torque calculation
            tau_stt, tau_fl = self._compute_spin_torques(
                m, current, polarization, volume, ms
            )
            torque_traj.append(np.linalg.norm(tau_stt) + np.linalg.norm(tau_fl))

        return {
            't': t_eval,
            'm': m_traj,
            'energy': np.array(energy_traj),
            'torques': np.array(torque_traj),
            'success': sol.success
        }

    def _compute_effective_field(
        self,
        m: np.ndarray,
        h_applied: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Compute total effective magnetic field."""
        h_eff = h_applied.copy()

        # Uniaxial anisotropy field
        k_u = params.get('uniaxial_anisotropy', 1e6)  # J/m³
        ms = params.get('saturation_magnetization', 800e3)
        easy_axis = params.get('easy_axis', np.array([0, 0, 1]))

        h_anis = (2 * k_u / (self.mu_0 * ms)) * np.dot(m, easy_axis) * easy_axis
        h_eff += h_anis

        # Demagnetization field (shape anisotropy)
        demag_factors = params.get('demag_factors', np.array([0, 0, 1]))
        h_demag = -ms * demag_factors * m
        h_eff += h_demag

        # Exchange field (simplified)
        a_ex = params.get('exchange_constant', 20e-12)  # J/m
        if a_ex > 0:
            # Simplified exchange: proportional to curvature
            h_exchange = (2 * a_ex / (self.mu_0 * ms)) * 0.1 * m  # Placeholder
            h_eff += h_exchange

        return h_eff

    def _compute_spin_torques(
        self,
        m: np.ndarray,
        current: float,
        polarization: float,
        volume: float,
        ms: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spin-transfer torque terms."""
        if abs(current) < 1e-12:
            return np.zeros(3), np.zeros(3)

        # Polarization direction (simplified - along z)
        p_hat = np.array([0, 0, 1])

        # STT efficiency parameters
        beta = polarization * self.gamma / (2 * ms * volume)
        beta_prime = 0.1 * beta  # Field-like term is typically smaller

        # Slonczewski torque terms
        m_cross_p = np.cross(m, p_hat)
        tau_stt = beta * current * np.cross(m, m_cross_p)
        tau_fl = beta_prime * current * m_cross_p

        return tau_stt, tau_fl

    def _compute_energy(
        self,
        m: np.ndarray,
        h_applied: np.ndarray,
        params: Dict[str, Any]
    ) -> float:
        """Compute magnetic energy of the system."""
        ms = params.get('saturation_magnetization', 800e3)
        volume = params.get('volume', 1e-24)

        # Zeeman energy
        e_zeeman = -self.mu_0 * ms * volume * np.dot(m, h_applied)

        # Uniaxial anisotropy energy
        k_u = params.get('uniaxial_anisotropy', 1e6)
        easy_axis = params.get('easy_axis', np.array([0, 0, 1]))
        cos_theta = np.dot(m, easy_axis)
        e_anis = -k_u * volume * cos_theta**2

        # Demagnetization energy
        demag_factors = params.get('demag_factors', np.array([0, 0, 1]))
        e_demag = 0.5 * self.mu_0 * ms**2 * volume * np.sum(demag_factors * m**2)

        return e_zeeman + e_anis + e_demag

    def find_stable_states(
        self,
        device_params: Dict[str, Any],
        n_trials: int = 100,
        threshold: float = 1e-6
    ) -> np.ndarray:
        """Find stable magnetization states by energy minimization."""
        stable_states = []

        for _ in range(n_trials):
            # Random initial state
            m_init = np.random.normal(0, 1, 3)
            m_init = m_init / np.linalg.norm(m_init)

            # Relax to equilibrium (no current, no applied field)
            try:
                result = self.solve(
                    m_init,
                    (0, 10e-9),  # 10 ns relaxation
                    device_params,
                    lambda t: 0.0,  # No current
                    lambda t: np.zeros(3),  # No field
                    thermal_noise=False,
                )

                if result['success']:
                    m_final = result['m'][-1]

                    # Check if this is a new stable state
                    is_new = True
                    for existing_state in stable_states:
                        if np.linalg.norm(m_final - existing_state) < threshold:
                            is_new = False
                            break

                    if is_new:
                        stable_states.append(m_final)

            except RuntimeError:
                continue

        return np.array(stable_states) if stable_states else np.array([[0, 0, 1]])
