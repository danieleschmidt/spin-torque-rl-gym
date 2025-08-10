"""Simplified and robust LLGS solver for basic functionality.

This module provides a simplified, fast, and robust implementation of the
Landau-Lifshitz-Gilbert-Slonczewski equation solver optimized for
reinforcement learning environments.
"""

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..utils.performance import PerformanceProfiler, get_optimizer

# Configure logging for physics solver
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SimpleLLGSSolver:
    """Simplified LLGS solver optimized for RL environments."""

    def __init__(
        self,
        method: str = 'euler',
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = 1e-12,
        timeout: float = 1.0  # Maximum solve time in seconds
    ):
        """Initialize simplified LLGS solver.
        
        Args:
            method: Integration method ('euler', 'rk4')
            rtol: Relative tolerance (not used in simplified version)
            atol: Absolute tolerance for convergence
            max_step: Maximum time step size (s)
            timeout: Maximum computation time (s)
        """
        self.method = method.lower()
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.timeout = timeout
        
        # Numerical stability enhancements
        self.min_dt = 1e-16  # Minimum time step
        self.stability_factor = 0.8  # CFL-like stability factor
        self.max_magnetization_change = 0.1  # Max magnitude change per step
        self.use_adaptive_stepping = True  # Enable adaptive time stepping

        # Validate method
        if self.method not in ['euler', 'rk4']:
            warnings.warn(f"Unknown method '{method}', using 'euler'")
            self.method = 'euler'

        # Physical constants
        self.gamma = 2.21e5  # Gyromagnetic ratio (m/(A⋅s))
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

        # Performance tracking
        self.last_solve_time = 0.0
        self.solve_count = 0
        self.timeout_count = 0

        # Performance optimization
        self.optimizer = get_optimizer()
        self.profiler = PerformanceProfiler()

    def solve(
        self,
        m_initial: np.ndarray,
        time_span: Tuple[float, float],
        device_params: Dict[str, Any],
        current_func: Optional[Callable[[float], float]] = None,
        field_func: Optional[Callable[[float], np.ndarray]] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0
    ) -> Dict[str, Any]:
        """Solve LLGS equation with simplified, robust implementation.
        
        Args:
            m_initial: Initial magnetization (normalized)
            time_span: (t_start, t_end) time span
            device_params: Device parameters
            current_func: Current density function I(t)
            field_func: Applied field function H(t)
            thermal_noise: Whether to include thermal fluctuations
            temperature: Temperature for thermal noise (K)
            
        Returns:
            Dictionary with solution data
        """
        import time
        start_time = time.time()

        with self.profiler.time_operation("llgs_solve"):
            try:
                self.solve_count += 1

                # Create cache key for similar problems
                cache_params = {
                    'm_initial_hash': hash(tuple(m_initial.round(6))),  # Round to avoid precision issues
                    'time_span': time_span,
                    'device_hash': self.optimizer.hash_params(device_params),
                    'thermal_noise': thermal_noise,
                    'temperature': round(temperature, 1)
                }

                # Try cached result for identical problems
                cache_key = f"llgs_solve_{self.optimizer.hash_params(cache_params)}"
                cached_result = self.optimizer.cache.get(cache_key)
                if cached_result is not None:
                    self.profiler.increment_counter("cache_hits")
                    return cached_result

                # Input validation and normalization
                m_initial = self._validate_magnetization(m_initial)
                t_start, t_end = time_span

                if t_end <= t_start:
                    return self._create_trivial_solution(m_initial, t_start, t_end)

                # Extract device parameters with defaults
                alpha = device_params.get('damping', 0.01)
                ms = device_params.get('saturation_magnetization', 800e3)
                k_u = device_params.get('uniaxial_anisotropy', 1e6)
                volume = device_params.get('volume', 1e-24)
                easy_axis = device_params.get('easy_axis', np.array([0, 0, 1]))
                polarization = device_params.get('polarization', 0.7)

                # Normalize easy axis
                easy_axis = easy_axis / np.linalg.norm(easy_axis)

                # Time stepping parameters
                dt = min(self.max_step, (t_end - t_start) / 100)
                n_steps = max(10, int((t_end - t_start) / dt))
                dt = (t_end - t_start) / n_steps

                # Initialize arrays
                t = np.linspace(t_start, t_end, n_steps + 1)
                m = np.zeros((n_steps + 1, 3))
                m[0] = m_initial.copy()

                # Solve with timeout protection
                for i in range(n_steps):
                    if time.time() - start_time > self.timeout:
                        self.timeout_count += 1
                        logger.warning(f"Solver timeout after {self.timeout}s at step {i}/{n_steps}")
                        # Return partial solution
                        return {
                            't': t[:i+1],
                            'm': m[:i+1],
                            'success': False,
                            'message': 'Timeout',
                            'solve_time': time.time() - start_time
                        }

                    try:
                        m[i+1] = self._integration_step(
                            m[i], t[i], dt, device_params,
                            current_func, field_func,
                            thermal_noise, temperature
                        )

                        # Ensure normalization
                        m[i+1] = m[i+1] / np.linalg.norm(m[i+1])

                    except Exception as e:
                        logger.warning(f"Integration failed at step {i}: {e}")
                        # Return solution up to failure point
                        return {
                            't': t[:i+1],
                            'm': m[:i+1],
                            'success': False,
                            'message': f'Integration failure: {e}',
                            'solve_time': time.time() - start_time
                        }

                solve_time = time.time() - start_time
                self.last_solve_time = solve_time

                result = {
                    't': t,
                    'm': m,
                    'success': True,
                    'message': 'Integration completed successfully',
                    'solve_time': solve_time,
                    'n_steps': n_steps
                }

                # Cache successful result
                self.optimizer.cache.put(cache_key, result)

                return result

            except Exception as e:
                logger.error(f"LLGS solver failed: {e}")
                return {
                    't': np.array([t_start, t_end]),
                    'm': np.array([m_initial, m_initial]),
                    'success': False,
                    'message': f'Solver error: {e}',
                    'solve_time': time.time() - start_time
                }

    def _validate_magnetization(self, m: np.ndarray) -> np.ndarray:
        """Validate and normalize magnetization vector."""
        if not isinstance(m, np.ndarray) or m.shape != (3,):
            raise ValueError("Magnetization must be a 3D numpy array")

        magnitude = np.linalg.norm(m)
        if magnitude < 1e-12:
            raise ValueError("Magnetization cannot have zero magnitude")

        return m / magnitude

    def _create_trivial_solution(self, m_initial: np.ndarray, t_start: float, t_end: float) -> Dict[str, Any]:
        """Create trivial solution for edge cases."""
        return {
            't': np.array([t_start, t_end]),
            'm': np.array([m_initial, m_initial]),
            'success': True,
            'message': 'Trivial solution (zero time span)',
            'solve_time': 0.0,
            'n_steps': 1
        }

    def _integration_step(
        self,
        m: np.ndarray,
        t: float,
        dt: float,
        device_params: Dict[str, Any],
        current_func: Optional[Callable] = None,
        field_func: Optional[Callable] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0
    ) -> np.ndarray:
        """Perform single integration step."""

        if self.method == 'euler':
            return self._euler_step(m, t, dt, device_params, current_func, field_func, thermal_noise, temperature)
        elif self.method == 'rk4':
            return self._rk4_step(m, t, dt, device_params, current_func, field_func, thermal_noise, temperature)
        else:
            # Fallback to euler
            return self._euler_step(m, t, dt, device_params, current_func, field_func, thermal_noise, temperature)

    def _euler_step(
        self,
        m: np.ndarray,
        t: float,
        dt: float,
        device_params: Dict[str, Any],
        current_func: Optional[Callable] = None,
        field_func: Optional[Callable] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0
    ) -> np.ndarray:
        """Euler integration step."""
        dmdt = self._compute_dmdt(m, t, device_params, current_func, field_func, thermal_noise, temperature)
        return m + dt * dmdt

    def _rk4_step(
        self,
        m: np.ndarray,
        t: float,
        dt: float,
        device_params: Dict[str, Any],
        current_func: Optional[Callable] = None,
        field_func: Optional[Callable] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0
    ) -> np.ndarray:
        """4th order Runge-Kutta integration step."""
        k1 = dt * self._compute_dmdt(m, t, device_params, current_func, field_func, thermal_noise, temperature)
        k2 = dt * self._compute_dmdt(m + k1/2, t + dt/2, device_params, current_func, field_func, thermal_noise, temperature)
        k3 = dt * self._compute_dmdt(m + k2/2, t + dt/2, device_params, current_func, field_func, thermal_noise, temperature)
        k4 = dt * self._compute_dmdt(m + k3, t + dt, device_params, current_func, field_func, thermal_noise, temperature)

        return m + (k1 + 2*k2 + 2*k3 + k4) / 6

    def _compute_dmdt(
        self,
        m: np.ndarray,
        t: float,
        device_params: Dict[str, Any],
        current_func: Optional[Callable] = None,
        field_func: Optional[Callable] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0
    ) -> np.ndarray:
        """Compute dm/dt according to LLGS equation."""

        # Extract parameters
        alpha = device_params.get('damping', 0.01)
        ms = device_params.get('saturation_magnetization', 800e3)
        k_u = device_params.get('uniaxial_anisotropy', 1e6)
        volume = device_params.get('volume', 1e-24)
        easy_axis = device_params.get('easy_axis', np.array([0, 0, 1]))
        polarization = device_params.get('polarization', 0.7)

        # Normalize easy axis
        easy_axis = easy_axis / np.linalg.norm(easy_axis)

        # Effective field
        h_eff = self._compute_effective_field(m, device_params, field_func, t, temperature if thermal_noise else 0)

        # Current-related terms
        if current_func is not None:
            current = current_func(t)
            if abs(current) > 1e-12:
                # Simplified spin-transfer torque term
                # For STT-MRAM: torque proportional to m × (m × p)
                p = easy_axis  # Spin polarization direction
                spin_torque = (polarization * current / (ms * volume)) * np.cross(m, np.cross(m, p))
            else:
                spin_torque = np.zeros(3)
        else:
            spin_torque = np.zeros(3)

        # LLGS equation: dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)] + spin torque
        gamma_eff = self.gamma / (1 + alpha**2)

        precession = np.cross(m, h_eff)
        damping = alpha * np.cross(m, precession)

        dmdt = -gamma_eff * (precession + damping) + spin_torque

        return dmdt

    def _compute_effective_field(
        self,
        m: np.ndarray,
        device_params: Dict[str, Any],
        field_func: Optional[Callable] = None,
        t: float = 0.0,
        temperature: float = 0.0
    ) -> np.ndarray:
        """Compute effective magnetic field."""

        # Extract parameters
        k_u = device_params.get('uniaxial_anisotropy', 1e6)
        ms = device_params.get('saturation_magnetization', 800e3)
        easy_axis = device_params.get('easy_axis', np.array([0, 0, 1]))

        easy_axis = easy_axis / np.linalg.norm(easy_axis)

        # Applied field
        if field_func is not None:
            h_applied = field_func(t)
        else:
            h_applied = np.zeros(3)

        # Anisotropy field
        h_k = (2 * k_u) / (self.mu_0 * ms)
        h_anis = h_k * np.dot(m, easy_axis) * easy_axis

        # Simplified demagnetization field (shape anisotropy)
        # Assume thin film with out-of-plane easy axis
        h_demag = -ms * m[2] * np.array([0, 0, 1])  # Simplified

        # Thermal field (random, simplified)
        if temperature > 0:
            # Thermal field strength
            kb = 1.38e-23
            volume = device_params.get('volume', 1e-24)
            h_thermal_strength = np.sqrt(2 * device_params.get('damping', 0.01) * kb * temperature /
                                       (self.mu_0 * ms * volume * self.gamma))
            h_thermal = h_thermal_strength * np.random.normal(0, 1, 3)
        else:
            h_thermal = np.zeros(3)

        return h_applied + h_anis + h_demag + h_thermal

    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver performance information."""
        return {
            'method': self.method,
            'solve_count': self.solve_count,
            'timeout_count': self.timeout_count,
            'last_solve_time': self.last_solve_time,
            'timeout_rate': self.timeout_count / max(self.solve_count, 1),
            'avg_solve_time': getattr(self, '_total_solve_time', 0) / max(self.solve_count, 1) if hasattr(self, '_total_solve_time') else self.last_solve_time
        }
