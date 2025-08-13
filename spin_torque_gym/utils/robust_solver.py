"""Robust physics solver with comprehensive error handling and validation."""

import logging
import time
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..physics.simple_solver import SimpleLLGSSolver
from .error_handling import (
    ValidationError,
    safe_execute,
)
from .monitoring import MetricsCollector
from .performance import PerformanceProfiler
from .validation import validate_magnetization, validate_parameters

logger = logging.getLogger(__name__)


class RobustLLGSSolver(SimpleLLGSSolver):
    """Robust LLGS solver with enhanced error handling and monitoring."""

    def __init__(
        self,
        method: str = 'euler',
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = 1e-12,
        timeout: float = 2.0,
        max_retries: int = 3,
        fallback_method: str = 'euler',
        enable_monitoring: bool = True,
        enable_validation: bool = True
    ):
        """Initialize robust LLGS solver.
        
        Args:
            method: Integration method ('euler', 'rk4')
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum time step
            timeout: Maximum computation time
            max_retries: Maximum retry attempts on failure
            fallback_method: Fallback integration method
            enable_monitoring: Enable performance monitoring
            enable_validation: Enable input/output validation
        """
        super().__init__(method, rtol, atol, max_step, timeout)

        self.max_retries = max_retries
        self.fallback_method = fallback_method
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation

        # Initialize monitoring and profiling
        if self.enable_monitoring:
            self.metrics = MetricsCollector()
            self.profiler = PerformanceProfiler()

        # Solver statistics
        self.stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'retries_used': 0,
            'fallback_used': 0,
            'average_solve_time': 0.0,
            'validation_errors': 0
        }

        logger.info(f"Initialized RobustLLGSSolver with method={method}, timeout={timeout}s")

    def solve(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        current_func: Optional[Callable] = None,
        field_func: Optional[Callable] = None,
        thermal_noise: bool = False,
        temperature: float = 300.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Solve LLGS equation with robust error handling.
        
        Args:
            m_initial: Initial magnetization vector
            t_span: Time span (t_start, t_end)
            device_params: Device parameters
            current_func: Current function
            field_func: External field function
            thermal_noise: Enable thermal noise
            temperature: Temperature in Kelvin
            **kwargs: Additional arguments
            
        Returns:
            Solution dictionary with robustness metrics
        """
        solve_start = time.time()
        self.stats['total_solves'] += 1

        if self.enable_monitoring:
            self.profiler.start_timer('solve')
            self.metrics.increment('solver_calls')

        try:
            # Input validation
            if self.enable_validation:
                self._validate_inputs(m_initial, t_span, device_params, temperature)

            # Attempt solve with retries
            result = self._solve_with_retries(
                m_initial, t_span, device_params,
                current_func, field_func, thermal_noise, temperature, **kwargs
            )

            # Output validation
            if self.enable_validation and result.get('success', False):
                self._validate_output(result)

            # Update statistics
            if result.get('success', False):
                self.stats['successful_solves'] += 1
            else:
                self.stats['failed_solves'] += 1

            solve_time = time.time() - solve_start
            self._update_timing_stats(solve_time)

            if self.enable_monitoring:
                self.profiler.end_timer('solve')
                self.metrics.record('solve_time', solve_time)
                self.metrics.record('solve_success', int(result.get('success', False)))

            logger.debug(f"Solve completed in {solve_time:.4f}s, success={result.get('success', False)}")
            return result

        except Exception as e:
            self.stats['failed_solves'] += 1

            if self.enable_monitoring:
                self.metrics.increment('solver_errors')
                self.profiler.end_timer('solve')

            logger.error(f"Solver failed with error: {e}")

            # Return safe fallback result
            return self._create_fallback_result(m_initial, t_span, str(e))

    def _validate_inputs(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        temperature: float
    ) -> None:
        """Validate solver inputs."""
        try:
            # Validate magnetization
            validate_magnetization(m_initial)

            # Validate time span
            if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
                raise ValidationError("t_span must be tuple/list of length 2")

            t_start, t_end = t_span
            if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
                raise ValidationError("t_span values must be numeric")

            if t_end <= t_start:
                raise ValidationError("t_end must be greater than t_start")

            if t_end - t_start > 1e-6:  # Maximum simulation time
                warnings.warn("Long simulation time may cause instability")

            # Validate device parameters
            validate_parameters(device_params)

            # Validate temperature
            if not isinstance(temperature, (int, float)) or temperature <= 0:
                raise ValidationError("Temperature must be positive number")

            if temperature > 1000:  # 1000K limit
                warnings.warn("High temperature may cause numerical instability")

        except ValidationError as e:
            self.stats['validation_errors'] += 1
            raise e

    def _validate_output(self, result: Dict[str, Any]) -> None:
        """Validate solver output."""
        if 'm' in result:
            m_array = result['m']
            if isinstance(m_array, np.ndarray) and m_array.ndim >= 2:
                # Validate each magnetization vector
                for i, m_vec in enumerate(m_array):
                    try:
                        validate_magnetization(m_vec)
                    except ValidationError as e:
                        logger.warning(f"Invalid magnetization at step {i}: {e}")
                        # Normalize if possible
                        if np.linalg.norm(m_vec) > 0:
                            result['m'][i] = m_vec / np.linalg.norm(m_vec)

    def _solve_with_retries(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        current_func: Optional[Callable],
        field_func: Optional[Callable],
        thermal_noise: bool,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Attempt solve with retry logic."""
        last_error = None
        original_method = self.method

        for attempt in range(self.max_retries + 1):
            try:
                # Use fallback method on retries
                if attempt > 0:
                    self.method = self.fallback_method
                    self.stats['retries_used'] += 1
                    logger.warning(f"Retry {attempt}/{self.max_retries} using {self.fallback_method}")

                # Attempt solve
                result = safe_execute(
                    super().solve,
                    m_initial, t_span, device_params,
                    current_func, field_func, thermal_noise, temperature,
                    **kwargs
                )

                if result.get('success', False):
                    if attempt > 0:
                        self.stats['fallback_used'] += 1
                        result['used_fallback'] = True
                        result['retry_attempt'] = attempt

                    # Restore original method
                    self.method = original_method
                    return result

            except Exception as e:
                last_error = e
                logger.warning(f"Solve attempt {attempt + 1} failed: {e}")

                # Adjust parameters for retry
                if attempt < self.max_retries:
                    self._adjust_for_retry(attempt)

        # All attempts failed
        self.method = original_method
        error_msg = f"All solve attempts failed. Last error: {last_error}"
        logger.error(error_msg)

        return self._create_fallback_result(m_initial, t_span, error_msg)

    def _adjust_for_retry(self, attempt: int) -> None:
        """Adjust solver parameters for retry."""
        # Reduce step size
        self.max_step = max(self.max_step * 0.5, 1e-15)

        # Increase timeout
        self.timeout = min(self.timeout * 1.5, 10.0)

        # Relax tolerances
        self.atol *= 10
        self.rtol *= 10

        logger.debug(f"Adjusted parameters for retry {attempt + 1}: "
                    f"max_step={self.max_step:.2e}, timeout={self.timeout:.1f}s")

    def _create_fallback_result(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        error_msg: str
    ) -> Dict[str, Any]:
        """Create safe fallback result when solver fails."""
        t_start, t_end = t_span
        n_points = max(2, int((t_end - t_start) / self.max_step))

        # Create minimal trajectory with initial state
        t = np.linspace(t_start, t_end, n_points)
        m = np.tile(m_initial, (n_points, 1))

        return {
            't': t,
            'm': m,
            'success': False,
            'message': f'Fallback result: {error_msg}',
            'solve_time': 0.0,
            'is_fallback': True
        }

    def _update_timing_stats(self, solve_time: float) -> None:
        """Update timing statistics."""
        total_solves = self.stats['total_solves']
        current_avg = self.stats['average_solve_time']

        # Running average
        self.stats['average_solve_time'] = (
            (current_avg * (total_solves - 1) + solve_time) / total_solves
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get solver performance statistics."""
        stats = self.stats.copy()

        if stats['total_solves'] > 0:
            stats['success_rate'] = stats['successful_solves'] / stats['total_solves']
            stats['failure_rate'] = stats['failed_solves'] / stats['total_solves']
            stats['retry_rate'] = stats['retries_used'] / stats['total_solves']
            stats['fallback_rate'] = stats['fallback_used'] / stats['total_solves']
        else:
            stats.update({
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'retry_rate': 0.0,
                'fallback_rate': 0.0
            })

        return stats

    def reset_statistics(self) -> None:
        """Reset solver statistics."""
        self.stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'retries_used': 0,
            'fallback_used': 0,
            'average_solve_time': 0.0,
            'validation_errors': 0
        }

        if self.enable_monitoring:
            self.metrics.reset()

        logger.info("Solver statistics reset")
