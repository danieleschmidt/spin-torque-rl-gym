"""Comprehensive error handling utilities for Spin Torque RL-Gym.

This module provides robust error handling, validation, and recovery mechanisms
for scientific computing applications.
"""

import functools
import logging
import sys
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class SpinTorqueError(Exception):
    """Base exception class for Spin Torque RL-Gym."""

    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.original_traceback = traceback.format_exc()


class PhysicsError(SpinTorqueError):
    """Exception raised for physics simulation errors."""
    pass


class DeviceError(SpinTorqueError):
    """Exception raised for device model errors."""
    pass


class EnvironmentError(SpinTorqueError):
    """Exception raised for RL environment errors."""
    pass


class ValidationError(SpinTorqueError):
    """Exception raised for parameter validation errors."""
    pass


class NumericalError(SpinTorqueError):
    """Exception raised for numerical computation errors."""
    pass


class ConfigurationError(SpinTorqueError):
    """Exception raised for configuration errors."""
    pass


def robust_computation(
    max_retries: int = 3,
    fallback_value: Optional[Any] = None,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 1.0
) -> Callable[[F], F]:
    """Decorator for robust scientific computations with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        fallback_value: Value to return if all retries fail
        exceptions: Tuple of exceptions to catch and retry
        backoff_factor: Exponential backoff multiplier for retries
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Validate result
                    if hasattr(result, '__len__') and len(result) == 0:
                        raise NumericalError("Empty result returned")

                    # Check for NaN or inf values
                    if _contains_invalid_values(result):
                        raise NumericalError("Result contains NaN or inf values")

                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logging.warning(
                            f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {wait_time:.2f}s. Error: {e}"
                        )

                        # Simple sleep substitute (busy wait)
                        import time
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")

            # All retries failed
            if fallback_value is not None:
                logging.warning(f"Using fallback value for {func.__name__}")
                return fallback_value

            # Re-raise the last exception
            raise NumericalError(
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                error_code="MAX_RETRIES_EXCEEDED",
                context={'attempts': max_retries + 1, 'last_error': str(last_exception)}
            ) from last_exception

        return wrapper
    return decorator


def _contains_invalid_values(obj: Any) -> bool:
    """Check if object contains NaN or infinite values."""
    try:
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            # Check iterable objects
            for item in obj:
                if _contains_invalid_values(item):
                    return True
        else:
            # Check scalar values
            if isinstance(obj, (int, float, complex)):
                import math
                if isinstance(obj, complex):
                    return math.isnan(obj.real) or math.isnan(obj.imag) or math.isinf(obj.real) or math.isinf(obj.imag)
                else:
                    return math.isnan(obj) or math.isinf(obj)
        return False
    except (TypeError, ValueError):
        return False


def validate_parameters(param_spec: Dict[str, Dict[str, Any]]) -> Callable[[F], F]:
    """Decorator for comprehensive parameter validation.
    
    Args:
        param_spec: Dictionary specifying parameter validation rules
                   Format: {'param_name': {'type': type, 'range': (min, max), 'required': bool}}
    
    Returns:
        Decorated function with parameter validation
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, spec in param_spec.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    _validate_single_parameter(param_name, value, spec)
                elif spec.get('required', False):
                    raise ValidationError(
                        f"Required parameter '{param_name}' is missing",
                        error_code="MISSING_REQUIRED_PARAMETER",
                        context={'function': func.__name__, 'parameter': param_name}
                    )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def _validate_single_parameter(name: str, value: Any, spec: Dict[str, Any]) -> None:
    """Validate a single parameter according to specification."""

    # Type validation
    if 'type' in spec:
        expected_type = spec['type']
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                error_code="TYPE_MISMATCH",
                context={'parameter': name, 'expected_type': expected_type.__name__, 'actual_type': type(value).__name__}
            )

    # Range validation
    if 'range' in spec and isinstance(value, (int, float)):
        min_val, max_val = spec['range']
        if value < min_val or value > max_val:
            raise ValidationError(
                f"Parameter '{name}' must be in range [{min_val}, {max_val}], got {value}",
                error_code="OUT_OF_RANGE",
                context={'parameter': name, 'value': value, 'range': spec['range']}
            )

    # Shape validation (for array-like objects)
    if 'shape' in spec and hasattr(value, '__len__'):
        expected_shape = spec['shape']
        if hasattr(value, 'shape'):
            actual_shape = value.shape
        else:
            actual_shape = (len(value),)

        if actual_shape != expected_shape:
            raise ValidationError(
                f"Parameter '{name}' must have shape {expected_shape}, got {actual_shape}",
                error_code="SHAPE_MISMATCH",
                context={'parameter': name, 'expected_shape': expected_shape, 'actual_shape': actual_shape}
            )

    # Custom validation
    if 'validator' in spec:
        validator = spec['validator']
        if not validator(value):
            raise ValidationError(
                f"Parameter '{name}' failed custom validation",
                error_code="CUSTOM_VALIDATION_FAILED",
                context={'parameter': name, 'value': value}
            )


def safe_division(numerator: Union[int, float], denominator: Union[int, float],
                 default: float = 0.0) -> float:
    """Safe division with zero-check and error handling.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if abs(denominator) < 1e-15:
            logging.warning(f"Division by near-zero value: {denominator}, using default: {default}")
            return default

        result = numerator / denominator

        # Check for overflow/underflow
        if abs(result) > 1e15:
            logging.warning(f"Division result too large: {result}, clamping to ±1e15")
            return 1e15 if result > 0 else -1e15

        return result

    except (ZeroDivisionError, OverflowError) as e:
        logging.warning(f"Division error: {e}, using default: {default}")
        return default


def safe_sqrt(value: Union[int, float], default: float = 0.0) -> float:
    """Safe square root with negative value handling.
    
    Args:
        value: Input value
        default: Default value for negative inputs
        
    Returns:
        Square root or default value
    """
    try:
        if value < 0:
            if abs(value) < 1e-12:
                # Small negative values likely due to numerical errors
                return 0.0
            else:
                logging.warning(f"Square root of negative value: {value}, using default: {default}")
                return default

        import math
        return math.sqrt(value)

    except (ValueError, TypeError) as e:
        logging.warning(f"Square root error: {e}, using default: {default}")
        return default


def safe_log(value: Union[int, float], base: Optional[float] = None, default: float = -float('inf')) -> float:
    """Safe logarithm with non-positive value handling.
    
    Args:
        value: Input value
        base: Logarithm base (None for natural log)
        default: Default value for non-positive inputs
        
    Returns:
        Logarithm or default value
    """
    try:
        if value <= 0:
            if abs(value) < 1e-15:
                logging.warning(f"Logarithm of near-zero value: {value}, using large negative default")
                return -50.0  # Large negative value instead of -inf
            else:
                logging.warning(f"Logarithm of non-positive value: {value}, using default: {default}")
                return default

        import math
        if base is None:
            return math.log(value)
        else:
            return math.log(value, base)

    except (ValueError, TypeError) as e:
        logging.warning(f"Logarithm error: {e}, using default: {default}")
        return default


def normalize_vector(vector: list, tolerance: float = 1e-12) -> list:
    """Safely normalize a vector with zero-magnitude handling.
    
    Args:
        vector: Input vector as list
        tolerance: Minimum magnitude tolerance
        
    Returns:
        Normalized unit vector
        
    Raises:
        NumericalError: If vector cannot be normalized
    """
    try:
        # Calculate magnitude
        magnitude_squared = sum(x**2 for x in vector)
        magnitude = safe_sqrt(magnitude_squared)

        if magnitude < tolerance:
            raise NumericalError(
                f"Cannot normalize vector with magnitude {magnitude} < {tolerance}",
                error_code="ZERO_MAGNITUDE_VECTOR",
                context={'vector': vector, 'magnitude': magnitude, 'tolerance': tolerance}
            )

        # Normalize
        normalized = [x / magnitude for x in vector]

        # Verify normalization
        new_magnitude = safe_sqrt(sum(x**2 for x in normalized))
        if abs(new_magnitude - 1.0) > 1e-10:
            logging.warning(f"Vector normalization resulted in magnitude {new_magnitude}, expected 1.0")

        return normalized

    except Exception as e:
        raise NumericalError(
            f"Vector normalization failed: {e}",
            error_code="NORMALIZATION_FAILED",
            context={'vector': vector, 'error': str(e)}
        ) from e


class ErrorRecoveryManager:
    """Manager for coordinated error recovery strategies."""

    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.max_error_rate = 0.1  # 10% error rate threshold
        self.error_window_size = 100  # Rolling window for error rate calculation

    def register_recovery_strategy(self, error_type: type, strategy: Callable) -> None:
        """Register a recovery strategy for specific error types.
        
        Args:
            error_type: Exception type to handle
            strategy: Recovery function to call
        """
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            context: Context information
            
        Returns:
            Recovery result if available, None otherwise
        """
        error_type = type(error)

        # Update error counts
        self._update_error_counts(error_type)

        # Check error rate
        if self._is_error_rate_too_high(error_type):
            logging.critical(f"High error rate detected for {error_type.__name__}, system may be unstable")

        # Attempt recovery
        if error_type in self.recovery_strategies:
            try:
                recovery_strategy = self.recovery_strategies[error_type]
                return recovery_strategy(error, context)
            except Exception as recovery_error:
                logging.error(f"Recovery strategy failed: {recovery_error}")

        # No recovery available
        logging.error(f"No recovery strategy for {error_type.__name__}: {error}")
        return None

    def _update_error_counts(self, error_type: type) -> None:
        """Update error count tracking."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = []

        # Add timestamp (simplified as counter)
        self.error_counts[error_type].append(1)

        # Keep only recent errors
        if len(self.error_counts[error_type]) > self.error_window_size:
            self.error_counts[error_type].pop(0)

    def _is_error_rate_too_high(self, error_type: type) -> bool:
        """Check if error rate exceeds threshold."""
        if error_type not in self.error_counts:
            return False

        error_count = len(self.error_counts[error_type])
        error_rate = error_count / self.error_window_size

        return error_rate > self.max_error_rate


# Global error recovery manager
error_recovery_manager = ErrorRecoveryManager()


def setup_error_handling(log_level: str = "WARNING") -> None:
    """Set up global error handling configuration.
    
    Args:
        log_level: Logging level for error handling
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Register default recovery strategies
    def physics_error_recovery(error: PhysicsError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for physics errors."""
        logging.warning("Attempting physics error recovery")

        # Return safe default values based on context
        if 'magnetization' in context:
            return [0.0, 0.0, 1.0]  # Default z-aligned magnetization
        elif 'field' in context:
            return [0.0, 0.0, 0.0]  # Zero field

        return None

    def device_error_recovery(error: DeviceError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for device errors."""
        logging.warning("Attempting device error recovery")

        # Return safe device state
        if 'resistance' in context:
            return 1000.0  # Safe resistance value
        elif 'energy' in context:
            return 0.0  # Zero energy consumption

        return None

    # Register recovery strategies
    error_recovery_manager.register_recovery_strategy(PhysicsError, physics_error_recovery)
    error_recovery_manager.register_recovery_strategy(DeviceError, device_error_recovery)

    logging.info("Error handling system initialized")


# Example usage and testing
if __name__ == "__main__":
    setup_error_handling()

    # Test robust computation
    @robust_computation(max_retries=2, fallback_value=0.0)
    def potentially_failing_function(x):
        if x < 0:
            raise ValueError("Negative input")
        return x ** 0.5

    # Test parameter validation
    @validate_parameters({
        'value': {'type': (int, float), 'range': (0, 100), 'required': True},
        'vector': {'type': list, 'shape': (3,), 'required': True}
    })
    def validated_function(value, vector):
        return value * sum(vector)

    print("Error handling module ready for use!")
