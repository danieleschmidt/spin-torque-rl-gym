"""Input validation and error handling utilities.

This module provides comprehensive validation functions for ensuring
data integrity and preventing runtime errors in the RL environment.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PhysicsValidator:
    """Validator for physics-related parameters."""

    @staticmethod
    def validate_magnetization(magnetization: np.ndarray, name: str = "magnetization") -> np.ndarray:
        """Validate and normalize magnetization vector.
        
        Args:
            magnetization: Input magnetization vector
            name: Parameter name for error messages
            
        Returns:
            Normalized magnetization vector
            
        Raises:
            ValidationError: If magnetization is invalid
        """
        if not isinstance(magnetization, np.ndarray):
            try:
                magnetization = np.array(magnetization, dtype=float)
            except (ValueError, TypeError):
                raise ValidationError(f"{name} must be convertible to numpy array")

        if magnetization.shape != (3,):
            raise ValidationError(f"{name} must be a 3D vector, got shape {magnetization.shape}")

        if not np.all(np.isfinite(magnetization)):
            raise ValidationError(f"{name} contains non-finite values: {magnetization}")

        magnitude = np.linalg.norm(magnetization)
        if magnitude < 1e-12:
            raise ValidationError(f"{name} has zero magnitude")

        # Normalize and return
        normalized = magnetization / magnitude

        if not np.all(np.isfinite(normalized)):
            raise ValidationError(f"{name} normalization failed")

        return normalized

    @staticmethod
    def validate_field(field: np.ndarray, name: str = "field") -> np.ndarray:
        """Validate magnetic field vector.
        
        Args:
            field: Input field vector
            name: Parameter name for error messages
            
        Returns:
            Validated field vector
            
        Raises:
            ValidationError: If field is invalid
        """
        if not isinstance(field, np.ndarray):
            try:
                field = np.array(field, dtype=float)
            except (ValueError, TypeError):
                raise ValidationError(f"{name} must be convertible to numpy array")

        if field.shape != (3,):
            raise ValidationError(f"{name} must be a 3D vector, got shape {field.shape}")

        if not np.all(np.isfinite(field)):
            raise ValidationError(f"{name} contains non-finite values: {field}")

        # Check for reasonable field magnitude (T)
        magnitude = np.linalg.norm(field)
        if magnitude > 10.0:  # 10 Tesla is very high
            logger.warning(f"{name} magnitude {magnitude:.2f} T is unusually high")

        return field

    @staticmethod
    def validate_positive_scalar(value: Union[int, float], name: str,
                                min_value: float = 1e-20) -> float:
        """Validate positive scalar parameter.
        
        Args:
            value: Input value
            name: Parameter name
            min_value: Minimum allowed value
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value is invalid
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be a number, got {type(value)}")

        if not np.isfinite(value):
            raise ValidationError(f"{name} must be finite, got {value}")

        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")

        if value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {value}")

        return value

    @staticmethod
    def validate_probability(value: Union[int, float], name: str) -> float:
        """Validate probability value (0 to 1).
        
        Args:
            value: Input probability
            name: Parameter name
            
        Returns:
            Validated probability
            
        Raises:
            ValidationError: If probability is invalid
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be a number, got {type(value)}")

        if not np.isfinite(value):
            raise ValidationError(f"{name} must be finite, got {value}")

        if not 0 <= value <= 1:
            raise ValidationError(f"{name} must be between 0 and 1, got {value}")

        return value

    @staticmethod
    def validate_temperature(temperature: Union[int, float]) -> float:
        """Validate temperature value.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Validated temperature
            
        Raises:
            ValidationError: If temperature is invalid
        """
        temp = PhysicsValidator.validate_positive_scalar(temperature, "temperature", min_value=1e-3)

        if temp > 2000:  # Above melting point of most materials
            logger.warning(f"Temperature {temp} K is very high")
        elif temp < 1:  # Below 1 K
            logger.warning(f"Temperature {temp} K is very low")

        return temp

    @staticmethod
    def validate_device_params(params: Dict[str, Any], device_type: str) -> Dict[str, Any]:
        """Validate device parameters.
        
        Args:
            params: Device parameter dictionary
            device_type: Type of device for validation
            
        Returns:
            Validated parameter dictionary
            
        Raises:
            ValidationError: If parameters are invalid
        """
        validated = {}

        # Common required parameters
        required_params = {
            'volume': (PhysicsValidator.validate_positive_scalar, {'min_value': 1e-30}),
            'saturation_magnetization': (PhysicsValidator.validate_positive_scalar, {'min_value': 1e3}),
            'damping': (PhysicsValidator.validate_probability, {}),
        }

        # Device-specific requirements
        if device_type in ['stt_mram', 'sot_mram', 'vcma_mram']:
            required_params.update({
                'uniaxial_anisotropy': (PhysicsValidator.validate_positive_scalar, {'min_value': 1e3}),
                'easy_axis': (PhysicsValidator.validate_magnetization, {}),
            })

        if device_type == 'stt_mram':
            required_params['polarization'] = (PhysicsValidator.validate_probability, {})

        if device_type == 'skyrmion':
            required_params.update({
                'dmi_constant': (PhysicsValidator.validate_positive_scalar, {'min_value': 1e-6}),
                'skyrmion_radius': (PhysicsValidator.validate_positive_scalar, {'min_value': 1e-9}),
            })

        # Validate required parameters
        for param_name, (validator, kwargs) in required_params.items():
            if param_name not in params:
                raise ValidationError(f"Missing required parameter: {param_name}")

            try:
                validated[param_name] = validator(params[param_name], param_name, **kwargs)
            except ValidationError as e:
                raise ValidationError(f"Invalid {param_name}: {e}")

        # Copy optional parameters with basic validation
        optional_params = set(params.keys()) - set(required_params.keys())
        for param_name in optional_params:
            value = params[param_name]
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    logger.warning(f"Non-finite optional parameter {param_name}: {value}")
                    continue
            validated[param_name] = value

        return validated


class ActionValidator:
    """Validator for RL actions."""

    @staticmethod
    def validate_continuous_action(action: np.ndarray,
                                 action_space: Any) -> np.ndarray:
        """Validate continuous action vector.
        
        Args:
            action: Input action vector
            action_space: Gymnasium action space
            
        Returns:
            Validated action vector
            
        Raises:
            ValidationError: If action is invalid
        """
        if not isinstance(action, np.ndarray):
            try:
                action = np.array(action, dtype=float)
            except (ValueError, TypeError):
                raise ValidationError("Action must be convertible to numpy array")

        if not np.all(np.isfinite(action)):
            raise ValidationError(f"Action contains non-finite values: {action}")

        if hasattr(action_space, 'shape') and action.shape != action_space.shape:
            raise ValidationError(f"Action shape {action.shape} doesn't match space {action_space.shape}")

        # Clip to bounds if available
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            clipped_action = np.clip(action, action_space.low, action_space.high)
            if not np.allclose(action, clipped_action, rtol=1e-10):
                logger.warning("Action was clipped to bounds")
            action = clipped_action

        return action

    @staticmethod
    def validate_discrete_action(action: Union[int, np.integer],
                                action_space: Any) -> int:
        """Validate discrete action.
        
        Args:
            action: Input action
            action_space: Gymnasium action space
            
        Returns:
            Validated action
            
        Raises:
            ValidationError: If action is invalid
        """
        try:
            action = int(action)
        except (ValueError, TypeError):
            raise ValidationError(f"Discrete action must be an integer, got {type(action)}")

        if hasattr(action_space, 'n'):
            if not 0 <= action < action_space.n:
                raise ValidationError(f"Action {action} out of range [0, {action_space.n})")

        return action


class NumericalValidator:
    """Validator for numerical stability."""

    @staticmethod
    def check_matrix_condition(matrix: np.ndarray, name: str = "matrix") -> float:
        """Check condition number of matrix.
        
        Args:
            matrix: Input matrix
            name: Matrix name for logging
            
        Returns:
            Condition number
        """
        try:
            cond = np.linalg.cond(matrix)

            if cond > 1e12:
                logger.warning(f"{name} is ill-conditioned (cond={cond:.2e})")
            elif cond > 1e8:
                logger.info(f"{name} condition number is high (cond={cond:.2e})")

            return cond
        except np.linalg.LinAlgError:
            logger.error(f"Failed to compute condition number for {name}")
            return np.inf

    @staticmethod
    def validate_integration_step(dt: float, stability_limit: float = 1e-6) -> float:
        """Validate integration time step for numerical stability.
        
        Args:
            dt: Time step
            stability_limit: Maximum allowed time step
            
        Returns:
            Validated time step
            
        Raises:
            ValidationError: If time step is invalid
        """
        dt = PhysicsValidator.validate_positive_scalar(dt, "time_step")

        if dt > stability_limit:
            logger.warning(f"Time step {dt:.2e} s may cause instability (limit: {stability_limit:.2e} s)")

        if dt < 1e-15:
            logger.warning(f"Time step {dt:.2e} s is extremely small")

        return dt

    @staticmethod
    def check_convergence(values: List[float], tolerance: float = 1e-8,
                         min_iterations: int = 3) -> Tuple[bool, float]:
        """Check if iterative process has converged.
        
        Args:
            values: Sequence of values from iterations
            tolerance: Convergence tolerance
            min_iterations: Minimum number of iterations
            
        Returns:
            (converged, error) tuple
        """
        if len(values) < min_iterations:
            return False, np.inf

        if len(values) < 2:
            return False, np.inf

        # Check relative change
        current = values[-1]
        previous = values[-2]

        if abs(current) < 1e-15 and abs(previous) < 1e-15:
            return True, 0.0

        if abs(current) > 1e-15:
            relative_error = abs((current - previous) / current)
        else:
            relative_error = abs(current - previous)

        converged = relative_error < tolerance

        return converged, relative_error


class SafetyValidator:
    """Validator for safety constraints."""

    @staticmethod
    def validate_current_density(current_density: float,
                                device_type: str) -> float:
        """Validate current density for device safety.
        
        Args:
            current_density: Current density (A/m²)
            device_type: Type of device
            
        Returns:
            Validated current density
            
        Raises:
            ValidationError: If current density is unsafe
        """
        if not np.isfinite(current_density):
            raise ValidationError("Current density must be finite")

        abs_current = abs(current_density)

        # Safety limits by device type (A/m²)
        safety_limits = {
            'stt_mram': 1e8,   # 100 MA/m²
            'sot_mram': 1e9,   # 1 GA/m²
            'vcma_mram': 1e7,  # 10 MA/m²
            'skyrmion': 1e8,   # 100 MA/m²
        }

        limit = safety_limits.get(device_type, 1e8)

        if abs_current > limit:
            logger.error(f"Current density {abs_current:.2e} A/m² exceeds safety limit {limit:.2e} A/m²")
            raise ValidationError(f"Current density too high for {device_type} device")

        # Warning thresholds (50% of limit)
        warning_limit = limit * 0.5
        if abs_current > warning_limit:
            logger.warning(f"High current density {abs_current:.2e} A/m² (limit: {limit:.2e} A/m²)")

        return current_density

    @staticmethod
    def validate_voltage(voltage: float, breakdown_voltage: float = 2.0) -> float:
        """Validate voltage for dielectric breakdown safety.
        
        Args:
            voltage: Applied voltage (V)
            breakdown_voltage: Dielectric breakdown voltage (V)
            
        Returns:
            Validated voltage
            
        Raises:
            ValidationError: If voltage is unsafe
        """
        if not np.isfinite(voltage):
            raise ValidationError("Voltage must be finite")

        abs_voltage = abs(voltage)

        if abs_voltage > breakdown_voltage:
            raise ValidationError(f"Voltage {abs_voltage:.2f} V exceeds breakdown limit {breakdown_voltage:.2f} V")

        # Warning at 80% of breakdown
        warning_voltage = 0.8 * breakdown_voltage
        if abs_voltage > warning_voltage:
            logger.warning(f"High voltage {abs_voltage:.2f} V (breakdown: {breakdown_voltage:.2f} V)")

        return voltage

    @staticmethod
    def validate_power(power: float, max_power: float = 1e-6) -> float:
        """Validate power consumption.
        
        Args:
            power: Power consumption (W)
            max_power: Maximum allowed power (W)
            
        Returns:
            Validated power
            
        Raises:
            ValidationError: If power is too high
        """
        power = PhysicsValidator.validate_positive_scalar(power, "power", min_value=0)

        if power > max_power:
            logger.warning(f"High power consumption {power:.2e} W (limit: {max_power:.2e} W)")

        return power


def validate_environment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate environment configuration.
    
    Args:
        config: Environment configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    validated = {}

    # Device type
    device_type = config.get('device_type', 'stt_mram')
    allowed_devices = ['stt_mram', 'sot_mram', 'vcma_mram', 'skyrmion']
    if device_type not in allowed_devices:
        raise ValidationError(f"Invalid device_type: {device_type}. Allowed: {allowed_devices}")
    validated['device_type'] = device_type

    # Temperature
    if 'temperature' in config:
        validated['temperature'] = PhysicsValidator.validate_temperature(config['temperature'])

    # Episode parameters
    if 'max_steps' in config:
        max_steps = int(config['max_steps'])
        if max_steps < 1:
            raise ValidationError("max_steps must be positive")
        if max_steps > 10000:
            logger.warning(f"Large max_steps: {max_steps}")
        validated['max_steps'] = max_steps

    # Success threshold
    if 'success_threshold' in config:
        validated['success_threshold'] = PhysicsValidator.validate_probability(
            config['success_threshold'], 'success_threshold'
        )

    # Action mode
    action_mode = config.get('action_mode', 'continuous')
    allowed_modes = ['continuous', 'discrete']
    if action_mode not in allowed_modes:
        raise ValidationError(f"Invalid action_mode: {action_mode}. Allowed: {allowed_modes}")
    validated['action_mode'] = action_mode

    # Copy other validated parameters
    for key, value in config.items():
        if key not in validated:
            validated[key] = value

    return validated
