"""Security utilities for SpinTorque Gym.

This module provides security measures including input sanitization,
rate limiting, and protection against malicious inputs.
"""

import hashlib
import time
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .logging_config import SecurityLogger


class RateLimiter:
    """Rate limiter for API calls and resource usage."""
    
    def __init__(self, max_calls: int = 1000, time_window: float = 60.0):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(deque)
        self.logger = SecurityLogger()
    
    def is_allowed(self, client_id: str = "default") -> bool:
        """Check if client is within rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if call is allowed, False otherwise
        """
        now = time.time()
        client_calls = self.calls[client_id]
        
        # Remove old calls outside time window
        while client_calls and client_calls[0] < now - self.time_window:
            client_calls.popleft()
        
        # Check if under limit
        if len(client_calls) < self.max_calls:
            client_calls.append(now)
            return True
        
        # Log rate limit violation
        self.logger.log_safety_violation(
            "rate_limit_exceeded",
            {
                'client_id': client_id,
                'current_calls': len(client_calls),
                'max_calls': self.max_calls,
                'time_window': self.time_window
            }
        )
        
        return False


class InputSanitizer:
    """Sanitizes and validates inputs for security."""
    
    def __init__(self):
        self.logger = SecurityLogger()
        
        # Regex patterns for validation
        self.safe_string_pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
        self.numeric_pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    
    def sanitize_string(self, value: str, max_length: int = 100) -> str:
        """Sanitize string input.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If string is unsafe
        """
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        original_value = value
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
            self.logger.log_parameter_sanitization(
                "string_length", original_value, value
            )
        
        # Check for safe characters only
        if not self.safe_string_pattern.match(value):
            # Remove unsafe characters
            safe_value = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', value)
            self.logger.log_parameter_sanitization(
                "string_characters", value, safe_value
            )
            value = safe_value
        
        return value
    
    def sanitize_numeric(self, value: Union[int, float, str], 
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None) -> float:
        """Sanitize numeric input.
        
        Args:
            value: Input value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Sanitized numeric value
            
        Raises:
            ValueError: If value cannot be sanitized
        """
        original_value = value
        
        # Convert to float
        try:
            if isinstance(value, str):
                if not self.numeric_pattern.match(value.strip()):
                    raise ValueError("Invalid numeric format")
                value = float(value)
            else:
                value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert to number: {original_value}")
        
        # Check for special values
        if not np.isfinite(value):
            self.logger.log_safety_violation(
                "non_finite_input",
                {'original_value': original_value, 'converted_value': value}
            )
            raise ValueError("Non-finite numeric value not allowed")
        
        # Apply bounds
        if min_value is not None and value < min_value:
            self.logger.log_parameter_sanitization(
                "numeric_lower_bound", original_value, min_value
            )
            value = min_value
        
        if max_value is not None and value > max_value:
            self.logger.log_parameter_sanitization(
                "numeric_upper_bound", original_value, max_value
            )
            value = max_value
        
        return value
    
    def sanitize_array(self, value: Union[List, np.ndarray], 
                      max_size: int = 1000,
                      element_type: type = float) -> np.ndarray:
        """Sanitize array input.
        
        Args:
            value: Input array
            max_size: Maximum array size
            element_type: Expected element type
            
        Returns:
            Sanitized numpy array
            
        Raises:
            ValueError: If array is unsafe
        """
        original_value = value
        
        # Convert to numpy array
        try:
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=element_type)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert to array: {original_value}")
        
        # Check size
        if value.size > max_size:
            self.logger.log_safety_violation(
                "array_size_exceeded",
                {'size': value.size, 'max_size': max_size}
            )
            raise ValueError(f"Array too large: {value.size} > {max_size}")
        
        # Check for finite values
        if not np.all(np.isfinite(value)):
            self.logger.log_safety_violation(
                "non_finite_array_elements",
                {'array_shape': value.shape}
            )
            raise ValueError("Array contains non-finite values")
        
        # Check for reasonable value ranges
        abs_max = np.max(np.abs(value))
        if abs_max > 1e15:
            self.logger.log_safety_violation(
                "extreme_array_values",
                {'max_abs_value': abs_max}
            )
            raise ValueError("Array contains extreme values")
        
        return value
    
    def sanitize_dict(self, value: Dict[str, Any], 
                     allowed_keys: Optional[List[str]] = None,
                     max_depth: int = 5,
                     _current_depth: int = 0) -> Dict[str, Any]:
        """Sanitize dictionary input.
        
        Args:
            value: Input dictionary
            allowed_keys: List of allowed keys (None = all allowed)
            max_depth: Maximum nesting depth
            _current_depth: Internal recursion depth counter
            
        Returns:
            Sanitized dictionary
            
        Raises:
            ValueError: If dictionary is unsafe
        """
        if not isinstance(value, dict):
            raise ValueError("Input must be a dictionary")
        
        if _current_depth >= max_depth:
            self.logger.log_safety_violation(
                "dict_depth_exceeded",
                {'depth': _current_depth, 'max_depth': max_depth}
            )
            raise ValueError("Dictionary nesting too deep")
        
        sanitized = {}
        
        for key, val in value.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            
            try:
                key = self.sanitize_string(key, max_length=50)
            except ValueError:
                continue  # Skip invalid keys
            
            # Check allowed keys
            if allowed_keys is not None and key not in allowed_keys:
                self.logger.log_parameter_sanitization(
                    "dict_key_filtered", key, None
                )
                continue
            
            # Recursively sanitize values
            if isinstance(val, dict):
                try:
                    sanitized[key] = self.sanitize_dict(
                        val, allowed_keys, max_depth, _current_depth + 1
                    )
                except ValueError:
                    continue  # Skip invalid nested dicts
            elif isinstance(val, (list, np.ndarray)):
                try:
                    sanitized[key] = self.sanitize_array(val)
                except ValueError:
                    continue  # Skip invalid arrays
            elif isinstance(val, (int, float)):
                try:
                    sanitized[key] = self.sanitize_numeric(val)
                except ValueError:
                    continue  # Skip invalid numbers
            elif isinstance(val, str):
                try:
                    sanitized[key] = self.sanitize_string(val)
                except ValueError:
                    continue  # Skip invalid strings
            else:
                # For other types, convert to string and sanitize
                try:
                    sanitized[key] = self.sanitize_string(str(val))
                except ValueError:
                    continue  # Skip unserializable values
        
        return sanitized


class SecurityManager:
    """Main security manager for SpinTorque Gym."""
    
    def __init__(self, 
                 rate_limit_calls: int = 1000,
                 rate_limit_window: float = 60.0):
        """Initialize security manager.
        
        Args:
            rate_limit_calls: Rate limit calls per window
            rate_limit_window: Rate limit window in seconds
        """
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        self.sanitizer = InputSanitizer()
        self.logger = SecurityLogger()
        
        # Track security events
        self.security_events = deque(maxlen=1000)
    
    def validate_environment_creation(self, config: Dict[str, Any], 
                                    client_id: str = "default") -> Dict[str, Any]:
        """Validate environment creation request.
        
        Args:
            config: Environment configuration
            client_id: Client identifier
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If validation fails
        """
        # Check rate limits
        if not self.rate_limiter.is_allowed(client_id):
            raise ValueError("Rate limit exceeded")
        
        # Sanitize configuration
        allowed_keys = [
            'device_type', 'max_steps', 'temperature', 'success_threshold',
            'action_mode', 'observation_mode', 'seed', 'physics_method'
        ]
        
        try:
            config = self.sanitizer.sanitize_dict(config, allowed_keys)
        except ValueError as e:
            self.logger.log_input_validation("environment_config", "failed", error=str(e))
            raise
        
        self.logger.log_input_validation("environment_config", "passed")
        return config
    
    def validate_action(self, action: Union[np.ndarray, int, float],
                       action_space: Any,
                       client_id: str = "default") -> Union[np.ndarray, int, float]:
        """Validate RL action.
        
        Args:
            action: Input action
            action_space: Action space for validation
            client_id: Client identifier
            
        Returns:
            Validated action
            
        Raises:
            ValueError: If action is invalid
        """
        # Check rate limits
        if not self.rate_limiter.is_allowed(client_id):
            raise ValueError("Rate limit exceeded")
        
        try:
            if isinstance(action, (list, np.ndarray)):
                action = self.sanitizer.sanitize_array(action, max_size=100)
            elif isinstance(action, (int, float)):
                action = self.sanitizer.sanitize_numeric(action)
            else:
                raise ValueError(f"Unsupported action type: {type(action)}")
        except ValueError as e:
            self.logger.log_input_validation("action", "failed", error=str(e))
            raise
        
        self.logger.log_input_validation("action", "passed")
        return action
    
    def validate_device_params(self, params: Dict[str, Any],
                             client_id: str = "default") -> Dict[str, Any]:
        """Validate device parameters.
        
        Args:
            params: Device parameters
            client_id: Client identifier
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Check rate limits
        if not self.rate_limiter.is_allowed(client_id):
            raise ValueError("Rate limit exceeded")
        
        # Device parameter security limits
        security_limits = {
            'volume': (1e-30, 1e-15),  # m³
            'saturation_magnetization': (1e3, 1e7),  # A/m
            'damping': (1e-6, 1.0),  # dimensionless
            'uniaxial_anisotropy': (1e3, 1e8),  # J/m³
            'temperature': (0.1, 2000),  # K
            'current_density': (-1e9, 1e9),  # A/m²
            'voltage': (-10, 10),  # V
        }
        
        validated = {}
        
        for key, value in params.items():
            try:
                key = self.sanitizer.sanitize_string(key)
                
                if key in security_limits:
                    min_val, max_val = security_limits[key]
                    value = self.sanitizer.sanitize_numeric(value, min_val, max_val)
                elif isinstance(value, (int, float)):
                    value = self.sanitizer.sanitize_numeric(value)
                elif isinstance(value, str):
                    value = self.sanitizer.sanitize_string(value)
                elif isinstance(value, (list, np.ndarray)):
                    value = self.sanitizer.sanitize_array(value)
                
                validated[key] = value
                
            except ValueError as e:
                self.logger.log_input_validation(
                    f"device_param_{key}", "failed", error=str(e)
                )
                continue  # Skip invalid parameters
        
        self.logger.log_input_validation("device_params", "passed")
        return validated
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event.
        
        Args:
            event_type: Type of security event
            details: Event details
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        self.logger.log_safety_violation(event_type, details)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics.
        
        Returns:
            Dictionary with security statistics
        """
        now = time.time()
        recent_events = [
            event for event in self.security_events
            if now - event['timestamp'] < 3600  # Last hour
        ]
        
        event_types = defaultdict(int)
        for event in recent_events:
            event_types[event['type']] += 1
        
        return {
            'total_events_last_hour': len(recent_events),
            'event_types': dict(event_types),
            'rate_limiter_clients': len(self.rate_limiter.calls),
            'total_recorded_events': len(self.security_events)
        }


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager