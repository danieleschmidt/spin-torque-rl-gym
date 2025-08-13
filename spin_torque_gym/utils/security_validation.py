"""Security and input validation system for Spin Torque RL-Gym.

This module provides comprehensive security measures, input sanitization,
and validation for scientific computing applications.
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union


class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationLevel(Enum):
    """Input validation level."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class SecurityConfig:
    """Security configuration."""
    validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_rate_limiting: bool = True
    enable_input_logging: bool = False
    max_input_size: int = 1024 * 1024  # 1MB
    allowed_file_extensions: List[str] = None
    blocked_patterns: List[str] = None

    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.json', '.yaml', '.yml', '.txt', '.csv']
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script.*?</script>',  # Script tags
                r'javascript:',          # JavaScript URLs
                r'vbscript:',           # VBScript URLs
                r'data:',               # Data URLs
                r'file:',               # File URLs
                r'\.\./.*\.\.',         # Path traversal
                r'[;&|`]',              # Command injection
            ]


class InputSanitizer:
    """Comprehensive input sanitization and validation."""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = []
        for pattern in self.config.blocked_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logging.warning(f"Invalid regex pattern '{pattern}': {e}")

    def sanitize_string(self, value: str, max_length: int = None) -> str:
        """Sanitize string input.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input is invalid or malicious
        """
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Length check
        max_len = max_length or self.config.max_input_size
        if len(value) > max_len:
            raise ValueError(f"Input too long: {len(value)} > {max_len}")

        # Check for blocked patterns
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                logging.warning(f"Blocked pattern detected: {pattern.pattern}")
                raise ValueError("Input contains potentially malicious content")

        # Basic sanitization
        if self.config.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Remove null bytes and control characters (except whitespace)
            sanitized = ''.join(char for char in value
                              if ord(char) >= 32 or char in '\t\n\r')

            # Additional paranoid checks
            if self.config.validation_level == ValidationLevel.PARANOID:
                # Check for suspicious character sequences
                suspicious_chars = ['<', '>', '"', "'", '&', '%']
                if any(char in sanitized for char in suspicious_chars):
                    # HTML entity encode
                    sanitized = self._html_encode(sanitized)
        else:
            sanitized = value

        # Log if enabled
        if self.config.enable_input_logging:
            logging.info(f"Sanitized input: {len(value)} -> {len(sanitized)} chars")

        return sanitized

    def _html_encode(self, text: str) -> str:
        """HTML encode special characters."""
        html_entities = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;',
        }
        for char, entity in html_entities.items():
            text = text.replace(char, entity)
        return text

    def validate_numeric_input(
        self,
        value: Union[int, float],
        min_value: float = None,
        max_value: float = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> Union[int, float]:
        """Validate numeric input.
        
        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            Validated numeric value
            
        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value, got {type(value)}")

        # Check for NaN and infinity
        import math
        if math.isnan(value) and not allow_nan:
            raise ValueError("NaN values are not allowed")

        if math.isinf(value) and not allow_inf:
            raise ValueError("Infinite values are not allowed")

        # Range validation
        if min_value is not None and value < min_value:
            raise ValueError(f"Value {value} below minimum {min_value}")

        if max_value is not None and value > max_value:
            raise ValueError(f"Value {value} above maximum {max_value}")

        return value

    def validate_array_input(
        self,
        value: List[Union[int, float]],
        expected_shape: tuple = None,
        element_range: tuple = None
    ) -> List[Union[int, float]]:
        """Validate array/list input.
        
        Args:
            value: Array to validate
            expected_shape: Expected array shape
            element_range: (min, max) range for elements
            
        Returns:
            Validated array
            
        Raises:
            ValueError: If array is invalid
        """
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected list or tuple, got {type(value)}")

        # Shape validation
        if expected_shape is not None:
            if len(expected_shape) == 1:
                # 1D array
                if len(value) != expected_shape[0]:
                    raise ValueError(f"Expected length {expected_shape[0]}, got {len(value)}")
            else:
                # Multi-dimensional - basic check
                if len(value) != expected_shape[0]:
                    raise ValueError(f"Expected first dimension {expected_shape[0]}, got {len(value)}")

        # Element validation
        validated_elements = []
        for i, element in enumerate(value):
            try:
                if element_range:
                    min_val, max_val = element_range
                    validated_element = self.validate_numeric_input(
                        element, min_value=min_val, max_value=max_val
                    )
                else:
                    validated_element = self.validate_numeric_input(element)
                validated_elements.append(validated_element)
            except ValueError as e:
                raise ValueError(f"Invalid element at index {i}: {e}")

        return validated_elements

    def validate_file_path(self, path: str) -> str:
        """Validate file path for security.
        
        Args:
            path: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            ValueError: If path is invalid or insecure
        """
        if not isinstance(path, str):
            raise ValueError("Path must be a string")

        # Basic sanitization
        sanitized_path = self.sanitize_string(path)

        # Path traversal prevention
        if '..' in sanitized_path:
            raise ValueError("Path traversal detected")

        # Absolute path check
        if os.path.isabs(sanitized_path):
            logging.warning(f"Absolute path provided: {sanitized_path}")
            if self.config.validation_level == ValidationLevel.PARANOID:
                raise ValueError("Absolute paths not allowed in paranoid mode")

        # File extension validation
        if self.config.allowed_file_extensions:
            _, ext = os.path.splitext(sanitized_path.lower())
            if ext not in self.config.allowed_file_extensions:
                raise ValueError(f"File extension '{ext}' not allowed")

        return sanitized_path


class RateLimiter:
    """Rate limiting for API protection."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> list of timestamps

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()

        # Initialize client if not exists
        if client_id not in self.requests:
            self.requests[client_id] = []

        client_requests = self.requests[client_id]

        # Remove old requests outside window
        cutoff_time = current_time - self.window_seconds
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(current_time)
        return True

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if client_id not in self.requests:
            return self.max_requests

        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        recent_requests = [req_time for req_time in self.requests[client_id]
                          if req_time > cutoff_time]

        return max(0, self.max_requests - len(recent_requests))


class SecurityAuditor:
    """Security auditing and monitoring."""

    def __init__(self):
        self.security_events = []
        self.suspicious_patterns = 0
        self.validation_failures = 0
        self.rate_limit_violations = 0

    def log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            message: Event description
            details: Additional event details
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity.value,
            'message': message,
            'details': details or {}
        }

        self.security_events.append(event)

        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]

        # Update counters
        if event_type == 'suspicious_pattern':
            self.suspicious_patterns += 1
        elif event_type == 'validation_failure':
            self.validation_failures += 1
        elif event_type == 'rate_limit_violation':
            self.rate_limit_violations += 1

        # Log based on severity
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }.get(severity, logging.WARNING)

        logging.log(log_level, f"SECURITY [{event_type}]: {message}")

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security audit summary."""
        return {
            'total_events': len(self.security_events),
            'suspicious_patterns': self.suspicious_patterns,
            'validation_failures': self.validation_failures,
            'rate_limit_violations': self.rate_limit_violations,
            'recent_events': self.security_events[-10:] if self.security_events else []
        }


class SecureHasher:
    """Secure hashing utilities."""

    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
        """Hash password securely.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 for password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # iterations
        )

        return password_hash.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: bytes) -> bool:
        """Verify password against hash.
        
        Args:
            password: Password to verify
            hashed_password: Stored hash (hex string)
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = SecureHasher.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hashed_password)

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Secure random token (hex string)
        """
        return secrets.token_hex(length)

    @staticmethod
    def compute_hmac(message: str, key: bytes) -> str:
        """Compute HMAC for message integrity.
        
        Args:
            message: Message to sign
            key: Secret key
            
        Returns:
            HMAC signature (hex string)
        """
        return hmac.new(key, message.encode('utf-8'), hashlib.sha256).hexdigest()


def secure_function(
    validation_level: ValidationLevel = ValidationLevel.STRICT,
    rate_limit: Optional[tuple] = None,  # (max_requests, window_seconds)
    require_auth: bool = False
):
    """Decorator for securing functions with validation and rate limiting.
    
    Args:
        validation_level: Input validation level
        rate_limit: Rate limiting configuration
        require_auth: Whether authentication is required
        
    Returns:
        Secured function decorator
    """
    def decorator(func):
        # Initialize components
        config = SecurityConfig(validation_level=validation_level)
        sanitizer = InputSanitizer(config)
        auditor = SecurityAuditor()

        if rate_limit:
            limiter = RateLimiter(max_requests=rate_limit[0], window_seconds=rate_limit[1])
        else:
            limiter = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting check
            if limiter:
                client_id = kwargs.get('client_id', 'default')
                if not limiter.is_allowed(client_id):
                    auditor.log_security_event(
                        'rate_limit_violation',
                        SecurityLevel.MEDIUM,
                        f"Rate limit exceeded for client {client_id}"
                    )
                    raise ValueError("Rate limit exceeded")

            # Authentication check (simplified)
            if require_auth:
                auth_token = kwargs.get('auth_token')
                if not auth_token:
                    auditor.log_security_event(
                        'auth_failure',
                        SecurityLevel.HIGH,
                        "Missing authentication token"
                    )
                    raise ValueError("Authentication required")

            # Input validation and sanitization
            try:
                validated_args = []
                for arg in args:
                    if isinstance(arg, str):
                        validated_args.append(sanitizer.sanitize_string(arg))
                    elif isinstance(arg, (int, float)):
                        validated_args.append(sanitizer.validate_numeric_input(arg))
                    elif isinstance(arg, (list, tuple)):
                        validated_args.append(sanitizer.validate_array_input(arg))
                    else:
                        validated_args.append(arg)

                validated_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str) and key not in ['client_id', 'auth_token']:
                        validated_kwargs[key] = sanitizer.sanitize_string(value)
                    elif isinstance(value, (int, float)):
                        validated_kwargs[key] = sanitizer.validate_numeric_input(value)
                    elif isinstance(value, (list, tuple)):
                        validated_kwargs[key] = sanitizer.validate_array_input(value)
                    else:
                        validated_kwargs[key] = value

                # Call original function with validated inputs
                return func(*validated_args, **validated_kwargs)

            except ValueError as e:
                auditor.log_security_event(
                    'validation_failure',
                    SecurityLevel.MEDIUM,
                    f"Input validation failed: {e}"
                )
                raise

        # Attach security components for inspection
        wrapper._security_config = config
        wrapper._security_auditor = auditor
        wrapper._rate_limiter = limiter

        return wrapper

    return decorator


# Global security components
global_config = SecurityConfig()
global_sanitizer = InputSanitizer(global_config)
global_auditor = SecurityAuditor()


def initialize_security(config: SecurityConfig = None) -> None:
    """Initialize global security system.
    
    Args:
        config: Security configuration
    """
    global global_config, global_sanitizer, global_auditor

    global_config = config or SecurityConfig()
    global_sanitizer = InputSanitizer(global_config)
    global_auditor = SecurityAuditor()

    logging.info(f"Security system initialized with {global_config.validation_level.value} validation")


def get_security_status() -> Dict[str, Any]:
    """Get current security system status."""
    return {
        'config': {
            'validation_level': global_config.validation_level.value,
            'rate_limiting_enabled': global_config.enable_rate_limiting,
            'input_logging_enabled': global_config.enable_input_logging,
            'max_input_size': global_config.max_input_size
        },
        'audit_summary': global_auditor.get_security_summary()
    }


# Example usage and testing
if __name__ == "__main__":
    # Initialize security
    initialize_security(SecurityConfig(validation_level=ValidationLevel.STRICT))

    # Test input sanitization
    sanitizer = InputSanitizer()

    # Test secure function decorator
    @secure_function(
        validation_level=ValidationLevel.STRICT,
        rate_limit=(10, 60),  # 10 requests per minute
        require_auth=False
    )
    def secure_calculation(x: float, y: float) -> float:
        """Example secure calculation function."""
        return x * y + 42.0

    try:
        result = secure_calculation(3.14, 2.0)
        print(f"Secure calculation result: {result}")

        # Test validation
        validated_array = sanitizer.validate_array_input([1, 2, 3], expected_shape=(3,))
        print(f"Validated array: {validated_array}")

        # Test security status
        status = get_security_status()
        print(f"Security status: {status['config']['validation_level']}")

        print("Security validation system ready!")

    except Exception as e:
        print(f"Security test failed: {e}")
