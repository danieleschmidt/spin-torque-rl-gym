"""Logging configuration for SpinTorque Gym.

This module provides centralized logging configuration with structured
logging, performance metrics, and monitoring capabilities.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'physics_params'):
            log_entry['physics_params'] = record.physics_params
        if hasattr(record, 'device_type'):
            log_entry['device_type'] = record.device_type
        if hasattr(record, 'step_count'):
            log_entry['step_count'] = record.step_count
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        if hasattr(record, 'memory_usage'):
            log_entry['memory_usage'] = record.memory_usage

        # Include exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""

    def __init__(self, logger_name: str = "SpinTorqueGym.Performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times = {}

    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()

    def end_timing(self, operation: str, **kwargs) -> float:
        """End timing an operation and log the duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time recorded for operation: {operation}")
            return 0.0

        duration = time.perf_counter() - self.start_times[operation]
        del self.start_times[operation]

        # Log with structured data
        extra = {'execution_time': duration, 'operation': operation}
        extra.update(kwargs)

        if duration > 1.0:  # Log slow operations
            self.logger.warning(
                f"Slow operation: {operation} took {duration:.3f}s",
                extra=extra
            )
        elif duration > 0.1:
            self.logger.info(
                f"Operation {operation} completed in {duration:.3f}s",
                extra=extra
            )
        else:
            self.logger.debug(
                f"Operation {operation} completed in {duration:.3f}s",
                extra=extra
            )

        return duration

    def log_memory_usage(self, context: str = "general") -> Optional[Dict[str, float]]:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            memory_data = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'context': context
            }

            self.logger.info(
                f"Memory usage - RSS: {memory_data['rss_mb']:.1f}MB, "
                f"VMS: {memory_data['vms_mb']:.1f}MB, "
                f"Percent: {memory_data['percent']:.1f}%",
                extra={'memory_usage': memory_data}
            )

            return memory_data
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
            return None


class SecurityLogger:
    """Logger for security events and audit trail."""

    def __init__(self, logger_name: str = "SpinTorqueGym.Security"):
        self.logger = logging.getLogger(logger_name)

    def log_input_validation(self, input_type: str, validation_result: str,
                           **kwargs) -> None:
        """Log input validation events."""
        extra = {
            'input_type': input_type,
            'validation_result': validation_result,
            'security_event': True
        }
        extra.update(kwargs)

        if validation_result == 'failed':
            self.logger.warning(
                f"Input validation failed for {input_type}",
                extra=extra
            )
        else:
            self.logger.debug(
                f"Input validation passed for {input_type}",
                extra=extra
            )

    def log_safety_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Log safety constraint violations."""
        extra = {
            'violation_type': violation_type,
            'safety_violation': True,
            'details': details
        }

        self.logger.error(
            f"Safety violation: {violation_type}",
            extra=extra
        )

    def log_parameter_sanitization(self, parameter: str,
                                 original_value: Any,
                                 sanitized_value: Any) -> None:
        """Log parameter sanitization."""
        extra = {
            'parameter': parameter,
            'original_value': str(original_value),
            'sanitized_value': str(sanitized_value),
            'security_event': True
        }

        self.logger.info(
            f"Parameter sanitized: {parameter}",
            extra=extra
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    console_output: bool = True
) -> None:
    """Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Use structured JSON logging
        console_output: Enable console output
    """

    # Convert log level string to constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Setup specific loggers
    setup_physics_logger(numeric_level)
    setup_environment_logger(numeric_level)
    setup_device_logger(numeric_level)


def setup_physics_logger(level: int) -> None:
    """Setup physics simulation logger."""
    physics_logger = logging.getLogger("SpinTorqueGym.Physics")
    physics_logger.setLevel(level)

    # Add physics-specific handling if needed
    if level <= logging.DEBUG:
        physics_logger.info("Physics debug logging enabled")


def setup_environment_logger(level: int) -> None:
    """Setup environment logger."""
    env_logger = logging.getLogger("SpinTorqueGym.Environment")
    env_logger.setLevel(level)

    if level <= logging.DEBUG:
        env_logger.info("Environment debug logging enabled")


def setup_device_logger(level: int) -> None:
    """Setup device model logger."""
    device_logger = logging.getLogger("SpinTorqueGym.Device")
    device_logger.setLevel(level)

    if level <= logging.DEBUG:
        device_logger.info("Device debug logging enabled")


class LoggingContext:
    """Context manager for structured logging with timing."""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation}", extra=self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time

        extra = dict(self.kwargs)
        extra.update({'execution_time': duration, 'operation': self.operation})

        if exc_type is not None:
            self.logger.error(
                f"Failed {self.operation} after {duration:.3f}s: {exc_val}",
                extra=extra
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {duration:.3f}s",
                extra=extra
            )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with standardized naming.
    
    Args:
        name: Logger name (will be prefixed with SpinTorqueGym)
        
    Returns:
        Configured logger instance
    """
    if not name.startswith('SpinTorqueGym'):
        name = f'SpinTorqueGym.{name}'
    return logging.getLogger(name)


# Initialize default logging if not already configured
if not logging.root.handlers:
    setup_logging(
        log_level=os.getenv('SPINTORQUE_LOG_LEVEL', 'WARNING'),
        log_file=os.getenv('SPINTORQUE_LOG_FILE'),
        structured=os.getenv('SPINTORQUE_STRUCTURED_LOGS', '').lower() == 'true'
    )
