"""Robust environment wrapper with comprehensive safety and monitoring."""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .error_handling import EnvironmentError, safe_execute
from .monitoring import HealthMonitor, MetricsCollector
from .security_validation import SecurityValidator
from .validation import validate_action, validate_observation

logger = logging.getLogger(__name__)


class RobustEnvironmentWrapper(gym.Wrapper):
    """Robust environment wrapper with safety, monitoring and error recovery."""

    def __init__(
        self,
        env: gym.Env,
        enable_monitoring: bool = True,
        enable_security: bool = True,
        enable_validation: bool = True,
        enable_recovery: bool = True,
        max_recovery_attempts: int = 3,
        timeout_seconds: float = 10.0
    ):
        """Initialize robust environment wrapper.
        
        Args:
            env: Base environment to wrap
            enable_monitoring: Enable performance and health monitoring
            enable_security: Enable security validation
            enable_validation: Enable input/output validation
            enable_recovery: Enable automatic error recovery
            max_recovery_attempts: Maximum recovery attempts
            timeout_seconds: Operation timeout in seconds
        """
        super().__init__(env)

        self.enable_monitoring = enable_monitoring
        self.enable_security = enable_security
        self.enable_validation = enable_validation
        self.enable_recovery = enable_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.timeout_seconds = timeout_seconds

        # Initialize subsystems
        if self.enable_monitoring:
            self.metrics = MetricsCollector()
            self.health_monitor = HealthMonitor()

        if self.enable_security:
            self.security_validator = SecurityValidator()

        # Environment state tracking
        self.last_valid_state = None
        self.episode_step = 0
        self.total_episodes = 0

        # Statistics
        self.stats = {
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'recovery_attempts': 0,
            'security_violations': 0,
            'validation_errors': 0,
            'timeouts': 0,
            'average_step_time': 0.0
        }

        logger.info(f"Initialized RobustEnvironmentWrapper for {env.__class__.__name__}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with robust error handling."""
        reset_start = time.time()

        if self.enable_monitoring:
            self.metrics.increment('env_resets')
            self.health_monitor.start_operation('reset')

        try:
            # Security validation
            if self.enable_security and options:
                self._validate_security(options, 'reset_options')

            # Attempt reset with recovery
            obs, info = self._reset_with_recovery(seed, options)

            # Validate output
            if self.enable_validation:
                obs = self._validate_and_sanitize_observation(obs)
                info = self._validate_and_sanitize_info(info)

            # Update state tracking
            self.last_valid_state = obs.copy() if isinstance(obs, np.ndarray) else obs
            self.episode_step = 0
            self.total_episodes += 1

            # Update timing stats
            reset_time = time.time() - reset_start
            self._update_timing_stats('reset', reset_time)

            if self.enable_monitoring:
                self.metrics.record('reset_time', reset_time)
                self.health_monitor.end_operation('reset', success=True)

            logger.debug(f"Environment reset successful in {reset_time:.4f}s")
            return obs, info

        except Exception as e:
            self.stats['failed_steps'] += 1

            if self.enable_monitoring:
                self.metrics.increment('reset_errors')
                self.health_monitor.end_operation('reset', success=False)

            logger.error(f"Environment reset failed: {e}")

            # Return safe fallback
            return self._create_fallback_reset()

    def step(
        self,
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with robust error handling."""
        step_start = time.time()
        self.stats['total_steps'] += 1

        if self.enable_monitoring:
            self.metrics.increment('env_steps')
            self.health_monitor.start_operation('step')

        try:
            # Input validation and sanitization
            if self.enable_validation:
                action = self._validate_and_sanitize_action(action)

            # Security validation
            if self.enable_security:
                self._validate_security(action, 'action')

            # Attempt step with timeout and recovery
            obs, reward, terminated, truncated, info = self._step_with_recovery(action)

            # Output validation and sanitization
            if self.enable_validation:
                obs = self._validate_and_sanitize_observation(obs)
                reward = self._validate_and_sanitize_reward(reward)
                info = self._validate_and_sanitize_info(info)

            # Update state tracking
            if isinstance(obs, np.ndarray):
                self.last_valid_state = obs.copy()
            self.episode_step += 1
            self.stats['successful_steps'] += 1

            # Update timing stats
            step_time = time.time() - step_start
            self._update_timing_stats('step', step_time)

            if self.enable_monitoring:
                self.metrics.record('step_time', step_time)
                self.metrics.record('reward', reward)
                self.health_monitor.end_operation('step', success=True)

            return obs, reward, terminated, truncated, info

        except Exception as e:
            self.stats['failed_steps'] += 1

            if self.enable_monitoring:
                self.metrics.increment('step_errors')
                self.health_monitor.end_operation('step', success=False)

            logger.error(f"Environment step failed: {e}")

            # Return safe fallback
            return self._create_fallback_step()

    def _reset_with_recovery(
        self,
        seed: Optional[int],
        options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with recovery attempts."""
        last_error = None

        for attempt in range(self.max_recovery_attempts + 1):
            try:
                # Attempt reset with timeout
                result = self._execute_with_timeout(
                    self.env.reset,
                    self.timeout_seconds,
                    seed=seed,
                    options=options
                )

                if result is not None:
                    return result

            except Exception as e:
                last_error = e

                if attempt < self.max_recovery_attempts:
                    self.stats['recovery_attempts'] += 1
                    logger.warning(f"Reset attempt {attempt + 1} failed: {e}. Retrying...")

                    # Wait before retry
                    time.sleep(0.1 * (attempt + 1))

        # All attempts failed
        error_msg = f"All reset attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

    def _step_with_recovery(
        self,
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step with recovery attempts."""
        last_error = None

        for attempt in range(self.max_recovery_attempts + 1):
            try:
                # Attempt step with timeout
                result = self._execute_with_timeout(
                    self.env.step,
                    self.timeout_seconds,
                    action
                )

                if result is not None:
                    return result

            except Exception as e:
                last_error = e

                if attempt < self.max_recovery_attempts and self.enable_recovery:
                    self.stats['recovery_attempts'] += 1
                    logger.warning(f"Step attempt {attempt + 1} failed: {e}. Retrying...")

                    # Try to recover environment state
                    self._attempt_state_recovery()

                    # Wait before retry
                    time.sleep(0.1 * (attempt + 1))

        # All attempts failed
        error_msg = f"All step attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

    def _execute_with_timeout(self, func, timeout: float, *args, **kwargs):
        """Execute function with timeout protection."""
        try:
            start_time = time.time()
            result = safe_execute(func, *args, **kwargs)

            # Check for timeout
            if time.time() - start_time > timeout:
                self.stats['timeouts'] += 1
                logger.warning(f"Operation timed out after {timeout}s")
                return None

            return result

        except Exception as e:
            logger.warning(f"Function execution failed: {e}")
            raise e

    def _attempt_state_recovery(self) -> None:
        """Attempt to recover environment to last valid state."""
        if self.last_valid_state is None:
            logger.warning("No valid state available for recovery")
            return

        try:
            # Try to reset to last valid configuration
            if hasattr(self.env, '_set_state') and callable(self.env._set_state):
                self.env._set_state(self.last_valid_state)
                logger.info("Environment state recovered")
            else:
                # Fallback: full reset
                logger.info("Full environment reset for recovery")
                self.env.reset()

        except Exception as e:
            logger.error(f"State recovery failed: {e}")

    def _validate_security(self, data: Any, context: str) -> None:
        """Validate data for security issues."""
        try:
            self.security_validator.validate(data, context)
        except Exception as e:
            self.stats['security_violations'] += 1
            logger.error(f"Security violation in {context}: {e}")
            raise e

    def _validate_and_sanitize_action(self, action: Any) -> Any:
        """Validate and sanitize action input."""
        try:
            return validate_action(action, self.action_space)
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.warning(f"Action validation failed: {e}")

            # Return safe default action
            if isinstance(self.action_space, gym.spaces.Box):
                return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
            elif isinstance(self.action_space, gym.spaces.Discrete):
                return 0
            else:
                raise EnvironmentError(f"Cannot create safe action for space {self.action_space}")

    def _validate_and_sanitize_observation(self, obs: Any) -> Any:
        """Validate and sanitize observation output."""
        try:
            return validate_observation(obs, self.observation_space)
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.warning(f"Observation validation failed: {e}")

            # Use last valid state if available
            if self.last_valid_state is not None:
                return self.last_valid_state.copy()

            # Create safe default observation
            if isinstance(self.observation_space, gym.spaces.Box):
                return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            else:
                raise EnvironmentError(f"Cannot create safe observation for space {self.observation_space}")

    def _validate_and_sanitize_reward(self, reward: float) -> float:
        """Validate and sanitize reward output."""
        if not isinstance(reward, (int, float)):
            logger.warning(f"Invalid reward type: {type(reward)}. Converting to float.")
            try:
                reward = float(reward)
            except (ValueError, TypeError):
                reward = 0.0

        # Check for invalid values
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"Invalid reward value: {reward}. Using 0.0")
            reward = 0.0

        # Clamp to reasonable range
        if abs(reward) > 1e6:
            logger.warning(f"Extreme reward value: {reward}. Clamping.")
            reward = np.sign(reward) * 1e6

        return float(reward)

    def _validate_and_sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize info dictionary."""
        if not isinstance(info, dict):
            logger.warning(f"Invalid info type: {type(info)}. Using empty dict.")
            return {}

        # Remove potentially dangerous keys
        safe_info = {}
        for key, value in info.items():
            if isinstance(key, str) and key.isalnum() and len(key) < 100:
                safe_info[key] = value
            else:
                logger.warning(f"Removing unsafe info key: {key}")

        return safe_info

    def _create_fallback_reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create safe fallback reset result."""
        logger.warning("Using fallback reset result")

        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        else:
            obs = self.observation_space.sample()

        info = {'fallback_reset': True, 'error': 'Reset failed, using fallback'}

        self.last_valid_state = obs.copy() if isinstance(obs, np.ndarray) else obs
        self.episode_step = 0

        return obs, info

    def _create_fallback_step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Create safe fallback step result."""
        logger.warning("Using fallback step result")

        # Use last valid state if available
        if self.last_valid_state is not None:
            obs = self.last_valid_state.copy()
        else:
            if isinstance(self.observation_space, gym.spaces.Box):
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            else:
                obs = self.observation_space.sample()

        # Safe fallback values
        reward = 0.0
        terminated = True  # End episode on failure
        truncated = False
        info = {'fallback_step': True, 'error': 'Step failed, using fallback'}

        return obs, reward, terminated, truncated, info

    def _update_timing_stats(self, operation: str, duration: float) -> None:
        """Update timing statistics."""
        if operation == 'step':
            total_steps = self.stats['total_steps']
            current_avg = self.stats['average_step_time']

            # Running average
            self.stats['average_step_time'] = (
                (current_avg * (total_steps - 1) + duration) / total_steps
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment wrapper statistics."""
        stats = self.stats.copy()

        if stats['total_steps'] > 0:
            stats['success_rate'] = stats['successful_steps'] / stats['total_steps']
            stats['failure_rate'] = stats['failed_steps'] / stats['total_steps']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0

        # Add monitoring data
        if self.enable_monitoring:
            stats['health_status'] = self.health_monitor.get_status()
            stats['metrics'] = self.metrics.get_metrics()

        stats.update({
            'total_episodes': self.total_episodes,
            'current_episode_step': self.episode_step,
            'recovery_rate': stats['recovery_attempts'] / max(stats['total_steps'], 1),
            'security_violation_rate': stats['security_violations'] / max(stats['total_steps'], 1),
            'validation_error_rate': stats['validation_errors'] / max(stats['total_steps'], 1)
        })

        return stats

    def reset_statistics(self) -> None:
        """Reset wrapper statistics."""
        self.stats = {
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'recovery_attempts': 0,
            'security_violations': 0,
            'validation_errors': 0,
            'timeouts': 0,
            'average_step_time': 0.0
        }

        if self.enable_monitoring:
            self.metrics.reset()
            self.health_monitor.reset()

        logger.info("Environment wrapper statistics reset")
