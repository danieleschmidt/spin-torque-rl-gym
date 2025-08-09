"""Monitoring and logging utilities for SpinTorque Gym environments.

This module provides comprehensive monitoring, logging, and health checking
capabilities for robust operation of spintronic device environments.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for environment monitoring."""
    total_steps: int = 0
    total_episodes: int = 0
    total_runtime: float = 0.0
    avg_step_time: float = 0.0
    avg_episode_length: float = 0.0
    success_rate: float = 0.0
    solver_timeout_rate: float = 0.0
    error_count: int = 0
    warning_count: int = 0


class EnvironmentMonitor:
    """Monitor environment performance and health."""

    def __init__(
        self,
        max_history: int = 1000,
        performance_window: int = 100,
        log_level: str = "INFO"
    ):
        """Initialize environment monitor.
        
        Args:
            max_history: Maximum number of historical records to keep
            performance_window: Window size for rolling performance metrics
            log_level: Logging level
        """
        self.max_history = max_history
        self.performance_window = performance_window

        # Setup logging
        self.logger = logging.getLogger("SpinTorqueGym")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Performance tracking
        self.step_times = deque(maxlen=performance_window)
        self.episode_lengths = deque(maxlen=performance_window)
        self.episode_rewards = deque(maxlen=performance_window)
        self.success_flags = deque(maxlen=performance_window)
        self.error_history = deque(maxlen=max_history)

        # Counters
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()

        # Health check thresholds
        self.thresholds = {
            'max_step_time': 1.0,  # seconds
            'max_solver_timeout_rate': 0.5,  # 50%
            'min_success_rate': 0.1,  # 10%
            'max_error_rate': 0.1,  # 10%
        }

        # Episode timing
        self.episode_start_time = None
        self.step_start_time = None

    def start_episode(self) -> None:
        """Mark the start of a new episode."""
        self.episode_start_time = time.time()
        self.logger.debug("Starting new episode")

    def end_episode(self, episode_reward: float, success: bool) -> None:
        """Mark the end of an episode.
        
        Args:
            episode_reward: Total episode reward
            success: Whether episode was successful
        """
        if self.episode_start_time is None:
            self.logger.warning("end_episode called without start_episode")
            return

        episode_time = time.time() - self.episode_start_time

        # Update metrics
        self.metrics.total_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.success_flags.append(success)

        # Update rolling averages
        if len(self.success_flags) > 0:
            self.metrics.success_rate = sum(self.success_flags) / len(self.success_flags)

        self.logger.debug(f"Episode ended: reward={episode_reward:.3f}, success={success}, time={episode_time:.3f}s")

        # Health check
        self._check_episode_health(episode_time, episode_reward, success)

        self.episode_start_time = None

    def start_step(self) -> None:
        """Mark the start of a step."""
        self.step_start_time = time.time()

    def end_step(self, reward: float, info: Dict[str, Any]) -> None:
        """Mark the end of a step.
        
        Args:
            reward: Step reward
            info: Step info dictionary
        """
        if self.step_start_time is None:
            self.logger.warning("end_step called without start_step")
            return

        step_time = time.time() - self.step_start_time

        # Update metrics
        self.metrics.total_steps += 1
        self.step_times.append(step_time)

        if len(self.step_times) > 0:
            self.metrics.avg_step_time = sum(self.step_times) / len(self.step_times)

        # Check for solver issues
        if info.get('simulation_success') is False:
            self._log_solver_issue(info)

        # Health check
        self._check_step_health(step_time, reward, info)

        self.step_start_time = None

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }

        self.error_history.append(error_info)
        self.metrics.error_count += 1

        self.logger.error(f"Error in {context}: {error}")

    def log_warning(self, message: str, context: str = "") -> None:
        """Log a warning with context.
        
        Args:
            message: Warning message
            context: Additional context information
        """
        self.metrics.warning_count += 1
        self.logger.warning(f"{context}: {message}" if context else message)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        current_time = time.time()
        self.metrics.total_runtime = current_time - self.start_time

        # Calculate rates
        error_rate = self.metrics.error_count / max(self.metrics.total_steps, 1)
        solver_issues = sum(1 for info in self.error_history if 'solver' in info.get('context', '').lower())
        solver_timeout_rate = solver_issues / max(self.metrics.total_steps, 1)

        # Health status
        health_issues = []
        if self.metrics.avg_step_time > self.thresholds['max_step_time']:
            health_issues.append(f"High step time: {self.metrics.avg_step_time:.3f}s")

        if solver_timeout_rate > self.thresholds['max_solver_timeout_rate']:
            health_issues.append(f"High solver timeout rate: {solver_timeout_rate:.2%}")

        if self.metrics.success_rate < self.thresholds['min_success_rate'] and self.metrics.total_episodes > 10:
            health_issues.append(f"Low success rate: {self.metrics.success_rate:.2%}")

        if error_rate > self.thresholds['max_error_rate']:
            health_issues.append(f"High error rate: {error_rate:.2%}")

        health_status = "HEALTHY" if not health_issues else "WARNING" if len(health_issues) < 3 else "CRITICAL"

        return {
            'timestamp': current_time,
            'health_status': health_status,
            'health_issues': health_issues,
            'performance_metrics': {
                'total_steps': self.metrics.total_steps,
                'total_episodes': self.metrics.total_episodes,
                'total_runtime': self.metrics.total_runtime,
                'avg_step_time': self.metrics.avg_step_time,
                'avg_episode_length': self.metrics.total_steps / max(self.metrics.total_episodes, 1),
                'success_rate': self.metrics.success_rate,
                'error_rate': error_rate,
                'solver_timeout_rate': solver_timeout_rate
            },
            'recent_performance': {
                'recent_rewards': list(self.episode_rewards)[-10:] if self.episode_rewards else [],
                'recent_step_times': list(self.step_times)[-10:] if self.step_times else [],
                'recent_errors': list(self.error_history)[-5:] if self.error_history else []
            }
        }

    def _check_step_health(self, step_time: float, reward: float, info: Dict[str, Any]) -> None:
        """Check step-level health indicators."""
        if step_time > self.thresholds['max_step_time']:
            self.log_warning(f"Slow step: {step_time:.3f}s", "performance")

        if np.isnan(reward) or np.isinf(reward):
            self.log_warning(f"Invalid reward: {reward}", "reward")

        if 'simulation_success' in info and not info['simulation_success']:
            self.log_warning("Simulation failed", "solver")

    def _check_episode_health(self, episode_time: float, episode_reward: float, success: bool) -> None:
        """Check episode-level health indicators."""
        if np.isnan(episode_reward) or np.isinf(episode_reward):
            self.log_warning(f"Invalid episode reward: {episode_reward}", "reward")

        if episode_time > 60:  # 1 minute
            self.log_warning(f"Very long episode: {episode_time:.1f}s", "performance")

    def _log_solver_issue(self, info: Dict[str, Any]) -> None:
        """Log solver-specific issues."""
        message = info.get('message', 'Unknown solver issue')
        self.log_warning(f"Solver issue: {message}", "solver")

    def reset(self) -> None:
        """Reset monitoring state."""
        self.step_times.clear()
        self.episode_lengths.clear()
        self.episode_rewards.clear()
        self.success_flags.clear()
        self.error_history.clear()

        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        self.episode_start_time = None
        self.step_start_time = None

        self.logger.info("Monitor reset")


class SafetyWrapper:
    """Safety wrapper for environment operations."""

    def __init__(self, monitor: EnvironmentMonitor):
        """Initialize safety wrapper.
        
        Args:
            monitor: Environment monitor instance
        """
        self.monitor = monitor
        self.safety_limits = {
            'max_current': 1e8,  # A/m²
            'max_duration': 1e-6,  # 1 μs
            'max_temperature': 1000,  # K
            'min_temperature': 0,  # K
        }

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clamp action to safe ranges.
        
        Args:
            action: Raw action from agent
            
        Returns:
            Validated and clamped action
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        if action.shape != (2,):
            self.monitor.log_warning(f"Invalid action shape: {action.shape}", "safety")
            action = np.array([0.0, 1e-12], dtype=np.float32)

        # Clamp current density
        action[0] = np.clip(action[0], -self.safety_limits['max_current'], self.safety_limits['max_current'])

        # Clamp duration (must be positive)
        action[1] = np.clip(action[1], 1e-12, self.safety_limits['max_duration'])

        # Check for NaN/Inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.monitor.log_warning("NaN/Inf detected in action", "safety")
            action = np.array([0.0, 1e-12], dtype=np.float32)

        return action

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        """Validate observation for safety.
        
        Args:
            observation: Raw observation
            
        Returns:
            Validated observation
        """
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            self.monitor.log_warning("NaN/Inf detected in observation", "safety")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def validate_reward(self, reward: float) -> float:
        """Validate reward for safety.
        
        Args:
            reward: Raw reward
            
        Returns:
            Validated reward
        """
        if np.isnan(reward) or np.isinf(reward):
            self.monitor.log_warning(f"Invalid reward: {reward}", "safety")
            reward = -1.0  # Penalty for invalid state

        # Clamp extreme rewards
        reward = np.clip(reward, -1e6, 1e6)

        return reward
