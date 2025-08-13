"""Health check and monitoring utilities.

This module provides comprehensive health monitoring for the RL environment,
including physics simulation health, device state monitoring, and system health.
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, Tuple

import numpy as np

from .logging_config import PerformanceLogger, get_logger


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str, warning_threshold: float = 0.8,
                 critical_threshold: float = 0.95):
        """Initialize health check.
        
        Args:
            name: Health check name
            warning_threshold: Warning threshold (0-1)
            critical_threshold: Critical threshold (0-1)
        """
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_check_time = 0.0
        self.status = "UNKNOWN"
        self.message = ""
        self.value = 0.0

    def check(self) -> Tuple[str, float, str]:
        """Perform health check.
        
        Returns:
            (status, value, message) tuple
        """
        self.last_check_time = time.time()
        status, value, message = self._perform_check()

        self.status = status
        self.value = value
        self.message = message

        return status, value, message

    def _perform_check(self) -> Tuple[str, float, str]:
        """Override in subclasses."""
        raise NotImplementedError


class PhysicsHealthCheck(HealthCheck):
    """Health check for physics simulation."""

    def __init__(self, solver_stats: Dict[str, Any]):
        """Initialize physics health check.
        
        Args:
            solver_stats: Solver statistics dictionary
        """
        super().__init__("Physics Simulation")
        self.solver_stats = solver_stats

    def _perform_check(self) -> Tuple[str, float, str]:
        """Check physics simulation health."""
        issues = []
        score = 1.0

        # Check solver timeout rate
        timeout_rate = self.solver_stats.get('timeout_rate', 0.0)
        if timeout_rate > 0.5:
            score *= 0.3
            issues.append(f"High timeout rate: {timeout_rate:.1%}")
        elif timeout_rate > 0.1:
            score *= 0.7
            issues.append(f"Moderate timeout rate: {timeout_rate:.1%}")

        # Check average solve time
        avg_solve_time = self.solver_stats.get('avg_solve_time', 0.0)
        if avg_solve_time > 1.0:
            score *= 0.4
            issues.append(f"Slow solving: {avg_solve_time:.2f}s average")
        elif avg_solve_time > 0.1:
            score *= 0.8
            issues.append(f"Moderate solving speed: {avg_solve_time:.2f}s average")

        # Check solve success rate
        solve_count = self.solver_stats.get('solve_count', 0)
        timeout_count = self.solver_stats.get('timeout_count', 0)
        if solve_count > 0:
            success_rate = 1.0 - (timeout_count / solve_count)
            if success_rate < 0.5:
                score *= 0.2
                issues.append(f"Low success rate: {success_rate:.1%}")
            elif success_rate < 0.8:
                score *= 0.6
                issues.append(f"Moderate success rate: {success_rate:.1%}")

        # Determine status
        if score >= self.critical_threshold:
            status = "HEALTHY"
            message = "Physics simulation running normally"
        elif score >= self.warning_threshold:
            status = "WARNING"
            message = f"Physics issues: {'; '.join(issues)}"
        else:
            status = "CRITICAL"
            message = f"Critical physics issues: {'; '.join(issues)}"

        return status, score, message


class DeviceHealthCheck(HealthCheck):
    """Health check for device models."""

    def __init__(self, device_info: Dict[str, Any]):
        """Initialize device health check.
        
        Args:
            device_info: Device information dictionary
        """
        super().__init__("Device Models")
        self.device_info = device_info

    def _perform_check(self) -> Tuple[str, float, str]:
        """Check device health."""
        issues = []
        score = 1.0

        # Check for required parameters
        required_params = ['saturation_magnetization', 'volume', 'damping']
        for param in required_params:
            if param not in self.device_info:
                score *= 0.5
                issues.append(f"Missing parameter: {param}")

        # Check parameter ranges
        if 'damping' in self.device_info:
            damping = self.device_info['damping']
            if damping > 1.0 or damping < 0:
                score *= 0.3
                issues.append(f"Invalid damping: {damping}")
            elif damping > 0.5:
                score *= 0.8
                issues.append(f"High damping: {damping}")

        if 'saturation_magnetization' in self.device_info:
            ms = self.device_info['saturation_magnetization']
            if ms <= 0:
                score *= 0.1
                issues.append("Invalid saturation magnetization")
            elif ms > 5e6:  # Very high
                score *= 0.9
                issues.append(f"Very high Ms: {ms:.1e} A/m")

        # Check device type consistency
        device_type = self.device_info.get('device_type', 'Unknown')
        if device_type == 'Unknown':
            score *= 0.7
            issues.append("Unknown device type")

        # Determine status
        if score >= self.critical_threshold:
            status = "HEALTHY"
            message = f"Device ({device_type}) functioning normally"
        elif score >= self.warning_threshold:
            status = "WARNING"
            message = f"Device issues: {'; '.join(issues)}"
        else:
            status = "CRITICAL"
            message = f"Critical device issues: {'; '.join(issues)}"

        return status, score, message


class EnvironmentHealthCheck(HealthCheck):
    """Health check for RL environment."""

    def __init__(self, env_stats: Dict[str, Any]):
        """Initialize environment health check.
        
        Args:
            env_stats: Environment statistics
        """
        super().__init__("RL Environment")
        self.env_stats = env_stats
        self.reward_history = deque(maxlen=100)
        self.step_time_history = deque(maxlen=100)

    def update_stats(self, reward: float, step_time: float) -> None:
        """Update running statistics.
        
        Args:
            reward: Latest reward value
            step_time: Time taken for step
        """
        self.reward_history.append(reward)
        self.step_time_history.append(step_time)

    def _perform_check(self) -> Tuple[str, float, str]:
        """Check environment health."""
        issues = []
        score = 1.0

        # Check step timing
        if self.step_time_history:
            avg_step_time = np.mean(self.step_time_history)
            if avg_step_time > 1.0:
                score *= 0.4
                issues.append(f"Slow steps: {avg_step_time:.2f}s average")
            elif avg_step_time > 0.1:
                score *= 0.8
                issues.append(f"Moderate step time: {avg_step_time:.2f}s")

        # Check reward distribution
        if len(self.reward_history) > 10:
            rewards = np.array(self.reward_history)

            # Check for stuck rewards
            if np.std(rewards) < 1e-10:
                score *= 0.6
                issues.append("Rewards not changing")

            # Check for extreme rewards
            if np.any(np.abs(rewards) > 1e10):
                score *= 0.3
                issues.append("Extreme reward values detected")

            # Check for NaN rewards
            if np.any(np.isnan(rewards)):
                score *= 0.1
                issues.append("NaN rewards detected")

        # Check episode statistics
        total_episodes = self.env_stats.get('total_episodes', 0)
        if total_episodes == 0:
            score *= 0.8
            issues.append("No episodes completed")

        success_rate = self.env_stats.get('success_rate', 0.0)
        if success_rate < 0.01:
            score *= 0.7
            issues.append(f"Low success rate: {success_rate:.1%}")

        # Determine status
        if score >= self.critical_threshold:
            status = "HEALTHY"
            message = "Environment functioning normally"
        elif score >= self.warning_threshold:
            status = "WARNING"
            message = f"Environment issues: {'; '.join(issues)}"
        else:
            status = "CRITICAL"
            message = f"Critical environment issues: {'; '.join(issues)}"

        return status, score, message


class SystemHealthCheck(HealthCheck):
    """Health check for system resources."""

    def __init__(self):
        """Initialize system health check."""
        super().__init__("System Resources")
        self.logger = get_logger("Health.System")

    def _perform_check(self) -> Tuple[str, float, str]:
        """Check system health."""
        issues = []
        score = 1.0

        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                score *= 0.4
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                score *= 0.8
                issues.append(f"Moderate CPU usage: {cpu_percent:.1f}%")

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                score *= 0.3
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            elif memory.percent > 70:
                score *= 0.8
                issues.append(f"Moderate memory usage: {memory.percent:.1f}%")

            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                score *= 0.2
                issues.append(f"Disk almost full: {disk.percent:.1f}%")
            elif disk.percent > 80:
                score *= 0.9
                issues.append(f"High disk usage: {disk.percent:.1f}%")

        except ImportError:
            score *= 0.9
            issues.append("psutil not available for system monitoring")
        except Exception as e:
            score *= 0.7
            issues.append(f"System monitoring error: {e}")

        # Determine status
        if score >= self.critical_threshold:
            status = "HEALTHY"
            message = "System resources normal"
        elif score >= self.warning_threshold:
            status = "WARNING"
            message = f"System issues: {'; '.join(issues)}"
        else:
            status = "CRITICAL"
            message = f"Critical system issues: {'; '.join(issues)}"

        return status, score, message


class HealthMonitor:
    """Comprehensive health monitoring system."""

    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger("Health")
        self.performance_logger = PerformanceLogger()

        self.checks = {}
        self.check_history = defaultdict(deque)
        self.last_full_check = 0.0
        self.check_interval = 10.0  # Check every 10 seconds

        # Initialize system health check
        self.add_check("system", SystemHealthCheck())

    def add_check(self, name: str, health_check: HealthCheck) -> None:
        """Add a health check.
        
        Args:
            name: Check identifier
            health_check: Health check instance
        """
        self.checks[name] = health_check
        self.logger.info(f"Added health check: {name}")

    def run_check(self, name: str) -> Dict[str, Any]:
        """Run specific health check.
        
        Args:
            name: Check name
            
        Returns:
            Check result dictionary
        """
        if name not in self.checks:
            return {
                'name': name,
                'status': 'ERROR',
                'value': 0.0,
                'message': f'Health check "{name}" not found'
            }

        check = self.checks[name]

        try:
            self.performance_logger.start_timing(f"health_check_{name}")
            status, value, message = check.check()
            self.performance_logger.end_timing(f"health_check_{name}")

            result = {
                'name': name,
                'status': status,
                'value': value,
                'message': message,
                'timestamp': check.last_check_time
            }

            # Store in history
            self.check_history[name].append(result)
            if len(self.check_history[name]) > 100:
                self.check_history[name].popleft()

            # Log if not healthy
            if status == "WARNING":
                self.logger.warning(f"Health warning - {name}: {message}")
            elif status == "CRITICAL":
                self.logger.error(f"Health critical - {name}: {message}")

            return result

        except Exception as e:
            error_result = {
                'name': name,
                'status': 'ERROR',
                'value': 0.0,
                'message': f'Health check failed: {e}',
                'timestamp': time.time()
            }

            self.logger.error(f"Health check error - {name}: {e}")
            return error_result

    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of all check results
        """
        self.last_full_check = time.time()
        results = {}

        for name in self.checks:
            results[name] = self.run_check(name)

        return results

    def get_overall_health(self) -> Tuple[str, float, str]:
        """Get overall system health.
        
        Returns:
            (status, score, message) tuple
        """
        if not self.checks:
            return "UNKNOWN", 0.0, "No health checks configured"

        # Run all checks if needed
        if time.time() - self.last_full_check > self.check_interval:
            self.run_all_checks()

        # Calculate overall health
        total_score = 0.0
        critical_issues = []
        warning_issues = []
        check_count = 0

        for name, check in self.checks.items():
            if check.last_check_time > 0:  # Check has been run
                total_score += check.value
                check_count += 1

                if check.status == "CRITICAL":
                    critical_issues.append(f"{name}: {check.message}")
                elif check.status == "WARNING":
                    warning_issues.append(f"{name}: {check.message}")

        if check_count == 0:
            return "UNKNOWN", 0.0, "No health checks have been run"

        avg_score = total_score / check_count

        # Determine overall status
        if critical_issues:
            status = "CRITICAL"
            message = f"Critical issues: {'; '.join(critical_issues[:3])}"
            if len(critical_issues) > 3:
                message += f" (+{len(critical_issues)-3} more)"
        elif warning_issues:
            status = "WARNING"
            message = f"Warnings: {'; '.join(warning_issues[:3])}"
            if len(warning_issues) > 3:
                message += f" (+{len(warning_issues)-3} more)"
        elif avg_score >= 0.95:
            status = "HEALTHY"
            message = "All systems healthy"
        else:
            status = "DEGRADED"
            message = f"Overall health score: {avg_score:.2f}"

        return status, avg_score, message

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        overall_status, overall_score, overall_message = self.get_overall_health()

        individual_checks = {}
        for name in self.checks:
            check = self.checks[name]
            individual_checks[name] = {
                'status': check.status,
                'value': check.value,
                'message': check.message,
                'last_check': check.last_check_time
            }

        return {
            'overall': {
                'status': overall_status,
                'score': overall_score,
                'message': overall_message
            },
            'checks': individual_checks,
            'timestamp': time.time(),
            'check_interval': self.check_interval
        }


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
