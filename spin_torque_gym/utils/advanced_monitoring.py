"""Advanced monitoring and observability system for Spin Torque RL-Gym.

This module provides comprehensive monitoring, metrics collection, health checks,
and observability features for scientific computing applications.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Union


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = None
    unit: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricsCollector:
    """Thread-safe metrics collection system."""

    def __init__(self, max_history: int = 1000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self._lock = threading.RLock()

    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self.record_metric(MetricValue(name, self.counters[name], time.time(), tags, "count"))

    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self.record_metric(MetricValue(name, value, time.time(), tags, "gauge"))

    def record_histogram(self, name: str, value: Union[int, float], tags: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]  # Keep half
            self.record_metric(MetricValue(name, value, time.time(), tags, "histogram"))

    def record_metric(self, metric: MetricValue) -> None:
        """Record a generic metric."""
        with self._lock:
            self.metrics[metric.name].append(metric)

    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}

            values = [m.value for m in self.metrics[name]]
            if not values:
                return {}

            return {
                'name': name,
                'count': len(values),
                'latest': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'latest_timestamp': self.metrics[name][-1].timestamp if values else None
            }

    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        with self._lock:
            return {name: self.get_metric_summary(name) for name in self.metrics.keys()}


class HealthChecker:
    """System health monitoring with customizable checks."""

    def __init__(self):
        self.checks = {}
        self.check_history = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.RLock()

    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a health check function."""
        with self._lock:
            self.checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        with self._lock:
            if name not in self.checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check '{name}' not registered"
                )

            try:
                result = self.checks[name]()
                self.check_history[name].append(result)
                return result
            except Exception as e:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {e}",
                    details={'exception': str(e)}
                )
                self.check_history[name].append(error_result)
                return error_result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        all_results = self.run_all_checks()
        if not all_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in all_results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


class PerformanceProfiler:
    """Performance profiling and timing utilities."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_timers = {}
        self._lock = threading.RLock()

    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{time.time()}_{id(threading.current_thread())}"
        with self._lock:
            self.active_timers[timer_id] = {
                'operation': operation,
                'start_time': time.time(),
                'thread_id': threading.current_thread().ident
            }
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """End timing and record metric."""
        with self._lock:
            if timer_id not in self.active_timers:
                logging.warning(f"Timer {timer_id} not found")
                return 0.0

            timer_info = self.active_timers.pop(timer_id)
            duration = time.time() - timer_info['start_time']

            # Record timing metric
            self.metrics.record_histogram(
                f"operation_duration_{timer_info['operation']}",
                duration,
                {'operation': timer_info['operation']}
            )

            return duration

    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation_name)


class TimingContext:
    """Context manager for operation timing."""

    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.timer_id = None

    def __enter__(self):
        self.timer_id = self.profiler.start_timer(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            duration = self.profiler.end_timer(self.timer_id)
            if exc_type is not None:
                logging.warning(f"Operation {self.operation_name} failed after {duration:.3f}s")


class SystemMonitor:
    """Comprehensive system monitoring coordination."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker()
        self.profiler = PerformanceProfiler(self.metrics)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30.0  # seconds
        self._setup_default_health_checks()

    def _setup_default_health_checks(self):
        """Set up default system health checks."""

        def memory_check() -> HealthCheckResult:
            """Check memory usage."""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent

                if memory_percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"High memory usage: {memory_percent:.1f}%"
                elif memory_percent > 80:
                    status = HealthStatus.WARNING
                    message = f"Moderate memory usage: {memory_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage OK: {memory_percent:.1f}%"

                return HealthCheckResult(
                    name="memory_usage",
                    status=status,
                    message=message,
                    details={'memory_percent': memory_percent}
                )
            except ImportError:
                # Fallback without psutil
                return HealthCheckResult(
                    name="memory_usage",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available for memory monitoring"
                )

        def disk_space_check() -> HealthCheckResult:
            """Check disk space."""
            try:
                import shutil
                total, used, free = shutil.disk_usage('/')
                free_percent = (free / total) * 100

                if free_percent < 5:
                    status = HealthStatus.CRITICAL
                    message = f"Low disk space: {free_percent:.1f}% free"
                elif free_percent < 15:
                    status = HealthStatus.WARNING
                    message = f"Moderate disk space: {free_percent:.1f}% free"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Disk space OK: {free_percent:.1f}% free"

                return HealthCheckResult(
                    name="disk_space",
                    status=status,
                    message=message,
                    details={
                        'total_gb': total / (1024**3),
                        'used_gb': used / (1024**3),
                        'free_gb': free / (1024**3),
                        'free_percent': free_percent
                    }
                )
            except Exception as e:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.UNKNOWN,
                    message=f"Disk space check failed: {e}"
                )

        def thread_count_check() -> HealthCheckResult:
            """Check active thread count."""
            try:
                thread_count = threading.active_count()

                if thread_count > 50:
                    status = HealthStatus.WARNING
                    message = f"High thread count: {thread_count}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Thread count OK: {thread_count}"

                return HealthCheckResult(
                    name="thread_count",
                    status=status,
                    message=message,
                    details={'thread_count': thread_count}
                )
            except Exception as e:
                return HealthCheckResult(
                    name="thread_count",
                    status=HealthStatus.UNKNOWN,
                    message=f"Thread count check failed: {e}"
                )

        # Register default checks
        self.health_checker.register_check("memory_usage", memory_check)
        self.health_checker.register_check("disk_space", disk_space_check)
        self.health_checker.register_check("thread_count", thread_count_check)

    def start_monitoring(self, interval: float = 30.0):
        """Start background monitoring."""
        if self.monitoring_active:
            logging.warning("Monitoring already active")
            return

        self.monitoring_interval = interval
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info(f"Started system monitoring with {interval}s interval")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logging.info("Stopped system monitoring")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                health_results = self.health_checker.run_all_checks()

                # Record health status as metrics
                for name, result in health_results.items():
                    status_value = {
                        HealthStatus.HEALTHY: 1,
                        HealthStatus.WARNING: 2,
                        HealthStatus.CRITICAL: 3,
                        HealthStatus.UNKNOWN: 0
                    }.get(result.status, 0)

                    self.metrics.set_gauge(
                        f"health_status_{name}",
                        status_value,
                        {'check_name': name, 'status': result.status.value}
                    )

                # Record system uptime
                uptime = time.time() - getattr(self, '_start_time', time.time())
                self.metrics.set_gauge('system_uptime_seconds', uptime)

                # Sleep until next check
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(min(self.monitoring_interval, 10.0))  # Shorter retry interval

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        health_results = self.health_checker.run_all_checks()
        metrics_summary = self.metrics.get_all_metrics_summary()
        overall_health = self.health_checker.get_overall_health()

        return {
            'timestamp': time.time(),
            'overall_health': overall_health.value,
            'health_checks': {name: asdict(result) for name, result in health_results.items()},
            'metrics_summary': metrics_summary,
            'monitoring_active': self.monitoring_active,
            'monitoring_interval': self.monitoring_interval
        }

    def export_metrics_json(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        try:
            status_report = self.get_status_report()

            with open(filepath, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)

            logging.info(f"Metrics exported to {filepath}")

        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")

    def register_physics_metrics(self):
        """Register physics-specific monitoring."""

        def physics_stability_check() -> HealthCheckResult:
            """Check physics simulation stability."""
            # Check recent physics error metrics
            physics_errors = self.metrics.get_metric_summary('physics_errors')

            if not physics_errors:
                return HealthCheckResult(
                    name="physics_stability",
                    status=HealthStatus.HEALTHY,
                    message="No physics errors recorded"
                )

            error_rate = physics_errors.get('count', 0) / max(physics_errors.get('count', 1), 100)

            if error_rate > 0.05:  # 5% error rate
                status = HealthStatus.CRITICAL
                message = f"High physics error rate: {error_rate:.1%}"
            elif error_rate > 0.01:  # 1% error rate
                status = HealthStatus.WARNING
                message = f"Moderate physics error rate: {error_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Physics error rate OK: {error_rate:.1%}"

            return HealthCheckResult(
                name="physics_stability",
                status=status,
                message=message,
                details={'error_rate': error_rate, 'error_count': physics_errors.get('count', 0)}
            )

        def magnetization_bounds_check() -> HealthCheckResult:
            """Check magnetization vector bounds."""
            mag_summary = self.metrics.get_metric_summary('magnetization_magnitude')

            if not mag_summary:
                return HealthCheckResult(
                    name="magnetization_bounds",
                    status=HealthStatus.UNKNOWN,
                    message="No magnetization data available"
                )

            min_mag = mag_summary.get('min', 1.0)
            max_mag = mag_summary.get('max', 1.0)

            if min_mag < 0.9 or max_mag > 1.1:
                status = HealthStatus.WARNING
                message = f"Magnetization bounds violated: [{min_mag:.3f}, {max_mag:.3f}]"
            else:
                status = HealthStatus.HEALTHY
                message = f"Magnetization bounds OK: [{min_mag:.3f}, {max_mag:.3f}]"

            return HealthCheckResult(
                name="magnetization_bounds",
                status=status,
                message=message,
                details={'min_magnitude': min_mag, 'max_magnitude': max_mag}
            )

        # Register physics-specific checks
        self.health_checker.register_check("physics_stability", physics_stability_check)
        self.health_checker.register_check("magnetization_bounds", magnetization_bounds_check)


# Global monitor instance
system_monitor = SystemMonitor()


def initialize_monitoring(start_background: bool = True, interval: float = 30.0) -> SystemMonitor:
    """Initialize the global monitoring system.
    
    Args:
        start_background: Whether to start background monitoring
        interval: Monitoring interval in seconds
        
    Returns:
        Configured system monitor
    """
    system_monitor._start_time = time.time()
    system_monitor.register_physics_metrics()

    if start_background:
        system_monitor.start_monitoring(interval)

    logging.info("Advanced monitoring system initialized")
    return system_monitor


def get_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return system_monitor


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize monitoring
    monitor = initialize_monitoring(start_background=False)

    # Demonstrate metrics collection
    monitor.metrics.increment_counter("test_counter")
    monitor.metrics.set_gauge("test_gauge", 42.0)
    monitor.metrics.record_histogram("test_histogram", 3.14)

    # Demonstrate performance profiling
    with monitor.profiler.time_operation("test_operation"):
        time.sleep(0.1)  # Simulate work

    # Run health checks
    health_results = monitor.health_checker.run_all_checks()
    for name, result in health_results.items():
        print(f"Health Check '{name}': {result.status.value} - {result.message}")

    # Get status report
    status = monitor.get_status_report()
    print(f"Overall Health: {status['overall_health']}")
    print(f"Metrics Count: {len(status['metrics_summary'])}")

    print("Advanced monitoring system ready!")
