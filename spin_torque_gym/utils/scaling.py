"""Auto-scaling and load balancing utilities.

This module provides automatic scaling of resources based on load,
intelligent load balancing, and adaptive resource management for
high-performance distributed operations.
"""

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .logging_config import get_logger


@dataclass
class LoadMetrics:
    """Load metrics for auto-scaling decisions."""

    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    active_tasks: int
    throughput: float  # tasks/second
    response_time: float  # seconds
    error_rate: float  # 0-1


class AutoScaler:
    """Automatic scaling controller for computational resources."""

    def __init__(self,
                 min_workers: int = 1,
                 max_workers: int = 16,
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.4,
                 cooldown_period: float = 60.0,
                 metrics_window: int = 10):
        """Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_utilization: Target CPU utilization (0-1)
            scale_up_threshold: Scale up when utilization exceeds this
            scale_down_threshold: Scale down when utilization below this
            cooldown_period: Minimum time between scaling actions (seconds)
            metrics_window: Number of metrics samples to consider
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period

        self.current_workers = min_workers
        self.last_scale_time = 0.0
        self.metrics_history = deque(maxlen=metrics_window)

        self.logger = get_logger("Scaling.AutoScaler")
        self.scaling_events = []

    def add_metrics(self, metrics: LoadMetrics) -> None:
        """Add new load metrics."""
        self.metrics_history.append(metrics)

    def should_scale(self) -> Tuple[bool, str, int]:
        """Determine if scaling is needed.
        
        Returns:
            (should_scale, direction, target_workers) tuple
        """
        if len(self.metrics_history) < 2:
            return False, "insufficient_data", self.current_workers

        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return False, "cooldown", self.current_workers

        # Calculate recent averages
        recent_metrics = list(self.metrics_history)[-5:]  # Last 5 samples

        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_queue = np.mean([m.queue_length for m in recent_metrics])
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])

        # Calculate scaling score
        scale_score = 0.0
        reasons = []

        # CPU utilization factor
        if avg_cpu > self.scale_up_threshold:
            cpu_factor = (avg_cpu - self.scale_up_threshold) / (1.0 - self.scale_up_threshold)
            scale_score += cpu_factor * 2.0
            reasons.append(f"high_cpu_{avg_cpu:.2f}")
        elif avg_cpu < self.scale_down_threshold:
            cpu_factor = (self.scale_down_threshold - avg_cpu) / self.scale_down_threshold
            scale_score -= cpu_factor * 1.0
            reasons.append(f"low_cpu_{avg_cpu:.2f}")

        # Queue length factor
        queue_threshold = self.current_workers * 2  # 2 tasks per worker
        if avg_queue > queue_threshold:
            queue_factor = avg_queue / queue_threshold - 1.0
            scale_score += queue_factor * 1.5
            reasons.append(f"high_queue_{avg_queue:.1f}")

        # Response time factor
        target_response_time = 1.0  # 1 second target
        if avg_response_time > target_response_time * 2:
            response_factor = avg_response_time / target_response_time - 1.0
            scale_score += response_factor * 1.0
            reasons.append(f"slow_response_{avg_response_time:.2f}s")

        # Error rate factor
        if avg_error_rate > 0.05:  # 5% error rate threshold
            error_factor = avg_error_rate * 10
            scale_score += error_factor * 2.0
            reasons.append(f"high_errors_{avg_error_rate:.2f}")

        # Determine scaling action
        if scale_score > 0.5 and self.current_workers < self.max_workers:
            # Scale up
            scale_factor = min(2.0, 1.0 + scale_score)
            target_workers = min(
                self.max_workers,
                int(math.ceil(self.current_workers * scale_factor))
            )

            reason = f"scale_up ({'; '.join(reasons)})"
            return True, reason, target_workers

        elif scale_score < -0.3 and self.current_workers > self.min_workers:
            # Scale down
            scale_factor = max(0.5, 1.0 + scale_score)
            target_workers = max(
                self.min_workers,
                int(math.floor(self.current_workers * scale_factor))
            )

            reason = f"scale_down ({'; '.join(reasons)})"
            return True, reason, target_workers

        return False, "stable", self.current_workers

    def execute_scaling(self, target_workers: int, reason: str) -> bool:
        """Execute scaling action.
        
        Args:
            target_workers: Target number of workers
            reason: Reason for scaling
            
        Returns:
            True if scaling was successful
        """
        if target_workers == self.current_workers:
            return True

        previous_workers = self.current_workers
        self.current_workers = target_workers
        self.last_scale_time = time.time()

        # Log scaling event
        event = {
            'timestamp': self.last_scale_time,
            'from_workers': previous_workers,
            'to_workers': target_workers,
            'reason': reason
        }

        self.scaling_events.append(event)

        self.logger.info(
            f"Scaled workers: {previous_workers} -> {target_workers} "
            f"(reason: {reason})"
        )

        return True

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        total_events = len(self.scaling_events)
        scale_ups = sum(1 for e in self.scaling_events
                       if e['to_workers'] > e['from_workers'])
        scale_downs = total_events - scale_ups

        if self.metrics_history:
            latest = self.metrics_history[-1]
            metrics_summary = {
                'cpu_utilization': latest.cpu_utilization,
                'memory_utilization': latest.memory_utilization,
                'queue_length': latest.queue_length,
                'active_tasks': latest.active_tasks,
                'throughput': latest.throughput,
                'response_time': latest.response_time,
                'error_rate': latest.error_rate
            }
        else:
            metrics_summary = {}

        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'target_utilization': self.target_utilization,
            'total_scaling_events': total_events,
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'last_scale_time': self.last_scale_time,
            'metrics_samples': len(self.metrics_history),
            'latest_metrics': metrics_summary
        }


class LoadBalancer:
    """Intelligent load balancer for distributing work."""

    def __init__(self,
                 initial_workers: List[Any],
                 selection_strategy: str = "least_loaded",
                 health_check_interval: float = 30.0):
        """Initialize load balancer.
        
        Args:
            initial_workers: List of initial worker instances
            selection_strategy: Worker selection strategy
            health_check_interval: Health check interval in seconds
        """
        self.workers = {}
        self.selection_strategy = selection_strategy
        self.health_check_interval = health_check_interval

        # Worker statistics
        self.worker_stats = defaultdict(lambda: {
            'total_requests': 0,
            'active_requests': 0,
            'total_response_time': 0.0,
            'error_count': 0,
            'last_health_check': 0.0,
            'healthy': True,
            'load_score': 0.0
        })

        self.logger = get_logger("Scaling.LoadBalancer")
        self.lock = threading.Lock()

        # Add initial workers
        for worker in initial_workers:
            self.add_worker(worker)

    def add_worker(self, worker: Any, worker_id: Optional[str] = None) -> str:
        """Add a worker to the load balancer.
        
        Args:
            worker: Worker instance
            worker_id: Optional worker ID
            
        Returns:
            Worker ID
        """
        if worker_id is None:
            worker_id = f"worker_{id(worker)}"

        with self.lock:
            self.workers[worker_id] = worker
            self.worker_stats[worker_id]['healthy'] = True

        self.logger.info(f"Added worker: {worker_id}")
        return worker_id

    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the load balancer.
        
        Args:
            worker_id: Worker ID to remove
            
        Returns:
            True if worker was removed
        """
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                # Keep stats for historical analysis
                self.worker_stats[worker_id]['healthy'] = False

                self.logger.info(f"Removed worker: {worker_id}")
                return True

            return False

    def select_worker(self) -> Optional[Tuple[str, Any]]:
        """Select optimal worker based on strategy.
        
        Returns:
            (worker_id, worker) tuple or None if no healthy workers
        """
        with self.lock:
            healthy_workers = {
                wid: worker for wid, worker in self.workers.items()
                if self.worker_stats[wid]['healthy']
            }

            if not healthy_workers:
                return None

            if self.selection_strategy == "round_robin":
                return self._select_round_robin(healthy_workers)
            elif self.selection_strategy == "least_loaded":
                return self._select_least_loaded(healthy_workers)
            elif self.selection_strategy == "fastest_response":
                return self._select_fastest_response(healthy_workers)
            elif self.selection_strategy == "weighted_response":
                return self._select_weighted_response(healthy_workers)
            else:
                # Default: least loaded
                return self._select_least_loaded(healthy_workers)

    def _select_round_robin(self, workers: Dict[str, Any]) -> Tuple[str, Any]:
        """Round-robin selection."""
        worker_ids = list(workers.keys())
        selected_id = min(worker_ids,
                         key=lambda wid: self.worker_stats[wid]['total_requests'])
        return selected_id, workers[selected_id]

    def _select_least_loaded(self, workers: Dict[str, Any]) -> Tuple[str, Any]:
        """Select worker with least active requests."""
        selected_id = min(workers.keys(),
                         key=lambda wid: self.worker_stats[wid]['active_requests'])
        return selected_id, workers[selected_id]

    def _select_fastest_response(self, workers: Dict[str, Any]) -> Tuple[str, Any]:
        """Select worker with fastest average response time."""
        def avg_response_time(worker_id):
            stats = self.worker_stats[worker_id]
            if stats['total_requests'] == 0:
                return 0.0
            return stats['total_response_time'] / stats['total_requests']

        selected_id = min(workers.keys(), key=avg_response_time)
        return selected_id, workers[selected_id]

    def _select_weighted_response(self, workers: Dict[str, Any]) -> Tuple[str, Any]:
        """Weighted selection based on performance metrics."""
        best_score = float('inf')
        best_worker = None

        for worker_id in workers:
            stats = self.worker_stats[worker_id]

            # Calculate composite load score
            active_load = stats['active_requests']
            avg_response = (stats['total_response_time'] / stats['total_requests']
                          if stats['total_requests'] > 0 else 0.0)
            error_rate = (stats['error_count'] / stats['total_requests']
                         if stats['total_requests'] > 0 else 0.0)

            # Weighted score (lower is better)
            score = (active_load * 2.0 +      # Active requests weight
                    avg_response * 1.0 +      # Response time weight
                    error_rate * 5.0)         # Error rate weight

            if score < best_score:
                best_score = score
                best_worker = worker_id

        return best_worker, workers[best_worker]

    def record_request_start(self, worker_id: str) -> None:
        """Record start of request processing."""
        with self.lock:
            stats = self.worker_stats[worker_id]
            stats['active_requests'] += 1
            stats['total_requests'] += 1

    def record_request_end(self, worker_id: str,
                          response_time: float,
                          success: bool = True) -> None:
        """Record end of request processing."""
        with self.lock:
            stats = self.worker_stats[worker_id]
            stats['active_requests'] = max(0, stats['active_requests'] - 1)
            stats['total_response_time'] += response_time

            if not success:
                stats['error_count'] += 1

    def health_check_worker(self, worker_id: str) -> bool:
        """Perform health check on specific worker.
        
        Args:
            worker_id: Worker ID to check
            
        Returns:
            True if worker is healthy
        """
        if worker_id not in self.workers:
            return False

        worker = self.workers[worker_id]
        current_time = time.time()

        try:
            # Basic health check - worker should have required methods
            healthy = (callable(worker) or
                      hasattr(worker, 'submit') or
                      hasattr(worker, 'execute'))

            # Update health status
            with self.lock:
                self.worker_stats[worker_id]['healthy'] = healthy
                self.worker_stats[worker_id]['last_health_check'] = current_time

            return healthy

        except Exception as e:
            self.logger.warning(f"Health check failed for {worker_id}: {e}")

            with self.lock:
                self.worker_stats[worker_id]['healthy'] = False
                self.worker_stats[worker_id]['last_health_check'] = current_time

            return False

    def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all workers.
        
        Returns:
            Dictionary mapping worker_id to health status
        """
        results = {}

        for worker_id in list(self.workers.keys()):
            results[worker_id] = self.health_check_worker(worker_id)

        return results

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            total_workers = len(self.workers)
            healthy_workers = sum(1 for stats in self.worker_stats.values()
                                if stats['healthy'])

            total_requests = sum(stats['total_requests']
                               for stats in self.worker_stats.values())
            total_errors = sum(stats['error_count']
                             for stats in self.worker_stats.values())

            active_requests = sum(stats['active_requests']
                                for stats in self.worker_stats.values())

            return {
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'selection_strategy': self.selection_strategy,
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
                'active_requests': active_requests,
                'worker_details': dict(self.worker_stats)
            }


class AdaptiveResourceManager:
    """Comprehensive adaptive resource management system."""

    def __init__(self,
                 resource_factory: Callable,
                 initial_workers: int = 2,
                 max_workers: int = 16):
        """Initialize adaptive resource manager.
        
        Args:
            resource_factory: Function to create worker resources
            initial_workers: Initial number of workers
            max_workers: Maximum number of workers
        """
        self.resource_factory = resource_factory

        # Create initial workers
        initial_worker_instances = []
        for _ in range(initial_workers):
            worker = resource_factory()
            initial_worker_instances.append(worker)

        # Initialize components
        self.load_balancer = LoadBalancer(initial_worker_instances)
        self.auto_scaler = AutoScaler(
            min_workers=1,
            max_workers=max_workers,
            cooldown_period=30.0
        )

        self.logger = get_logger("Scaling.AdaptiveManager")
        self.metrics_thread = None
        self.running = False

        # Performance tracking
        self.request_history = deque(maxlen=1000)
        self.last_metrics_time = time.time()

    def start(self) -> None:
        """Start the adaptive resource manager."""
        self.running = True
        self.metrics_thread = threading.Thread(target=self._metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()

        self.logger.info("Adaptive resource manager started")

    def stop(self) -> None:
        """Stop the adaptive resource manager."""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5.0)

        self.logger.info("Adaptive resource manager stopped")

    def execute_request(self, func: Callable, *args, **kwargs) -> Any:
        """Execute request with load balancing and scaling.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        start_time = time.perf_counter()

        # Select worker
        selection = self.load_balancer.select_worker()
        if selection is None:
            raise RuntimeError("No healthy workers available")

        worker_id, worker = selection

        try:
            # Record request start
            self.load_balancer.record_request_start(worker_id)

            # Execute request
            if callable(worker):
                result = worker(func, *args, **kwargs)
            elif hasattr(worker, 'submit'):
                future = worker.submit(func, *args, **kwargs)
                result = future.result()
            else:
                result = func(*args, **kwargs)  # Fallback

            # Record success
            response_time = time.perf_counter() - start_time
            self.load_balancer.record_request_end(worker_id, response_time, True)

            # Track request for metrics
            self.request_history.append({
                'timestamp': time.time(),
                'worker_id': worker_id,
                'response_time': response_time,
                'success': True
            })

            return result

        except Exception as e:
            # Record failure
            response_time = time.perf_counter() - start_time
            self.load_balancer.record_request_end(worker_id, response_time, False)

            self.request_history.append({
                'timestamp': time.time(),
                'worker_id': worker_id,
                'response_time': response_time,
                'success': False,
                'error': str(e)
            })

            raise

    def _metrics_loop(self) -> None:
        """Background thread for metrics collection and scaling decisions."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.auto_scaler.add_metrics(metrics)

                # Check if scaling is needed
                should_scale, reason, target_workers = self.auto_scaler.should_scale()

                if should_scale:
                    self._execute_scaling(target_workers, reason)

                # Health check workers
                self.load_balancer.health_check_all()

                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
                time.sleep(30)  # Longer sleep on error

    def _collect_metrics(self) -> LoadMetrics:
        """Collect current load metrics."""
        current_time = time.time()

        # Recent requests (last 60 seconds)
        recent_requests = [
            req for req in self.request_history
            if current_time - req['timestamp'] < 60
        ]

        # Calculate metrics
        if recent_requests:
            throughput = len(recent_requests) / 60  # requests per second
            avg_response_time = np.mean([req['response_time'] for req in recent_requests])
            error_rate = sum(1 for req in recent_requests if not req['success']) / len(recent_requests)
        else:
            throughput = 0.0
            avg_response_time = 0.0
            error_rate = 0.0

        # Load balancer stats
        lb_stats = self.load_balancer.get_load_balancer_stats()

        # Estimate resource utilization
        active_requests = lb_stats['active_requests']
        total_workers = lb_stats['healthy_workers']

        cpu_utilization = min(1.0, active_requests / max(1, total_workers))
        memory_utilization = min(1.0, len(self.request_history) / 1000)  # Rough estimate

        return LoadMetrics(
            timestamp=current_time,
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            queue_length=active_requests,
            active_tasks=active_requests,
            throughput=throughput,
            response_time=avg_response_time,
            error_rate=error_rate
        )

    def _execute_scaling(self, target_workers: int, reason: str) -> None:
        """Execute scaling by adding/removing workers."""
        current_workers = len(self.load_balancer.workers)

        if target_workers > current_workers:
            # Scale up - add workers
            workers_to_add = target_workers - current_workers
            for _ in range(workers_to_add):
                try:
                    new_worker = self.resource_factory()
                    self.load_balancer.add_worker(new_worker)
                except Exception as e:
                    self.logger.error(f"Failed to create worker: {e}")
                    break

        elif target_workers < current_workers:
            # Scale down - remove workers
            workers_to_remove = current_workers - target_workers
            worker_ids = list(self.load_balancer.workers.keys())

            # Remove least active workers first
            worker_loads = [
                (wid, self.load_balancer.worker_stats[wid]['active_requests'])
                for wid in worker_ids
            ]
            worker_loads.sort(key=lambda x: x[1])  # Sort by active requests

            for i in range(min(workers_to_remove, len(worker_loads))):
                worker_id = worker_loads[i][0]
                self.load_balancer.remove_worker(worker_id)

        # Update auto-scaler
        self.auto_scaler.execute_scaling(target_workers, reason)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'load_balancer': self.load_balancer.get_load_balancer_stats(),
            'auto_scaler': self.auto_scaler.get_scaling_stats(),
            'recent_requests': len(self.request_history),
            'running': self.running
        }
