"""Performance optimization system for Spin Torque RL-Gym.

This module provides comprehensive performance optimization including:
- Adaptive caching and memoization
- Parallel processing and concurrency
- Memory pool management
- Auto-scaling and load balancing
- Performance profiling and optimization
"""

import functools
import hashlib
import logging
import multiprocessing
import os
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class OptimizationLevel(Enum):
    """Performance optimization level."""
    DISABLED = "disabled"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class CacheStrategy(Enum):
    """Caching strategy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    enable_caching: bool = True
    enable_parallelization: bool = True
    enable_memory_pooling: bool = True
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    max_workers: Optional[int] = None
    memory_threshold: float = 0.8  # 80% memory usage threshold
    cpu_threshold: float = 0.8     # 80% CPU usage threshold

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)


class AdaptiveCache:
    """High-performance adaptive caching system."""

    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl_seconds: int = 3600
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds

        # Cache storage
        self._cache = OrderedDict()
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._creation_times = {}

        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Thread safety
        self._lock = threading.RLock()

        # Auto-cleanup timer
        self._cleanup_timer = None
        self._start_cleanup_timer()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        with self._lock:
            current_time = time.time()

            if key not in self._cache:
                self._misses += 1
                return None

            # Check TTL expiration
            if self.ttl_seconds > 0:
                if current_time - self._creation_times.get(key, 0) > self.ttl_seconds:
                    self._evict_key(key)
                    self._misses += 1
                    return None

            # Update access statistics
            self._access_times[key] = current_time
            self._access_counts[key] += 1

            # Move to end (for LRU)
            value = self._cache.pop(key)
            self._cache[key] = value

            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Set cached value for key."""
        with self._lock:
            current_time = time.time()

            # If key exists, update it
            if key in self._cache:
                self._cache[key] = value
                self._access_times[key] = current_time
                self._creation_times[key] = current_time
                return

            # Check if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_one()

            # Add new entry
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] = 1
            self._creation_times[key] = current_time

    def _evict_one(self) -> None:
        """Evict one item based on strategy."""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            # Least recently used
            key_to_evict = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            # Least frequently used
            key_to_evict = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        elif self.strategy == CacheStrategy.TTL:
            # Oldest by creation time
            key_to_evict = min(self._creation_times.keys(), key=lambda k: self._creation_times[k])
        else:  # ADAPTIVE
            key_to_evict = self._adaptive_eviction()

        self._evict_key(key_to_evict)

    def _adaptive_eviction(self) -> str:
        """Smart adaptive eviction strategy."""
        current_time = time.time()

        # Score each key based on multiple factors
        scores = {}
        for key in self._cache.keys():
            # Factors: recency, frequency, age
            recency_score = current_time - self._access_times.get(key, 0)
            frequency_score = 1.0 / (self._access_counts[key] + 1)
            age_score = current_time - self._creation_times.get(key, 0)

            # Weighted combination (higher score = more likely to evict)
            scores[key] = (0.4 * recency_score +
                          0.4 * frequency_score +
                          0.2 * age_score)

        return max(scores.keys(), key=lambda k: scores[k])

    def _evict_key(self, key: str) -> None:
        """Evict specific key."""
        if key in self._cache:
            del self._cache[key]
            self._access_times.pop(key, None)
            self._access_counts.pop(key, None)
            self._creation_times.pop(key, None)
            self._evictions += 1

    def _start_cleanup_timer(self):
        """Start periodic cleanup of expired entries."""
        def cleanup():
            self._cleanup_expired()
            # Restart timer
            self._cleanup_timer = threading.Timer(60.0, cleanup)  # Clean every minute
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        self._cleanup_timer = threading.Timer(60.0, cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _cleanup_expired(self):
        """Remove expired entries."""
        if self.ttl_seconds <= 0:
            return

        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, creation_time in self._creation_times.items()
                if current_time - creation_time > self.ttl_seconds
            ]

            for key in expired_keys:
                self._evict_key(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'strategy': self.strategy.value
            }

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()

    def __del__(self):
        """Cleanup on destruction."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()


def cached(
    max_size: int = 128,
    strategy: CacheStrategy = CacheStrategy.LRU,
    ttl_seconds: int = 3600,
    key_func: Optional[Callable] = None
):
    """Decorator for function caching with advanced strategies.
    
    Args:
        max_size: Maximum cache size
        strategy: Caching strategy
        ttl_seconds: Time-to-live for cached entries
        key_func: Custom key generation function
    """
    def decorator(func):
        cache = AdaptiveCache(max_size, strategy, ttl_seconds)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)

            return result

        # Attach cache for inspection
        wrapper._cache = cache
        wrapper.cache_info = cache.get_stats
        wrapper.cache_clear = cache.clear

        return wrapper

    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate consistent cache key from function arguments."""
    # Convert arguments to string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items())) if kwargs else ""

    # Create hash for consistent key length
    key_data = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()


class WorkerPool:
    """Adaptive worker pool for parallel processing."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._lock = threading.Lock()

        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize thread and process pools."""
        if self.config.enable_parallelization:
            max_workers = self.config.max_workers

            self.thread_pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="SpinTorque"
            )

            # Process pool for CPU-intensive tasks
            if multiprocessing.cpu_count() > 1:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=min(max_workers, multiprocessing.cpu_count())
                )

    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O-bound task to thread pool."""
        if not self.thread_pool:
            # Fallback to synchronous execution
            return func(*args, **kwargs)

        with self._lock:
            self._active_tasks += 1

        future = self.thread_pool.submit(self._wrapped_task, func, *args, **kwargs)
        return future

    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU-bound task to process pool."""
        if not self.process_pool:
            # Fallback to thread pool or synchronous
            return self.submit_io_task(func, *args, **kwargs)

        with self._lock:
            self._active_tasks += 1

        future = self.process_pool.submit(self._wrapped_task, func, *args, **kwargs)
        return future

    def _wrapped_task(self, func: Callable, *args, **kwargs):
        """Wrapper for task execution with error handling."""
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._completed_tasks += 1
            return result
        except Exception as e:
            with self._lock:
                self._failed_tasks += 1
            logging.error(f"Task execution failed: {e}")
            raise
        finally:
            with self._lock:
                self._active_tasks -= 1

    def map_parallel(
        self,
        func: Callable,
        iterable: List[Any],
        cpu_bound: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Execute function in parallel over iterable."""
        if max_workers:
            # Use temporary pool with specific worker count
            executor_class = ProcessPoolExecutor if cpu_bound else ThreadPoolExecutor
            with executor_class(max_workers=max_workers) as executor:
                futures = [executor.submit(func, item) for item in iterable]
                results = [future.result() for future in as_completed(futures)]
        else:
            # Use existing pools
            executor = self.process_pool if cpu_bound else self.thread_pool
            if executor:
                futures = [executor.submit(func, item) for item in iterable]
                results = [future.result() for future in as_completed(futures)]
            else:
                # Fallback to sequential execution
                results = [func(item) for item in iterable]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            return {
                'active_tasks': self._active_tasks,
                'completed_tasks': self._completed_tasks,
                'failed_tasks': self._failed_tasks,
                'thread_pool_size': self.thread_pool._max_workers if self.thread_pool else 0,
                'process_pool_size': self.process_pool._max_workers if self.process_pool else 0
            }

    def shutdown(self, wait: bool = True):
        """Shutdown worker pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)

        if self.process_pool:
            self.process_pool.shutdown(wait=wait)


class MemoryPool:
    """Memory pool for efficient object allocation."""

    def __init__(self, object_factory: Callable, initial_size: int = 10, max_size: int = 100):
        self.object_factory = object_factory
        self.initial_size = initial_size
        self.max_size = max_size

        self._pool = []
        self._borrowed = set()
        self._lock = threading.Lock()

        # Pre-allocate initial objects
        for _ in range(initial_size):
            obj = self._create_object()
            self._pool.append(obj)

    def _create_object(self):
        """Create new object instance."""
        return self.object_factory()

    def borrow(self):
        """Borrow object from pool."""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
            else:
                # Create new object if pool is empty
                obj = self._create_object()

            self._borrowed.add(id(obj))
            return obj

    def return_object(self, obj):
        """Return object to pool."""
        with self._lock:
            obj_id = id(obj)
            if obj_id not in self._borrowed:
                raise ValueError("Object not borrowed from this pool")

            self._borrowed.remove(obj_id)

            # Return to pool if under max size
            if len(self._pool) < self.max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
            # Otherwise let it be garbage collected

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'borrowed_count': len(self._borrowed),
                'max_size': self.max_size
            }


class AutoScaler:
    """Automatic resource scaling based on load."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_workers = config.max_workers
        self.scaling_history = []
        self._last_scale_time = 0
        self._scale_cooldown = 60  # seconds

        # Resource monitoring
        self._cpu_usage_history = []
        self._memory_usage_history = []
        self._task_queue_size_history = []

    def record_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        task_queue_size: int
    ):
        """Record performance metrics for scaling decisions."""
        current_time = time.time()

        self._cpu_usage_history.append((current_time, cpu_usage))
        self._memory_usage_history.append((current_time, memory_usage))
        self._task_queue_size_history.append((current_time, task_queue_size))

        # Keep only recent history
        cutoff_time = current_time - 300  # 5 minutes
        self._cpu_usage_history = [(t, v) for t, v in self._cpu_usage_history if t > cutoff_time]
        self._memory_usage_history = [(t, v) for t, v in self._memory_usage_history if t > cutoff_time]
        self._task_queue_size_history = [(t, v) for t, v in self._task_queue_size_history if t > cutoff_time]

    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed."""
        if not self._cpu_usage_history or not self._memory_usage_history:
            return False

        # Calculate recent averages
        recent_cpu = sum(usage for _, usage in self._cpu_usage_history[-10:]) / len(self._cpu_usage_history[-10:])
        recent_memory = sum(usage for _, usage in self._memory_usage_history[-10:]) / len(self._memory_usage_history[-10:])
        recent_queue_size = sum(size for _, size in self._task_queue_size_history[-10:]) / len(self._task_queue_size_history[-10:]) if self._task_queue_size_history else 0

        # Scale up conditions
        conditions = [
            recent_cpu > self.config.cpu_threshold,
            recent_memory > self.config.memory_threshold,
            recent_queue_size > 10  # Queue backlog
        ]

        return any(conditions)

    def should_scale_down(self) -> bool:
        """Determine if scaling down is possible."""
        if not self._cpu_usage_history:
            return False

        # Calculate recent averages
        recent_cpu = sum(usage for _, usage in self._cpu_usage_history[-10:]) / len(self._cpu_usage_history[-10:])
        recent_queue_size = sum(size for _, size in self._task_queue_size_history[-10:]) / len(self._task_queue_size_history[-10:]) if self._task_queue_size_history else 0

        # Scale down conditions (conservative)
        conditions = [
            recent_cpu < 0.3,  # Low CPU usage
            recent_queue_size < 2,  # Low queue size
            self.current_workers > 2  # Don't go below minimum
        ]

        return all(conditions)

    def get_recommended_workers(self) -> int:
        """Get recommended number of workers."""
        current_time = time.time()

        # Respect cooldown period
        if current_time - self._last_scale_time < self._scale_cooldown:
            return self.current_workers

        if self.should_scale_up():
            new_workers = min(self.current_workers * 2, self.config.max_workers * 2)
            self.scaling_history.append((current_time, 'up', self.current_workers, new_workers))
            self._last_scale_time = current_time
            return new_workers
        elif self.should_scale_down():
            new_workers = max(self.current_workers // 2, 2)
            self.scaling_history.append((current_time, 'down', self.current_workers, new_workers))
            self._last_scale_time = current_time
            return new_workers

        return self.current_workers


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.worker_pool = WorkerPool(self.config)
        self.memory_pools = {}
        self.auto_scaler = AutoScaler(self.config)

        # Global cache instance
        self.global_cache = AdaptiveCache(
            max_size=self.config.max_cache_size,
            strategy=CacheStrategy.ADAPTIVE,
            ttl_seconds=self.config.cache_ttl_seconds
        )

        # Performance metrics
        self._start_time = time.time()
        self._optimization_events = []

    def create_memory_pool(
        self,
        name: str,
        object_factory: Callable,
        initial_size: int = 10,
        max_size: int = 100
    ) -> MemoryPool:
        """Create named memory pool."""
        pool = MemoryPool(object_factory, initial_size, max_size)
        self.memory_pools[name] = pool
        return pool

    def get_memory_pool(self, name: str) -> Optional[MemoryPool]:
        """Get memory pool by name."""
        return self.memory_pools.get(name)

    def optimize_function(
        self,
        cache_size: int = 128,
        cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_parallel: bool = True
    ):
        """Decorator to optimize function performance."""
        def decorator(func):
            # Add caching
            cached_func = cached(
                max_size=cache_size,
                strategy=cache_strategy,
                ttl_seconds=self.config.cache_ttl_seconds
            )(func)

            if enable_parallel:
                @functools.wraps(cached_func)
                def parallel_wrapper(*args, **kwargs):
                    # For single calls, use cached function directly
                    if not isinstance(args[0], (list, tuple)):
                        return cached_func(*args, **kwargs)

                    # For batch calls, use parallel processing
                    iterable = args[0]
                    remaining_args = args[1:]

                    def single_call(item):
                        return cached_func(item, *remaining_args, **kwargs)

                    return self.worker_pool.map_parallel(single_call, iterable)

                return parallel_wrapper
            else:
                return cached_func

        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        current_time = time.time()
        uptime = current_time - self._start_time

        report = {
            'uptime_seconds': uptime,
            'optimization_level': self.config.optimization_level.value,
            'worker_pool_stats': self.worker_pool.get_stats(),
            'global_cache_stats': self.global_cache.get_stats(),
            'memory_pools': {
                name: pool.get_stats()
                for name, pool in self.memory_pools.items()
            },
            'auto_scaling': {
                'current_workers': self.auto_scaler.current_workers,
                'scaling_events': len(self.auto_scaler.scaling_history),
                'last_scaling': self.auto_scaler.scaling_history[-1] if self.auto_scaler.scaling_history else None
            }
        }

        return report

    def cleanup(self):
        """Cleanup resources."""
        self.worker_pool.shutdown()
        self.global_cache.clear()

        for pool in self.memory_pools.values():
            # Memory pools will be cleaned up by garbage collection
            pass


# Global optimizer instance
global_optimizer = PerformanceOptimizer()


def initialize_performance_optimization(config: PerformanceConfig = None) -> PerformanceOptimizer:
    """Initialize global performance optimization.
    
    Args:
        config: Performance configuration
        
    Returns:
        Configured performance optimizer
    """
    global global_optimizer

    if config:
        global_optimizer = PerformanceOptimizer(config)

    logging.info(f"Performance optimization initialized at {config.optimization_level.value if config else 'basic'} level")
    return global_optimizer


def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    return global_optimizer


# Example usage and testing
if __name__ == "__main__":
    # Initialize performance optimization
    config = PerformanceConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        max_cache_size=1000,
        max_workers=8
    )

    optimizer = initialize_performance_optimization(config)

    # Test caching decorator
    @optimizer.optimize_function(cache_size=100, enable_parallel=True)
    def expensive_computation(x: float) -> float:
        """Simulate expensive computation."""
        time.sleep(0.01)  # Simulate work
        return x ** 2 + 42.0

    # Test performance
    start_time = time.time()

    # First call (cache miss)
    result1 = expensive_computation(3.14)

    # Second call (cache hit)
    result2 = expensive_computation(3.14)

    # Batch processing
    batch_results = expensive_computation([1.0, 2.0, 3.0, 4.0, 5.0])

    end_time = time.time()

    print(f"Computation results: {result1}, {result2}")
    print(f"Batch results: {batch_results}")
    print(f"Total time: {end_time - start_time:.3f}s")

    # Performance report
    report = optimizer.get_performance_report()
    print(f"Cache hit rate: {report['global_cache_stats']['hit_rate']:.2%}")
    print(f"Worker pool stats: {report['worker_pool_stats']}")

    print("Performance optimization system ready!")

    # Cleanup
    optimizer.cleanup()
