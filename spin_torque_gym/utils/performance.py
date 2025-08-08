"""Performance optimization utilities for SpinTorque Gym.

This module provides caching, vectorization, and performance optimization
tools to scale spintronic device simulations efficiently.
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings


@dataclass
class CacheEntry:
    """Cache entry for storing computation results."""
    result: Any
    timestamp: float
    access_count: int = 0
    hit_count: int = 0


class AdaptiveCache:
    """Adaptive cache with LRU eviction and performance tracking."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 300.0,  # 5 minutes
        adaptive_sizing: bool = True
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time-to-live for cache entries (seconds)
            adaptive_sizing: Whether to adapt cache size based on performance
        """
        self.max_size = max_size
        self.ttl = ttl
        self.adaptive_sizing = adaptive_sizing
        
        self._cache = {}
        self._access_order = deque()
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_access_time = 0.0
        
        # Adaptive parameters
        self.target_hit_rate = 0.8
        self.resize_threshold = 100  # Access count before considering resize
        self.access_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()
        
        with self._lock:
            self.access_count += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl:
                    self._remove_key(key)
                    self.misses += 1
                    return None
                
                # Update access tracking
                entry.access_count += 1
                entry.hit_count += 1
                
                # Move to end of access order (LRU)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                self.hits += 1
                self.total_access_time += time.time() - start_time
                return entry.result
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._remove_key(key)
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                result=value,
                timestamp=time.time(),
                access_count=1,
                hit_count=0
            )
            
            self._cache[key] = entry
            self._access_order.append(key)
            
            # Adaptive sizing
            if self.adaptive_sizing and self.access_count % self.resize_threshold == 0:
                self._adapt_cache_size()
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                self.evictions += 1
    
    def _adapt_cache_size(self) -> None:
        """Adapt cache size based on hit rate."""
        if self.hits + self.misses == 0:
            return
        
        hit_rate = self.hits / (self.hits + self.misses)
        
        if hit_rate < self.target_hit_rate and self.max_size < 10000:
            # Increase cache size if hit rate is low
            new_size = min(self.max_size * 2, 10000)
            self.max_size = new_size
        elif hit_rate > 0.9 and self.max_size > 100:
            # Decrease cache size if hit rate is very high (possible over-caching)
            new_size = max(self.max_size // 2, 100)
            # Only shrink if we won't lose too much data
            if len(self._cache) <= new_size * 1.2:
                self.max_size = new_size
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        avg_access_time = self.total_access_time / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'avg_access_time': avg_access_time,
            'total_requests': total_requests
        }


class ComputationOptimizer:
    """Optimization utilities for heavy computations."""
    
    def __init__(self, cache_size: int = 1000, enable_vectorization: bool = True):
        """Initialize computation optimizer.
        
        Args:
            cache_size: Size of computation cache
            enable_vectorization: Whether to enable vectorized operations
        """
        self.cache = AdaptiveCache(max_size=cache_size)
        self.enable_vectorization = enable_vectorization
        
        # Pre-computed lookup tables
        self._trig_cache = {}
        self._exp_cache = {}
        
        # Vectorization settings
        self.batch_size = 100
        self.max_workers = 4
        
        # Initialize lookup tables
        self._init_lookup_tables()
    
    def _init_lookup_tables(self) -> None:
        """Initialize lookup tables for common functions."""
        # Common angles for trigonometric functions
        angles = np.linspace(0, 2*np.pi, 1000)
        self._trig_cache = {
            'sin': np.sin(angles),
            'cos': np.cos(angles),
            'angles': angles
        }
        
        # Common exponential values
        exp_values = np.linspace(-10, 10, 1000)
        self._exp_cache = {
            'exp': np.exp(exp_values),
            'values': exp_values
        }
    
    def hash_params(self, params: Dict[str, Any]) -> str:
        """Generate hash for parameter dictionary.
        
        Args:
            params: Parameters to hash
            
        Returns:
            Hash string
        """
        # Create deterministic string representation
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def cached_computation(
        self,
        func: Callable,
        params: Dict[str, Any],
        cache_key_prefix: str = ""
    ) -> Any:
        """Execute computation with caching.
        
        Args:
            func: Function to execute
            params: Function parameters
            cache_key_prefix: Prefix for cache key
            
        Returns:
            Function result
        """
        # Generate cache key
        param_hash = self.hash_params(params)
        cache_key = f"{cache_key_prefix}_{param_hash}"
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = func(**params)
        self.cache.put(cache_key, result)
        
        return result
    
    def fast_trig(self, angle: float, func: str = 'sin') -> float:
        """Fast trigonometric function using lookup table.
        
        Args:
            angle: Angle in radians
            func: Function name ('sin', 'cos')
            
        Returns:
            Function result
        """
        if func not in self._trig_cache:
            return getattr(np, func)(angle)
        
        # Normalize angle to [0, 2π]
        angle = angle % (2 * np.pi)
        
        # Find closest index
        angles = self._trig_cache['angles']
        idx = int(angle / (2 * np.pi) * len(angles))
        idx = min(idx, len(angles) - 1)
        
        return self._trig_cache[func][idx]
    
    def fast_exp(self, x: float) -> float:
        """Fast exponential function using lookup table.
        
        Args:
            x: Input value
            
        Returns:
            Exponential result
        """
        # Clamp to lookup table range
        x_clamped = np.clip(x, -10, 10)
        
        # Find closest index
        values = self._exp_cache['values']
        idx = int((x_clamped + 10) / 20 * len(values))
        idx = min(max(idx, 0), len(values) - 1)
        
        return self._exp_cache['exp'][idx]
    
    def vectorized_solve(
        self,
        solve_func: Callable,
        params_list: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Solve multiple problems in parallel.
        
        Args:
            solve_func: Solver function
            params_list: List of parameter dictionaries
            max_workers: Maximum number of worker threads
            
        Returns:
            List of results
        """
        if not self.enable_vectorization or len(params_list) < 2:
            # Sequential execution for small batches
            return [solve_func(**params) for params in params_list]
        
        # Parallel execution
        max_workers = max_workers or min(self.max_workers, len(params_list))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(solve_func, **params) for params in params_list]
            results = [future.result() for future in futures]
        
        return results
    
    def batch_process(
        self,
        process_func: Callable,
        data: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Process data in batches for memory efficiency.
        
        Args:
            process_func: Processing function
            data: Data to process
            batch_size: Batch size
            
        Returns:
            Processed results
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = process_func(batch)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get optimizer performance statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'cache': cache_stats,
            'vectorization_enabled': self.enable_vectorization,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'lookup_tables': {
                'trig_size': len(self._trig_cache.get('angles', [])),
                'exp_size': len(self._exp_cache.get('values', []))
            }
        }


class PerformanceProfiler:
    """Profiler for measuring and optimizing performance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self._start_times = {}
        self._lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation.
        
        Args:
            name: Operation name
        """
        with self._lock:
            self._start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Elapsed time
        """
        end_time = time.time()
        
        with self._lock:
            if name in self._start_times:
                elapsed = end_time - self._start_times[name]
                self.timings[name].append(elapsed)
                del self._start_times[name]
                return elapsed
            else:
                warnings.warn(f"Timer '{name}' was not started")
                return 0.0
    
    def time_operation(self, name: str):
        """Context manager for timing operations.
        
        Args:
            name: Operation name
        """
        return TimingContext(self, name)
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self._lock:
            self.counters[name] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self._lock:
            stats = {}
            
            # Timing statistics
            for name, times in self.timings.items():
                if times:
                    stats[f"{name}_avg_time"] = np.mean(times)
                    stats[f"{name}_total_time"] = np.sum(times)
                    stats[f"{name}_count"] = len(times)
                    stats[f"{name}_min_time"] = np.min(times)
                    stats[f"{name}_max_time"] = np.max(times)
            
            # Counter statistics
            stats.update(dict(self.counters))
            
            return stats
    
    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self.timings.clear()
            self.counters.clear()
            self._start_times.clear()


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        """Initialize timing context.
        
        Args:
            profiler: Performance profiler
            name: Operation name
        """
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)


# Global performance optimizer instance
_global_optimizer = None

def get_optimizer() -> ComputationOptimizer:
    """Get global computation optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ComputationOptimizer()
    return _global_optimizer