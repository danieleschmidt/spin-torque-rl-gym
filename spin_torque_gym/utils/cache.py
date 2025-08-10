"""Advanced caching system for SpinTorque Gym.

This module provides adaptive caching with performance optimization,
automatic eviction, and intelligent cache management for scalable
operations.
"""

import hashlib
import pickle
import time
import threading
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .logging_config import get_logger, PerformanceLogger


class CacheStats:
    """Statistics tracking for cache performance."""
    
    def __init__(self):
        """Initialize cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.writes = 0
        self.total_access_time = 0.0
        self.total_compute_time = 0.0
        self.start_time = time.time()
        
        # Access pattern tracking
        self.access_frequency = defaultdict(int)
        self.access_times = defaultdict(list)
    
    def record_hit(self, key: str, access_time: float) -> None:
        """Record cache hit."""
        self.hits += 1
        self.total_access_time += access_time
        self.access_frequency[key] += 1
        self.access_times[key].append(time.time())
    
    def record_miss(self, key: str, compute_time: float) -> None:
        """Record cache miss."""
        self.misses += 1
        self.total_compute_time += compute_time
        self.access_frequency[key] += 1
        self.access_times[key].append(time.time())
    
    def record_write(self, key: str) -> None:
        """Record cache write."""
        self.writes += 1
    
    def record_eviction(self, key: str) -> None:
        """Record cache eviction."""
        self.evictions += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return self.hits + self.misses
    
    @property
    def average_access_time(self) -> float:
        """Average access time for hits."""
        return self.total_access_time / self.hits if self.hits > 0 else 0.0
    
    @property
    def average_compute_time(self) -> float:
        """Average computation time for misses."""
        return self.total_compute_time / self.misses if self.misses > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'writes': self.writes,
            'hit_rate': self.hit_rate,
            'total_requests': self.total_requests,
            'average_access_time': self.average_access_time,
            'average_compute_time': self.average_compute_time,
            'uptime': uptime,
            'requests_per_second': self.total_requests / uptime if uptime > 0 else 0.0,
            'unique_keys': len(self.access_frequency),
            'most_accessed': sorted(
                self.access_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class LRUCache:
    """Least Recently Used cache with adaptive sizing."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds (0 = no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.compute_times = {}  # Track how long items took to compute
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.logger = get_logger("Cache.LRU")
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        # Create deterministic key from arguments
        key_data = (args, tuple(sorted(kwargs.items())))
        
        # Handle numpy arrays specially
        def serialize_item(item):
            if isinstance(item, np.ndarray):
                return f"array_{hash(item.tobytes())}_{item.shape}"
            elif isinstance(item, dict):
                return tuple(sorted(item.items()))
            elif callable(item):
                return f"func_{item.__name__}_{id(item)}"
            else:
                return item
        
        serializable_data = serialize_item(key_data)
        key_str = str(serializable_data)
        
        # Use hash for very long keys
        if len(key_str) > 200:
            key_str = hashlib.md5(key_str.encode()).hexdigest()
        
        return key_str
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from cache.
        
        Returns:
            (found, value) tuple
        """
        with self.lock:
            start_time = time.perf_counter()
            
            if key in self.cache:
                # Check TTL
                if self.ttl > 0:
                    age = time.time() - self.access_times[key]
                    if age > self.ttl:
                        # Expired
                        del self.cache[key]
                        del self.access_times[key]
                        if key in self.compute_times:
                            del self.compute_times[key]
                        return False, None
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                
                access_time = time.perf_counter() - start_time
                self.stats.record_hit(key, access_time)
                
                return True, value
            else:
                return False, None
    
    def put(self, key: str, value: Any, compute_time: float = 0.0) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            compute_time: Time it took to compute this value
        """
        with self.lock:
            current_time = time.time()
            
            # If key already exists, update it
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.compute_times[key] = compute_time
            self.stats.record_write(key)
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = next(iter(self.cache))
        
        # Remove from all data structures
        del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        if lru_key in self.compute_times:
            del self.compute_times[lru_key]
        
        self.stats.record_eviction(lru_key)
        self.logger.debug(f"Evicted cache entry: {lru_key}")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry.
        
        Returns:
            True if key was found and removed
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                if key in self.compute_times:
                    del self.compute_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.compute_times.clear()
            self.logger.info("Cache cleared")
    
    def resize(self, new_size: int) -> None:
        """Resize cache to new maximum size."""
        with self.lock:
            self.max_size = new_size
            
            # Evict excess entries
            while len(self.cache) > self.max_size:
                self._evict_lru()
            
            self.logger.info(f"Cache resized to {new_size}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self.lock:
            stats = self.stats.get_stats()
            stats.update({
                'current_size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'memory_efficiency': self._estimate_memory_efficiency()
            })
            return stats
    
    def _estimate_memory_efficiency(self) -> float:
        """Estimate memory efficiency based on access patterns."""
        if not self.cache:
            return 1.0
        
        total_compute_time = sum(self.compute_times.values())
        total_access_freq = sum(self.stats.access_frequency.values())
        
        if total_compute_time == 0 or total_access_freq == 0:
            return 1.0
        
        # Efficiency based on saved computation time vs cache overhead
        saved_time = (self.stats.hits * self.stats.average_compute_time) if self.stats.average_compute_time > 0 else 0
        cache_overhead = self.stats.total_access_time
        
        efficiency = saved_time / (saved_time + cache_overhead) if (saved_time + cache_overhead) > 0 else 1.0
        return min(efficiency, 1.0)


class AdaptiveCache:
    """Adaptive cache that automatically tunes its behavior."""
    
    def __init__(self, 
                 initial_size: int = 500,
                 max_size: int = 5000,
                 min_size: int = 100,
                 ttl: float = 1800.0,
                 adaptation_interval: float = 60.0):
        """Initialize adaptive cache.
        
        Args:
            initial_size: Initial cache size
            max_size: Maximum cache size
            min_size: Minimum cache size
            ttl: Time-to-live in seconds
            adaptation_interval: How often to adapt (seconds)
        """
        self.cache = LRUCache(initial_size, ttl)
        self.max_size = max_size
        self.min_size = min_size
        self.adaptation_interval = adaptation_interval
        self.last_adaptation = time.time()
        
        self.performance_history = []
        self.logger = get_logger("Cache.Adaptive")
        self.performance_logger = PerformanceLogger()
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from cache with adaptation."""
        # Check if we should adapt
        self._maybe_adapt()
        
        return self.cache.get(key)
    
    def put(self, key: str, value: Any, compute_time: float = 0.0) -> None:
        """Put item in cache."""
        self.cache.put(key, value, compute_time)
        
        # Record miss for adaptation
        if compute_time > 0:
            self.cache.stats.record_miss(key, compute_time)
    
    def _maybe_adapt(self) -> None:
        """Adapt cache size based on performance metrics."""
        current_time = time.time()
        
        if current_time - self.last_adaptation < self.adaptation_interval:
            return
        
        self.last_adaptation = current_time
        
        # Get current performance metrics
        stats = self.cache.get_info()
        self.performance_history.append({
            'timestamp': current_time,
            'hit_rate': stats['hit_rate'],
            'memory_efficiency': stats['memory_efficiency'],
            'size': stats['current_size'],
            'requests_per_second': stats['requests_per_second']
        })
        
        # Keep only recent history
        cutoff_time = current_time - 3600  # 1 hour
        self.performance_history = [
            entry for entry in self.performance_history
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(self.performance_history) < 2:
            return
        
        # Analyze trends
        recent = self.performance_history[-5:]  # Last 5 measurements
        hit_rates = [entry['hit_rate'] for entry in recent]
        efficiencies = [entry['memory_efficiency'] for entry in recent]
        
        avg_hit_rate = np.mean(hit_rates)
        avg_efficiency = np.mean(efficiencies)
        current_size = stats['current_size']
        
        # Adaptation logic
        new_size = current_size
        
        if avg_hit_rate < 0.3 and avg_efficiency > 0.7:
            # Low hit rate but good efficiency - increase size
            new_size = min(int(current_size * 1.2), self.max_size)
            reason = "low hit rate, good efficiency"
        elif avg_hit_rate > 0.8 and avg_efficiency < 0.4:
            # High hit rate but poor efficiency - decrease size
            new_size = max(int(current_size * 0.8), self.min_size)
            reason = "high hit rate, poor efficiency"
        elif avg_hit_rate < 0.1:
            # Very low hit rate - decrease size significantly
            new_size = max(int(current_size * 0.5), self.min_size)
            reason = "very low hit rate"
        elif stats['requests_per_second'] > 100 and avg_hit_rate > 0.6:
            # High load with good hit rate - increase size
            new_size = min(int(current_size * 1.1), self.max_size)
            reason = "high load, good hit rate"
        
        if new_size != current_size:
            self.cache.resize(new_size)
            self.logger.info(
                f"Adapted cache size: {current_size} -> {new_size} "
                f"(hit_rate: {avg_hit_rate:.2f}, efficiency: {avg_efficiency:.2f}, "
                f"reason: {reason})"
            )
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get adaptation information."""
        return {
            'max_size': self.max_size,
            'min_size': self.min_size,
            'adaptation_interval': self.adaptation_interval,
            'performance_history_length': len(self.performance_history),
            'last_adaptation': self.last_adaptation,
            'cache_info': self.cache.get_info()
        }


class CacheManager:
    """Global cache manager for different cache types."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches = {}
        self.logger = get_logger("Cache.Manager")
        self.performance_logger = PerformanceLogger()
    
    def get_cache(self, name: str, cache_type: str = "adaptive", **kwargs) -> Union[LRUCache, AdaptiveCache]:
        """Get or create a named cache.
        
        Args:
            name: Cache name
            cache_type: Type of cache ("lru" or "adaptive")
            **kwargs: Cache configuration parameters
        
        Returns:
            Cache instance
        """
        if name not in self.caches:
            if cache_type == "lru":
                self.caches[name] = LRUCache(**kwargs)
            elif cache_type == "adaptive":
                self.caches[name] = AdaptiveCache(**kwargs)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
            
            self.logger.info(f"Created {cache_type} cache: {name}")
        
        return self.caches[name]
    
    def cached(self, cache_name: str = "default", ttl: float = 1800.0):
        """Decorator for caching function results.
        
        Args:
            cache_name: Name of cache to use
            ttl: Time-to-live for cached results
        """
        def decorator(func: Callable) -> Callable:
            cache = self.get_cache(cache_name, ttl=ttl)
            
            def wrapper(*args, **kwargs):
                # Create cache key
                key = cache.cache._make_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                found, result = cache.get(key)
                if found:
                    return result
                
                # Compute result
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                compute_time = time.perf_counter() - start_time
                
                # Cache result
                cache.put(key, result, compute_time)
                
                return result
            
            wrapper.__wrapped__ = func
            wrapper.__cache__ = cache
            return wrapper
        
        return decorator
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_hits = 0
        total_misses = 0
        total_size = 0
        
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_info'):
                cache_stats = cache.get_info()
            elif hasattr(cache, 'get_adaptation_info'):
                cache_info = cache.get_adaptation_info()
                cache_stats = cache_info.get('cache_info', {})
            else:
                cache_stats = {'current_size': 0, 'hits': 0, 'misses': 0}
            
            stats[name] = cache_stats
            
            total_hits += cache_stats.get('hits', 0)
            total_misses += cache_stats.get('misses', 0)
            total_size += cache_stats.get('current_size', 0)
        
        global_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        stats['global'] = {
            'total_caches': len(self.caches),
            'total_hits': total_hits,
            'total_misses': total_misses,
            'global_hit_rate': global_hit_rate,
            'total_cached_items': total_size
        }
        
        return stats
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("Cleared all caches")


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager