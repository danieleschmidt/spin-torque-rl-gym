"""Scalable physics solver with performance optimization and concurrent processing."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .robust_solver import RobustLLGSSolver
from .cache import LRUCache, adaptive_cache
from .concurrency import ThreadPoolManager, process_batch_concurrent
from .performance_optimization import (
    OptimizedComputation,
    VectorizedSolver,
    adaptive_batching,
    jit_compile_if_available,
)

logger = logging.getLogger(__name__)


class ScalableLLGSSolver(RobustLLGSSolver):
    """Scalable LLGS solver with performance optimization and parallel processing."""

    def __init__(
        self,
        method: str = 'euler',
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = 1e-12,
        timeout: float = 2.0,
        max_retries: int = 3,
        enable_caching: bool = True,
        enable_vectorization: bool = True,
        enable_jit: bool = True,
        max_workers: int = 4,
        batch_size: int = 32,
        cache_size: int = 1000
    ):
        """Initialize scalable LLGS solver.
        
        Args:
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance  
            max_step: Maximum time step
            timeout: Maximum computation time
            max_retries: Maximum retry attempts
            enable_caching: Enable result caching
            enable_vectorization: Enable vectorized computations
            enable_jit: Enable just-in-time compilation
            max_workers: Maximum concurrent workers
            batch_size: Batch size for concurrent processing
            cache_size: Maximum cache entries
        """
        super().__init__(method, rtol, atol, max_step, timeout, max_retries)
        
        self.enable_caching = enable_caching
        self.enable_vectorization = enable_vectorization
        self.enable_jit = enable_jit
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialize optimization components
        if self.enable_caching:
            self.result_cache = LRUCache(cache_size)
            self.computation_cache = adaptive_cache(max_size=cache_size)
        
        if self.enable_vectorization:
            self.vectorized_solver = VectorizedSolver()
            self.optimized_computation = OptimizedComputation()
        
        # Thread pool for concurrent processing
        self.thread_pool = ThreadPoolManager(max_workers=self.max_workers)
        
        # JIT compiled functions
        if self.enable_jit:
            self._setup_jit_functions()
        
        # Performance tracking
        self.perf_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'vectorized_operations': 0,
            'concurrent_batches': 0,
            'jit_compilations': 0,
            'average_solve_time': 0.0,
            'peak_memory_usage': 0,
            'throughput_ops_per_sec': 0.0
        }
        
        logger.info(f"Initialized ScalableLLGSSolver with {max_workers} workers, "
                   f"batch_size={batch_size}, caching={enable_caching}")

    def solve_batch(
        self,
        batch_params: List[Dict[str, Any]],
        concurrent: bool = True
    ) -> List[Dict[str, Any]]:
        """Solve batch of problems with optimization.
        
        Args:
            batch_params: List of solve parameter dictionaries
            concurrent: Enable concurrent processing
            
        Returns:
            List of solution dictionaries
        """
        batch_start = time.time()
        batch_size = len(batch_params)
        
        if self.enable_monitoring:
            self.metrics.increment('batch_solves')
            self.metrics.record('batch_size', batch_size)
        
        try:
            # Check cache for existing solutions
            cached_results, uncached_params, cache_indices = self._check_batch_cache(batch_params)
            
            if len(uncached_params) == 0:
                # All results cached
                self.perf_stats['cache_hits'] += batch_size
                logger.debug(f"Batch fully cached: {batch_size} results")
                return cached_results
            
            # Process uncached parameters
            if concurrent and len(uncached_params) > 1:
                uncached_results = self._solve_batch_concurrent(uncached_params)
                self.perf_stats['concurrent_batches'] += 1
            elif self.enable_vectorization and len(uncached_params) > 1:
                uncached_results = self._solve_batch_vectorized(uncached_params)
                self.perf_stats['vectorized_operations'] += 1
            else:
                uncached_results = self._solve_batch_sequential(uncached_params)
            
            # Merge cached and uncached results
            final_results = self._merge_batch_results(
                cached_results, uncached_results, cache_indices
            )
            
            # Cache new results
            if self.enable_caching:
                self._cache_batch_results(uncached_params, uncached_results)
            
            # Update performance stats
            batch_time = time.time() - batch_start
            throughput = batch_size / batch_time
            self.perf_stats['throughput_ops_per_sec'] = throughput
            
            logger.debug(f"Batch solved: {batch_size} problems in {batch_time:.3f}s "
                        f"({throughput:.1f} ops/sec)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch solve failed: {e}")
            # Return fallback results
            return [self._create_fallback_result(
                params.get('m_initial', np.array([0, 0, 1])),
                params.get('t_span', (0, 1e-9)),
                str(e)
            ) for params in batch_params]

    def solve_adaptive(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Solve with adaptive optimization based on problem characteristics.
        
        Args:
            m_initial: Initial magnetization
            t_span: Time span
            device_params: Device parameters
            **kwargs: Additional arguments
            
        Returns:
            Solution dictionary with optimization metadata
        """
        solve_start = time.time()
        
        # Analyze problem characteristics
        problem_profile = self._analyze_problem(m_initial, t_span, device_params)
        
        # Select optimal strategy
        strategy = self._select_optimization_strategy(problem_profile)
        
        try:
            # Execute with selected strategy
            if strategy == 'vectorized':
                result = self._solve_vectorized_single(
                    m_initial, t_span, device_params, **kwargs
                )
                self.perf_stats['vectorized_operations'] += 1
                
            elif strategy == 'cached':
                cache_key = self._generate_cache_key(m_initial, t_span, device_params)
                result = self.computation_cache.get_or_compute(
                    cache_key,
                    lambda: super().solve(m_initial, t_span, device_params, **kwargs)
                )
                
            elif strategy == 'jit':
                result = self._solve_with_jit(
                    m_initial, t_span, device_params, **kwargs
                )
                
            else:
                # Standard solve
                result = super().solve(m_initial, t_span, device_params, **kwargs)
            
            # Add optimization metadata
            solve_time = time.time() - solve_start
            result.update({
                'optimization_strategy': strategy,
                'problem_profile': problem_profile,
                'solve_time_optimized': solve_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive solve failed: {e}")
            return super().solve(m_initial, t_span, device_params, **kwargs)

    def _check_batch_cache(
        self,
        batch_params: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
        """Check cache for batch parameters."""
        if not self.enable_caching:
            return [], batch_params, list(range(len(batch_params)))
        
        cached_results = []
        uncached_params = []
        cache_indices = []
        
        for i, params in enumerate(batch_params):
            cache_key = self._generate_cache_key(
                params['m_initial'],
                params['t_span'],
                params['device_params']
            )
            
            cached_result = self.result_cache.get(cache_key)
            if cached_result is not None:
                cached_results.append(cached_result)
                self.perf_stats['cache_hits'] += 1
            else:
                uncached_params.append(params)
                cache_indices.append(i)
                self.perf_stats['cache_misses'] += 1
        
        return cached_results, uncached_params, cache_indices

    def _solve_batch_concurrent(
        self,
        batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Solve batch using concurrent processing."""
        def solve_single(params):
            return super(ScalableLLGSSolver, self).solve(**params)
        
        return process_batch_concurrent(
            solve_single,
            batch_params,
            max_workers=self.max_workers
        )

    def _solve_batch_vectorized(
        self,
        batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Solve batch using vectorized operations."""
        if not self.enable_vectorization:
            return self._solve_batch_sequential(batch_params)
        
        try:
            # Extract arrays for vectorized processing
            m_initial_batch = np.array([p['m_initial'] for p in batch_params])
            device_params_batch = [p['device_params'] for p in batch_params]
            
            # Use vectorized solver
            results = self.vectorized_solver.solve_batch(
                m_initial_batch,
                batch_params[0]['t_span'],  # Assume same time span
                device_params_batch
            )
            
            return results
            
        except Exception as e:
            logger.warning(f"Vectorized batch solve failed: {e}. Falling back to sequential.")
            return self._solve_batch_sequential(batch_params)

    def _solve_batch_sequential(
        self,
        batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Solve batch sequentially."""
        results = []
        for params in batch_params:
            result = super().solve(**params)
            results.append(result)
        return results

    def _merge_batch_results(
        self,
        cached_results: List[Dict[str, Any]],
        uncached_results: List[Dict[str, Any]],
        cache_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Merge cached and uncached results in correct order."""
        if not cached_results:
            return uncached_results
        
        final_results = [None] * (len(cached_results) + len(uncached_results))
        
        # Place cached results
        cached_idx = 0
        uncached_idx = 0
        
        for i in range(len(final_results)):
            if i in cache_indices:
                final_results[i] = uncached_results[uncached_idx]
                uncached_idx += 1
            else:
                final_results[i] = cached_results[cached_idx]
                cached_idx += 1
        
        return final_results

    def _cache_batch_results(
        self,
        params_list: List[Dict[str, Any]],
        results_list: List[Dict[str, Any]]
    ) -> None:
        """Cache batch results."""
        for params, result in zip(params_list, results_list):
            cache_key = self._generate_cache_key(
                params['m_initial'],
                params['t_span'],
                params['device_params']
            )
            self.result_cache.set(cache_key, result)

    def _analyze_problem(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze problem characteristics for optimization."""
        t_start, t_end = t_span
        duration = t_end - t_start
        
        # Problem complexity metrics
        damping = device_params.get('damping', 0.01)
        ms = device_params.get('saturation_magnetization', 800e3)
        anisotropy = device_params.get('uniaxial_anisotropy', 1e6)
        
        complexity_score = (duration / 1e-9) * (1 / damping) * (ms / 1e6)
        
        return {
            'duration': duration,
            'complexity_score': complexity_score,
            'initial_magnitude': np.linalg.norm(m_initial),
            'damping': damping,
            'has_high_anisotropy': anisotropy > 1e6
        }

    def _select_optimization_strategy(
        self,
        problem_profile: Dict[str, Any]
    ) -> str:
        """Select optimal strategy based on problem profile."""
        complexity = problem_profile['complexity_score']
        duration = problem_profile['duration']
        
        # Strategy selection logic
        if complexity < 10 and self.enable_caching:
            return 'cached'
        elif duration > 1e-8 and self.enable_vectorization:
            return 'vectorized'
        elif self.enable_jit and complexity > 100:
            return 'jit'
        else:
            return 'standard'

    def _solve_vectorized_single(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Solve single problem using vectorized operations."""
        return self.vectorized_solver.solve_single(
            m_initial, t_span, device_params, **kwargs
        )

    def _solve_with_jit(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Solve using JIT compiled functions."""
        if hasattr(self, 'jit_solve_func'):
            return self.jit_solve_func(m_initial, t_span, device_params)
        else:
            return super().solve(m_initial, t_span, device_params, **kwargs)

    def _setup_jit_functions(self) -> None:
        """Setup JIT compiled functions."""
        try:
            # Create JIT compiled version of key functions
            self.jit_solve_func = jit_compile_if_available(
                super().solve,
                cache=True
            )
            self.perf_stats['jit_compilations'] += 1
            logger.info("JIT compilation successful")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
            self.enable_jit = False

    @lru_cache(maxsize=1000)
    def _generate_cache_key(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params_tuple: tuple
    ) -> str:
        """Generate cache key for parameters."""
        m_key = tuple(np.round(m_initial, 6))
        t_key = tuple(np.round(t_span, 12))
        
        # Convert device params to sorted tuple for consistency
        if isinstance(device_params_tuple, dict):
            device_params_tuple = tuple(sorted(device_params_tuple.items()))
        
        return f"{m_key}_{t_key}_{hash(device_params_tuple)}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        stats = self.perf_stats.copy()
        
        total_operations = stats['cache_hits'] + stats['cache_misses']
        if total_operations > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_operations
            stats['cache_efficiency'] = stats['cache_hit_rate']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['cache_efficiency'] = 0.0
        
        # Add resource utilization
        stats['thread_pool_utilization'] = self.thread_pool.get_utilization()
        stats['memory_efficiency'] = self._estimate_memory_efficiency()
        
        return stats

    def _estimate_memory_efficiency(self) -> float:
        """Estimate memory efficiency."""
        if self.enable_caching:
            cache_utilization = len(self.result_cache) / self.result_cache.maxsize
            return min(1.0, cache_utilization * 1.2)  # Bonus for good cache usage
        return 0.8  # Default efficiency without caching

    def optimize_for_workload(self, workload_profile: Dict[str, Any]) -> None:
        """Optimize solver configuration for specific workload."""
        batch_size = workload_profile.get('typical_batch_size', self.batch_size)
        problem_complexity = workload_profile.get('complexity', 'medium')
        memory_constraint = workload_profile.get('memory_limit_mb', 1000)
        
        # Adjust parameters based on workload
        if problem_complexity == 'high':
            self.max_workers = min(self.max_workers, 2)  # Reduce for complex problems
            self.timeout *= 2
        elif problem_complexity == 'low':
            self.max_workers = min(8, self.max_workers * 2)  # Increase for simple problems
        
        # Adjust cache size based on memory constraint
        if self.enable_caching:
            max_cache_entries = memory_constraint * 10  # Rough estimate
            self.result_cache.resize(min(max_cache_entries, 5000))
        
        # Adjust batch size
        self.batch_size = adaptive_batching.optimize_batch_size(
            current_size=batch_size,
            problem_complexity=problem_complexity,
            available_workers=self.max_workers
        )
        
        logger.info(f"Optimized for workload: workers={self.max_workers}, "
                   f"batch_size={self.batch_size}, cache_size={len(self.result_cache)}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
        
        if self.enable_caching:
            self.result_cache.clear()
            self.computation_cache.clear()
        
        logger.info("ScalableLLGSSolver cleanup completed")