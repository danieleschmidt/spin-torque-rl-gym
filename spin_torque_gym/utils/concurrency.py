"""Concurrency and parallel processing utilities.

This module provides thread pools, process pools, and async capabilities
for high-performance parallel execution of physics simulations and
environment steps.
"""

import asyncio
import concurrent.futures
import multiprocessing
import queue
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .logging_config import get_logger, PerformanceLogger


class ResourcePool:
    """Generic resource pool with automatic management."""
    
    def __init__(self, 
                 resource_factory: Callable[[], Any],
                 min_size: int = 2,
                 max_size: int = 10,
                 idle_timeout: float = 300.0):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size  
            idle_timeout: Seconds before idle resources are removed
        """
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self.pool = queue.Queue()
        self.active_count = 0
        self.created_count = 0
        self.lock = threading.Lock()
        
        self.logger = get_logger("Concurrency.ResourcePool")
        
        # Pre-populate with minimum resources
        for _ in range(min_size):
            resource = self._create_resource()
            self.pool.put((resource, time.time()))
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        with self.lock:
            if self.created_count >= self.max_size:
                raise RuntimeError("Maximum pool size exceeded")
            
            resource = self.resource_factory()
            self.created_count += 1
            self.logger.debug(f"Created resource {self.created_count}/{self.max_size}")
            return resource
    
    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """Acquire a resource from the pool.
        
        Args:
            timeout: Timeout in seconds
            
        Yields:
            Resource instance
        """
        start_time = time.time()
        resource = None
        
        try:
            # Try to get existing resource
            try:
                resource, last_used = self.pool.get(timeout=timeout)
                
                # Check if resource is still valid/fresh
                if time.time() - last_used > self.idle_timeout:
                    # Resource too old, create new one
                    resource = self._create_resource()
                    
            except queue.Empty:
                # No available resources, create new one
                resource = self._create_resource()
            
            with self.lock:
                self.active_count += 1
            
            yield resource
            
        finally:
            # Return resource to pool
            if resource is not None:
                try:
                    self.pool.put((resource, time.time()), block=False)
                except queue.Full:
                    # Pool is full, discard resource
                    pass
                
                with self.lock:
                    self.active_count -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self.pool.qsize(),
            'active_count': self.active_count,
            'created_count': self.created_count,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'utilization': self.active_count / self.created_count if self.created_count > 0 else 0.0
        }


class PhysicsWorkerPool:
    """Specialized worker pool for physics simulations."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False):
        """Initialize physics worker pool.
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            use_processes: Use processes instead of threads
        """
        if max_workers is None:
            max_workers = min(8, multiprocessing.cpu_count())
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers)
        
        self.logger = get_logger("Concurrency.PhysicsPool")
        self.performance_logger = PerformanceLogger()
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_compute_time = 0.0
        
        self.logger.info(
            f"Initialized {'process' if use_processes else 'thread'} pool "
            f"with {max_workers} workers"
        )
    
    def submit_simulation(self, 
                         simulation_func: Callable,
                         *args, 
                         **kwargs) -> concurrent.futures.Future:
        """Submit physics simulation task.
        
        Args:
            simulation_func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future object for result
        """
        future = self.executor.submit(self._wrapped_simulation, 
                                     simulation_func, *args, **kwargs)
        
        task_id = id(future)
        self.active_tasks[task_id] = {
            'future': future,
            'start_time': time.time(),
            'function': simulation_func.__name__
        }
        
        # Add completion callback
        future.add_done_callback(self._task_completed)
        
        return future
    
    def _wrapped_simulation(self, func: Callable, *args, **kwargs) -> Any:
        """Wrapped simulation with error handling."""
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            compute_time = time.perf_counter() - start_time
            
            return {
                'result': result,
                'compute_time': compute_time,
                'success': True,
                'error': None
            }
        
        except Exception as e:
            compute_time = time.perf_counter() - start_time
            
            return {
                'result': None,
                'compute_time': compute_time,
                'success': False,
                'error': str(e)
            }
    
    def _task_completed(self, future: concurrent.futures.Future) -> None:
        """Handle task completion."""
        task_id = id(future)
        
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            total_time = time.time() - task_info['start_time']
            
            try:
                result = future.result()
                if result['success']:
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
                    self.logger.warning(f"Task failed: {result['error']}")
                
                self.total_compute_time += result['compute_time']
                
            except Exception as e:
                self.failed_tasks += 1
                self.logger.error(f"Task exception: {e}")
            
            del self.active_tasks[task_id]
    
    def submit_batch(self, 
                    simulation_func: Callable,
                    param_list: List[Tuple],
                    timeout: Optional[float] = None) -> List[Any]:
        """Submit batch of simulations.
        
        Args:
            simulation_func: Function to execute
            param_list: List of parameter tuples for each simulation
            timeout: Timeout for entire batch
            
        Returns:
            List of results
        """
        # Submit all tasks
        futures = []
        for params in param_list:
            future = self.submit_simulation(simulation_func, *params)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    'result': None,
                    'success': False,
                    'error': str(e),
                    'compute_time': 0.0
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 0.0
        avg_compute_time = self.total_compute_time / self.completed_tasks if self.completed_tasks > 0 else 0.0
        
        return {
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'average_compute_time': avg_compute_time,
            'total_compute_time': self.total_compute_time
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)
        self.logger.info("Worker pool shutdown")


class AsyncEnvironmentManager:
    """Asynchronous environment manager for concurrent episode execution."""
    
    def __init__(self, 
                 env_factory: Callable,
                 max_concurrent: int = 4):
        """Initialize async environment manager.
        
        Args:
            env_factory: Function to create environment instances
            max_concurrent: Maximum concurrent environments
        """
        self.env_factory = env_factory
        self.max_concurrent = max_concurrent
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = get_logger("Concurrency.AsyncEnv")
        
        self.active_episodes = 0
        self.completed_episodes = 0
        self.episode_times = []
    
    async def run_episode(self, 
                         policy: Callable,
                         episode_id: int = 0,
                         max_steps: int = 1000) -> Dict[str, Any]:
        """Run single episode asynchronously.
        
        Args:
            policy: Policy function (obs -> action)
            episode_id: Episode identifier
            max_steps: Maximum steps per episode
            
        Returns:
            Episode results
        """
        async with self.semaphore:
            start_time = time.time()
            self.active_episodes += 1
            
            try:
                # Create environment (this should be thread-safe)
                env = self.env_factory()
                
                # Run episode
                obs, info = env.reset()
                episode_reward = 0.0
                step_count = 0
                
                for step in range(max_steps):
                    action = policy(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                    
                    # Yield control periodically
                    if step % 10 == 0:
                        await asyncio.sleep(0)
                
                env.close()
                
                episode_time = time.time() - start_time
                self.episode_times.append(episode_time)
                
                result = {
                    'episode_id': episode_id,
                    'reward': episode_reward,
                    'steps': step_count,
                    'time': episode_time,
                    'success': info.get('is_success', False),
                    'terminated': terminated,
                    'truncated': truncated
                }
                
                self.completed_episodes += 1
                return result
                
            except Exception as e:
                self.logger.error(f"Episode {episode_id} failed: {e}")
                return {
                    'episode_id': episode_id,
                    'error': str(e),
                    'success': False,
                    'reward': 0.0,
                    'steps': 0,
                    'time': time.time() - start_time
                }
            
            finally:
                self.active_episodes -= 1
    
    async def run_batch(self, 
                       policy: Callable,
                       num_episodes: int,
                       max_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run batch of episodes concurrently.
        
        Args:
            policy: Policy function
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            List of episode results
        """
        # Create tasks
        tasks = []
        for i in range(num_episodes):
            task = asyncio.create_task(
                self.run_episode(policy, episode_id=i, max_steps=max_steps)
            )
            tasks.append(task)
        
        # Wait for all episodes to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                clean_results.append({
                    'episode_id': i,
                    'error': str(result),
                    'success': False,
                    'reward': 0.0,
                    'steps': 0,
                    'time': 0.0
                })
            else:
                clean_results.append(result)
        
        return clean_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async environment statistics."""
        avg_time = np.mean(self.episode_times) if self.episode_times else 0.0
        
        return {
            'max_concurrent': self.max_concurrent,
            'active_episodes': self.active_episodes,
            'completed_episodes': self.completed_episodes,
            'average_episode_time': avg_time,
            'total_episode_time': sum(self.episode_times)
        }


class ParallelBenchmark:
    """Benchmarking tool for parallel performance."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.logger = get_logger("Concurrency.Benchmark")
        self.results = {}
    
    def benchmark_function(self, 
                          func: Callable,
                          args_list: List[Tuple],
                          worker_counts: List[int] = [1, 2, 4, 8],
                          use_processes: bool = False) -> Dict[str, Any]:
        """Benchmark function with different worker counts.
        
        Args:
            func: Function to benchmark
            args_list: List of argument tuples
            worker_counts: Worker counts to test
            use_processes: Use processes vs threads
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for worker_count in worker_counts:
            self.logger.info(f"Benchmarking with {worker_count} workers")
            
            if worker_count == 1:
                # Serial execution
                start_time = time.perf_counter()
                serial_results = []
                
                for args in args_list:
                    result = func(*args)
                    serial_results.append(result)
                
                serial_time = time.perf_counter() - start_time
                
                results[worker_count] = {
                    'execution_time': serial_time,
                    'throughput': len(args_list) / serial_time,
                    'speedup': 1.0,
                    'efficiency': 1.0,
                    'worker_type': 'serial'
                }
                
                baseline_time = serial_time
                
            else:
                # Parallel execution
                pool = PhysicsWorkerPool(worker_count, use_processes)
                
                start_time = time.perf_counter()
                batch_results = pool.submit_batch(func, args_list, timeout=300)
                parallel_time = time.perf_counter() - start_time
                
                speedup = baseline_time / parallel_time if parallel_time > 0 else 0.0
                efficiency = speedup / worker_count
                
                results[worker_count] = {
                    'execution_time': parallel_time,
                    'throughput': len(args_list) / parallel_time,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'worker_type': 'process' if use_processes else 'thread'
                }
                
                pool.shutdown()
        
        # Find optimal worker count
        best_throughput = 0
        best_worker_count = 1
        
        for worker_count, result in results.items():
            if result['throughput'] > best_throughput:
                best_throughput = result['throughput']
                best_worker_count = worker_count
        
        summary = {
            'worker_results': results,
            'optimal_workers': best_worker_count,
            'max_throughput': best_throughput,
            'task_count': len(args_list)
        }
        
        self.results[func.__name__] = summary
        return summary
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations."""
        recommendations = {}
        
        for func_name, results in self.results.items():
            optimal_workers = results['optimal_workers']
            max_throughput = results['max_throughput']
            
            # Efficiency analysis
            worker_results = results['worker_results']
            efficiencies = [r['efficiency'] for r in worker_results.values()]
            avg_efficiency = np.mean(efficiencies)
            
            recommendations[func_name] = {
                'recommended_workers': optimal_workers,
                'expected_throughput': max_throughput,
                'average_efficiency': avg_efficiency,
                'parallelization_benefit': max_throughput / worker_results[1]['throughput'] if 1 in worker_results else 1.0
            }
        
        return recommendations


# Decorators for easy parallel execution
def parallel_map(max_workers: int = None, use_processes: bool = False):
    """Decorator for parallel map operations.
    
    Args:
        max_workers: Maximum number of workers
        use_processes: Use processes instead of threads
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(iterable, *args, **kwargs):
            pool = PhysicsWorkerPool(max_workers, use_processes)
            
            # Convert iterable to list of argument tuples
            arg_list = [(item,) + args for item in iterable]
            
            results = pool.submit_batch(func, arg_list)
            pool.shutdown()
            
            # Extract actual results
            return [r['result'] for r in results if r['success']]
        
        return wrapper
    return decorator


def async_cached(cache_name: str = "async_default"):
    """Decorator for async function caching."""
    from .cache import get_cache_manager
    
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        cache = cache_manager.get_cache(cache_name)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = cache.cache._make_key(func.__name__, *args, **kwargs)
            
            # Try cache first
            found, result = cache.get(key)
            if found:
                return result
            
            # Compute result
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            compute_time = time.perf_counter() - start_time
            
            # Cache result
            cache.put(key, result, compute_time)
            
            return result
        
        return wrapper
    return decorator