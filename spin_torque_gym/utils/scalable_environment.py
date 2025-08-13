"""Scalable environment with auto-scaling, load balancing and resource optimization."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np

from .cache import adaptive_cache
from .monitoring import MetricsCollector, PerformanceProfiler
from .performance_optimization import EnvironmentPool, VectorizedEnvironment
from .scaling import AutoScaler, LoadBalancer, ResourceOptimizer

logger = logging.getLogger(__name__)


class ScalableEnvironmentManager:
    """Manages multiple environments with auto-scaling and load balancing."""

    def __init__(
        self,
        env_factory: callable,
        initial_pool_size: int = 4,
        max_pool_size: int = 16,
        min_pool_size: int = 1,
        enable_auto_scaling: bool = True,
        enable_load_balancing: bool = True,
        enable_vectorization: bool = True,
        enable_caching: bool = True,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        monitoring_interval: float = 10.0
    ):
        """Initialize scalable environment manager.
        
        Args:
            env_factory: Factory function to create environment instances
            initial_pool_size: Initial number of environments
            max_pool_size: Maximum number of environments
            min_pool_size: Minimum number of environments
            enable_auto_scaling: Enable automatic scaling
            enable_load_balancing: Enable load balancing
            enable_vectorization: Enable vectorized operations
            enable_caching: Enable result caching
            scale_up_threshold: CPU utilization threshold for scaling up
            scale_down_threshold: CPU utilization threshold for scaling down
            monitoring_interval: Monitoring interval in seconds
        """
        self.env_factory = env_factory
        self.initial_pool_size = initial_pool_size
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_load_balancing = enable_load_balancing
        self.enable_vectorization = enable_vectorization
        self.enable_caching = enable_caching
        self.monitoring_interval = monitoring_interval

        # Initialize components
        self.environment_pool = EnvironmentPool(env_factory, initial_pool_size)

        if enable_auto_scaling:
            self.auto_scaler = AutoScaler(
                min_instances=min_pool_size,
                max_instances=max_pool_size,
                scale_up_threshold=scale_up_threshold,
                scale_down_threshold=scale_down_threshold
            )

        if enable_load_balancing:
            self.load_balancer = LoadBalancer(strategy='least_loaded')

        if enable_vectorization:
            self.vectorized_env = VectorizedEnvironment(env_factory)

        self.resource_optimizer = ResourceOptimizer()

        # Caching
        if enable_caching:
            self.state_cache = adaptive_cache(max_size=1000)
            self.result_cache = adaptive_cache(max_size=2000)

        # Monitoring
        self.metrics = MetricsCollector()
        self.profiler = PerformanceProfiler()

        # Statistics
        self.stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'concurrent_episodes': 0,
            'scale_events': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_throughput': 0.0,
            'resource_efficiency': 0.0,
            'load_balance_factor': 1.0
        }

        logger.info(f"Initialized ScalableEnvironmentManager with {initial_pool_size} environments")

    def run_episode_batch(
        self,
        policies: List[callable],
        max_steps: int = 1000,
        concurrent: bool = True
    ) -> List[Dict[str, Any]]:
        """Run batch of episodes with optimization.
        
        Args:
            policies: List of policy functions
            max_steps: Maximum steps per episode
            concurrent: Enable concurrent execution
            
        Returns:
            List of episode results
        """
        batch_start = time.time()
        batch_size = len(policies)

        self.stats['concurrent_episodes'] = batch_size
        self.metrics.increment('episode_batches')
        self.metrics.record('batch_size', batch_size)

        try:
            if concurrent and batch_size > 1:
                results = self._run_batch_concurrent(policies, max_steps)
            elif self.enable_vectorization and batch_size > 1:
                results = self._run_batch_vectorized(policies, max_steps)
            else:
                results = self._run_batch_sequential(policies, max_steps)

            # Update statistics
            batch_time = time.time() - batch_start
            throughput = batch_size / batch_time
            self.stats['average_throughput'] = throughput
            self.stats['total_episodes'] += batch_size

            # Auto-scaling decision
            if self.enable_auto_scaling:
                self._update_scaling_metrics(throughput, batch_size)

            logger.debug(f"Batch completed: {batch_size} episodes in {batch_time:.3f}s "
                        f"({throughput:.1f} eps/sec)")

            return results

        except Exception as e:
            logger.error(f"Episode batch failed: {e}")
            return [{'error': str(e), 'success': False} for _ in policies]

    def run_step_batch(
        self,
        env_states: List[Tuple[Any, int]],  # (env_id, action)
        actions: List[Union[np.ndarray, int, float]]
    ) -> List[Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]]:
        """Run batch of environment steps.
        
        Args:
            env_states: List of (environment_id, state) tuples
            actions: List of actions for each environment
            
        Returns:
            List of step results
        """
        step_start = time.time()
        batch_size = len(env_states)

        self.metrics.increment('step_batches')
        self.metrics.record('step_batch_size', batch_size)

        try:
            # Load balancing
            if self.enable_load_balancing:
                env_assignments = self.load_balancer.assign_batch(env_states)
            else:
                env_assignments = list(enumerate(env_states))

            # Process batch
            if self.enable_vectorization and batch_size > 1:
                results = self._step_batch_vectorized(env_assignments, actions)
            else:
                results = self._step_batch_distributed(env_assignments, actions)

            # Update statistics
            step_time = time.time() - step_start
            step_throughput = batch_size / step_time
            self.stats['total_steps'] += batch_size

            self.metrics.record('step_batch_time', step_time)
            self.metrics.record('step_throughput', step_throughput)

            return results

        except Exception as e:
            logger.error(f"Step batch failed: {e}")
            # Return safe fallback results
            return self._create_fallback_step_results(batch_size)

    def _run_batch_concurrent(
        self,
        policies: List[callable],
        max_steps: int
    ) -> List[Dict[str, Any]]:
        """Run episode batch concurrently."""
        def run_single_episode(policy_idx):
            policy = policies[policy_idx]
            env = self.environment_pool.get_environment()

            try:
                return self._run_single_episode(env, policy, max_steps)
            finally:
                self.environment_pool.return_environment(env)

        # Use thread pool for concurrent execution
        with ThreadPoolExecutor(max_workers=min(len(policies), self.max_pool_size)) as executor:
            futures = [executor.submit(run_single_episode, i) for i in range(len(policies))]
            results = [future.result() for future in futures]

        return results

    def _run_batch_vectorized(
        self,
        policies: List[callable],
        max_steps: int
    ) -> List[Dict[str, Any]]:
        """Run episode batch using vectorized environment."""
        if not self.enable_vectorization:
            return self._run_batch_sequential(policies, max_steps)

        try:
            return self.vectorized_env.run_episode_batch(policies, max_steps)
        except Exception as e:
            logger.warning(f"Vectorized batch failed: {e}. Falling back to sequential.")
            return self._run_batch_sequential(policies, max_steps)

    def _run_batch_sequential(
        self,
        policies: List[callable],
        max_steps: int
    ) -> List[Dict[str, Any]]:
        """Run episode batch sequentially."""
        results = []
        env = self.environment_pool.get_environment()

        try:
            for policy in policies:
                result = self._run_single_episode(env, policy, max_steps)
                results.append(result)
        finally:
            self.environment_pool.return_environment(env)

        return results

    def _run_single_episode(
        self,
        env: gym.Env,
        policy: callable,
        max_steps: int
    ) -> Dict[str, Any]:
        """Run single episode with caching and optimization."""
        episode_start = time.time()

        # Check cache for similar episodes
        if self.enable_caching:
            policy_key = self._get_policy_key(policy)
            cached_result = self.result_cache.get(policy_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
            self.stats['cache_misses'] += 1

        # Run episode
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        episode_data = []

        for step in range(max_steps):
            # Get action from policy
            action = policy(obs)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Store step data
            episode_data.append({
                'step': step,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated
            })

            if terminated or truncated:
                break

        # Create result
        episode_time = time.time() - episode_start
        result = {
            'total_reward': total_reward,
            'steps': steps,
            'episode_time': episode_time,
            'success': True,
            'episode_data': episode_data,
            'final_state': obs.copy() if isinstance(obs, np.ndarray) else obs
        }

        # Cache result
        if self.enable_caching:
            self.result_cache.set(policy_key, result)

        return result

    def _step_batch_vectorized(
        self,
        env_assignments: List[Tuple[int, Any]],
        actions: List[Any]
    ) -> List[Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]]:
        """Process step batch using vectorized operations."""
        try:
            return self.vectorized_env.step_batch(env_assignments, actions)
        except Exception as e:
            logger.warning(f"Vectorized step batch failed: {e}. Using distributed approach.")
            return self._step_batch_distributed(env_assignments, actions)

    def _step_batch_distributed(
        self,
        env_assignments: List[Tuple[int, Any]],
        actions: List[Any]
    ) -> List[Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]]:
        """Process step batch across distributed environments."""
        results = []

        for (env_idx, env_state), action in zip(env_assignments, actions):
            env = self.environment_pool.get_environment_by_id(env_idx)

            try:
                # Restore environment state if needed
                if hasattr(env, '_set_state') and env_state is not None:
                    env._set_state(env_state)

                # Take step
                result = env.step(action)
                results.append(result)

            except Exception as e:
                logger.warning(f"Step failed for env {env_idx}: {e}")
                results.append(self._create_fallback_step_result())
            finally:
                self.environment_pool.return_environment(env)

        return results

    def _create_fallback_step_results(self, batch_size: int) -> List[Tuple]:
        """Create fallback step results for batch."""
        return [self._create_fallback_step_result() for _ in range(batch_size)]

    def _create_fallback_step_result(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Create safe fallback step result."""
        obs = np.zeros(12, dtype=np.float32)  # Default observation size
        reward = 0.0
        terminated = True
        truncated = False
        info = {'fallback': True, 'error': 'Step failed'}

        return obs, reward, terminated, truncated, info

    def _get_policy_key(self, policy: callable) -> str:
        """Generate cache key for policy."""
        try:
            return f"policy_{hash(policy)}_{id(policy)}"
        except:
            return f"policy_uncacheable_{id(policy)}"

    def _update_scaling_metrics(self, throughput: float, batch_size: int) -> None:
        """Update metrics for auto-scaling decisions."""
        current_load = batch_size / self.environment_pool.size()
        resource_usage = self.resource_optimizer.get_current_usage()

        scaling_decision = self.auto_scaler.should_scale(
            current_load=current_load,
            resource_usage=resource_usage,
            throughput=throughput
        )

        if scaling_decision['scale_up']:
            new_size = min(self.max_pool_size, self.environment_pool.size() + 1)
            self.environment_pool.resize(new_size)
            self.stats['scale_events'] += 1
            logger.info(f"Scaled up to {new_size} environments")

        elif scaling_decision['scale_down']:
            new_size = max(self.min_pool_size, self.environment_pool.size() - 1)
            self.environment_pool.resize(new_size)
            self.stats['scale_events'] += 1
            logger.info(f"Scaled down to {new_size} environments")

        # Update load balance factor
        self.stats['load_balance_factor'] = self.load_balancer.get_balance_factor()

        # Update resource efficiency
        self.stats['resource_efficiency'] = self.resource_optimizer.get_efficiency_score()

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        return {
            'current_pool_size': self.environment_pool.size(),
            'max_pool_size': self.max_pool_size,
            'min_pool_size': self.min_pool_size,
            'active_environments': self.environment_pool.active_count(),
            'idle_environments': self.environment_pool.idle_count(),
            'load_balance_factor': self.stats['load_balance_factor'],
            'resource_efficiency': self.stats['resource_efficiency'],
            'auto_scaling_enabled': self.enable_auto_scaling,
            'scale_events': self.stats['scale_events'],
            'cache_hit_rate': (
                self.stats['cache_hits'] /
                max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
            )
        }

    def optimize_for_workload(self, workload_profile: Dict[str, Any]) -> None:
        """Optimize manager configuration for specific workload."""
        expected_concurrent = workload_profile.get('concurrent_episodes', 4)
        episode_duration = workload_profile.get('avg_episode_duration', 10.0)
        memory_constraint = workload_profile.get('memory_limit_mb', 2000)

        # Adjust pool size
        optimal_pool_size = min(
            self.max_pool_size,
            max(self.min_pool_size, expected_concurrent)
        )
        self.environment_pool.resize(optimal_pool_size)

        # Adjust caching based on memory constraint
        if self.enable_caching:
            cache_entries = memory_constraint // 10  # Rough estimate
            self.state_cache.resize(min(cache_entries, 1000))
            self.result_cache.resize(min(cache_entries * 2, 2000))

        # Configure auto-scaler
        if self.enable_auto_scaling:
            self.auto_scaler.configure(
                response_time=episode_duration / 10,
                aggressiveness='medium' if expected_concurrent > 8 else 'conservative'
            )

        logger.info(f"Optimized for workload: pool_size={optimal_pool_size}, "
                   f"expected_concurrent={expected_concurrent}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        stats = self.stats.copy()

        # Add current metrics
        if hasattr(self, 'metrics'):
            stats['current_metrics'] = self.metrics.get_metrics()

        # Add scaling status
        stats['scaling_status'] = self.get_scaling_status()

        # Add resource usage
        stats['resource_usage'] = self.resource_optimizer.get_detailed_usage()

        # Calculate efficiency metrics
        total_episodes = max(stats['total_episodes'], 1)
        stats['episodes_per_scale_event'] = total_episodes / max(stats['scale_events'], 1)
        stats['average_concurrent_efficiency'] = (
            stats['concurrent_episodes'] / self.environment_pool.size()
        )

        return stats

    def shutdown(self) -> None:
        """Shutdown manager and cleanup resources."""
        logger.info("Shutting down ScalableEnvironmentManager")

        # Shutdown environment pool
        self.environment_pool.shutdown()

        # Clear caches
        if self.enable_caching:
            self.state_cache.clear()
            self.result_cache.clear()

        # Cleanup monitoring
        if hasattr(self, 'profiler'):
            self.profiler.cleanup()

        logger.info("ScalableEnvironmentManager shutdown completed")
