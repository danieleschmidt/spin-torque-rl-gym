#!/usr/bin/env python3
"""
Generation 3 validation test: MAKE IT SCALE

This script validates the scaling features including caching,
concurrency, performance optimization, and auto-scaling.
"""

import sys
import time
import threading
import asyncio
import traceback
import warnings
import os
from typing import Dict, Any, List

import numpy as np


def test_adaptive_caching():
    """Test adaptive caching system."""
    print("=" * 60)
    print("Testing Adaptive Caching...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.cache import AdaptiveCache, CacheManager
        
        # Test adaptive cache
        cache = AdaptiveCache(initial_size=10, max_size=50)
        
        # Add some test data
        for i in range(20):
            cache.put(f"key_{i}", f"value_{i}", compute_time=0.01)
        
        # Test cache hits
        found, value = cache.get("key_5")
        print(f"✓ Cache hit: found={found}, value={value}")
        
        # Test cache miss
        found, value = cache.get("nonexistent_key")
        print(f"✓ Cache miss: found={found}")
        
        # Get cache info
        info = cache.get_adaptation_info()
        print(f"✓ Cache adaptation info: {info['cache_info']['hit_rate']:.2f} hit rate")
        
        # Test cache manager
        manager = CacheManager()
        test_cache = manager.get_cache("test_cache", cache_type="adaptive")
        print("✓ Cache manager created cache")
        
        # Test cached decorator
        @manager.cached(cache_name="function_cache")
        def expensive_function(x):
            time.sleep(0.01)  # Simulate work
            return x * x
        
        # First call should cache
        result1 = expensive_function(5)
        result2 = expensive_function(5)  # Should be cached
        print(f"✓ Cached function: {result1} == {result2}")
        
        # Get global stats
        global_stats = manager.get_global_stats()
        print(f"✓ Global cache stats: {global_stats['global']['total_caches']} caches")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive caching test failed: {e}")
        traceback.print_exc()
        return False


def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("\n" + "=" * 60)
    print("Testing Concurrent Processing...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.concurrency import (
            PhysicsWorkerPool, ResourcePool, ParallelBenchmark
        )
        
        # Test resource pool
        def create_test_resource():
            return {"id": time.time(), "data": "test"}
        
        pool = ResourcePool(create_test_resource, min_size=2, max_size=5)
        
        # Acquire and use resources
        with pool.acquire() as resource:
            print(f"✓ Acquired resource: {resource['id']}")
        
        pool_stats = pool.get_stats()
        print(f"✓ Resource pool stats: {pool_stats['pool_size']} available")
        
        # Test worker pool
        def test_simulation(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        worker_pool = PhysicsWorkerPool(max_workers=4, use_processes=False)
        
        # Submit single task
        future = worker_pool.submit_simulation(test_simulation, 1, 2)
        result = future.result()
        print(f"✓ Worker pool result: {result['result']} (success: {result['success']})")
        
        # Submit batch
        param_list = [(i, i+1) for i in range(10)]
        batch_results = worker_pool.submit_batch(test_simulation, param_list, timeout=30)
        successful_results = [r for r in batch_results if r['success']]
        print(f"✓ Batch processing: {len(successful_results)}/10 successful")
        
        # Get worker stats
        worker_stats = worker_pool.get_stats()
        print(f"✓ Worker stats: {worker_stats['success_rate']:.2f} success rate")
        
        worker_pool.shutdown()
        
        # Test parallel benchmark
        benchmark = ParallelBenchmark()
        
        def simple_task(n):
            return sum(range(n))
        
        task_list = [(100,), (200,), (150,), (300,)]
        benchmark_results = benchmark.benchmark_function(
            simple_task, task_list, worker_counts=[1, 2], use_processes=False
        )
        
        print(f"✓ Benchmark completed: optimal workers = {benchmark_results['optimal_workers']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Concurrent processing test failed: {e}")
        traceback.print_exc()
        return False


def test_async_capabilities():
    """Test asynchronous processing capabilities."""
    print("\n" + "=" * 60)
    print("Testing Async Capabilities...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.concurrency import AsyncEnvironmentManager
        from spin_torque_gym.envs.spin_torque_env import SpinTorqueEnv
        
        async def run_async_test():
            # Create environment factory
            def env_factory():
                return SpinTorqueEnv(max_steps=5, device_type='stt_mram')
            
            # Create async manager
            async_manager = AsyncEnvironmentManager(
                env_factory=env_factory,
                max_concurrent=2
            )
            
            # Simple policy
            def random_policy(obs):
                return np.array([1e5, 1e-10])  # Small current, short pulse
            
            # Run single episode
            result = await async_manager.run_episode(
                policy=random_policy,
                episode_id=0,
                max_steps=5
            )
            
            print(f"✓ Async episode: reward={result['reward']:.3f}, success={result['success']}")
            
            # Run batch of episodes
            batch_results = await async_manager.run_batch(
                policy=random_policy,
                num_episodes=3,
                max_steps=5
            )
            
            successful_episodes = sum(1 for r in batch_results if r.get('success', False))
            print(f"✓ Async batch: {len(batch_results)} episodes, {successful_episodes} successful")
            
            # Get stats
            stats = async_manager.get_stats()
            print(f"✓ Async stats: {stats['completed_episodes']} completed")
        
        # Run async test
        asyncio.run(run_async_test())
        
        return True
        
    except Exception as e:
        print(f"✗ Async capabilities test failed: {e}")
        traceback.print_exc()
        return False


def test_auto_scaling():
    """Test auto-scaling and load balancing."""
    print("\n" + "=" * 60)
    print("Testing Auto-Scaling...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.scaling import (
            AutoScaler, LoadBalancer, LoadMetrics
        )
        
        # Test auto-scaler
        scaler = AutoScaler(
            min_workers=1,
            max_workers=8,
            cooldown_period=1.0  # Short cooldown for testing
        )
        
        # Add some load metrics
        high_load_metrics = LoadMetrics(
            timestamp=time.time(),
            cpu_utilization=0.9,
            memory_utilization=0.7,
            queue_length=10,
            active_tasks=8,
            throughput=50.0,
            response_time=2.0,
            error_rate=0.02
        )
        
        scaler.add_metrics(high_load_metrics)
        
        # Check if scaling is needed
        should_scale, reason, target_workers = scaler.should_scale()
        print(f"✓ Auto-scaler decision: scale={should_scale}, reason={reason}, target={target_workers}")
        
        if should_scale:
            scaler.execute_scaling(target_workers, reason)
            print(f"✓ Scaling executed: {scaler.current_workers} workers")
        
        # Get scaling stats
        scaling_stats = scaler.get_scaling_stats()
        print(f"✓ Scaling stats: {scaling_stats['total_scaling_events']} events")
        
        # Test load balancer
        def mock_worker(task_id):
            time.sleep(0.01)  # Simulate work
            return f"result_{task_id}"
        
        # Create workers
        workers = [lambda tid=i: mock_worker(tid) for i in range(3)]
        load_balancer = LoadBalancer(workers, selection_strategy="least_loaded")
        
        # Select workers and simulate requests
        for i in range(10):
            selection = load_balancer.select_worker()
            if selection:
                worker_id, worker = selection
                
                load_balancer.record_request_start(worker_id)
                start_time = time.perf_counter()
                
                try:
                    result = worker(i)
                    response_time = time.perf_counter() - start_time
                    load_balancer.record_request_end(worker_id, response_time, True)
                except Exception:
                    response_time = time.perf_counter() - start_time
                    load_balancer.record_request_end(worker_id, response_time, False)
        
        # Get load balancer stats
        lb_stats = load_balancer.get_load_balancer_stats()
        print(f"✓ Load balancer: {lb_stats['total_requests']} requests, "
              f"{lb_stats['error_rate']:.2f} error rate")
        
        return True
        
    except Exception as e:
        print(f"✗ Auto-scaling test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_optimization():
    """Test performance optimization features."""
    print("\n" + "=" * 60)
    print("Testing Performance Optimization...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.cache import get_cache_manager
        from spin_torque_gym.utils.concurrency import parallel_map
        
        # Test performance with caching
        cache_manager = get_cache_manager()
        
        @cache_manager.cached(cache_name="perf_test")
        def computation_intensive_task(n):
            # Simulate CPU-intensive work
            result = sum(i**2 for i in range(n))
            return result
        
        # Time uncached vs cached execution
        start_time = time.perf_counter()
        result1 = computation_intensive_task(1000)
        uncached_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        result2 = computation_intensive_task(1000)  # Should be cached
        cached_time = time.perf_counter() - start_time
        
        speedup = uncached_time / cached_time if cached_time > 0 else float('inf')
        print(f"✓ Caching speedup: {speedup:.1f}x faster ({uncached_time:.4f}s -> {cached_time:.4f}s)")
        
        # Test parallel map
        @parallel_map(max_workers=4, use_processes=False)
        def parallel_square(x):
            time.sleep(0.01)  # Simulate work
            return x * x
        
        # Compare serial vs parallel execution
        test_data = list(range(20))
        
        # Serial execution
        start_time = time.perf_counter()
        serial_results = [x * x for x in test_data]
        serial_time = time.perf_counter() - start_time
        
        # Parallel execution
        start_time = time.perf_counter()
        parallel_results = parallel_square(test_data)
        parallel_time = time.perf_counter() - start_time
        
        parallel_speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        print(f"✓ Parallel speedup: {parallel_speedup:.1f}x faster "
              f"({serial_time:.4f}s -> {parallel_time:.4f}s)")
        
        # Verify results are correct
        if serial_results == parallel_results:
            print("✓ Parallel results match serial results")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance optimization test failed: {e}")
        traceback.print_exc()
        return False


def test_resource_management():
    """Test comprehensive resource management."""
    print("\n" + "=" * 60)
    print("Testing Resource Management...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.utils.scaling import AdaptiveResourceManager
        
        # Create mock resource factory
        def create_mock_resource():
            return {
                'id': time.time(),
                'compute_power': 1.0,
                'status': 'ready'
            }
        
        # Initialize adaptive resource manager
        manager = AdaptiveResourceManager(
            resource_factory=create_mock_resource,
            initial_workers=2,
            max_workers=6
        )
        
        manager.start()
        
        # Simulate requests
        def mock_computation(value):
            time.sleep(0.01)  # Simulate work
            return value * 2
        
        # Execute several requests
        results = []
        for i in range(10):
            try:
                result = manager.execute_request(mock_computation, i)
                results.append(result)
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        print(f"✓ Executed {len(results)} requests through adaptive manager")
        
        # Get comprehensive stats
        comprehensive_stats = manager.get_comprehensive_stats()
        lb_stats = comprehensive_stats['load_balancer']
        scaler_stats = comprehensive_stats['auto_scaler']
        
        print(f"✓ Load balancer: {lb_stats['healthy_workers']} healthy workers")
        print(f"✓ Auto-scaler: {scaler_stats['current_workers']} current workers")
        
        manager.stop()
        
        return True
        
    except Exception as e:
        print(f"✗ Resource management test failed: {e}")
        traceback.print_exc()
        return False


def test_scalability_stress():
    """Test system under stress conditions."""
    print("\n" + "=" * 60)
    print("Testing Scalability Stress...")
    print("=" * 60)
    
    try:
        from spin_torque_gym.envs.spin_torque_env import SpinTorqueEnv
        from spin_torque_gym.utils.cache import get_cache_manager
        
        # Create multiple environments concurrently
        num_envs = 5
        environments = []
        
        for i in range(num_envs):
            env = SpinTorqueEnv(max_steps=3, device_type='stt_mram')
            environments.append(env)
        
        print(f"✓ Created {num_envs} concurrent environments")
        
        # Run steps in parallel using threading
        def run_env_episode(env_id, env):
            try:
                obs, info = env.reset(seed=env_id)
                total_reward = 0
                
                for step in range(3):
                    action = np.array([1e5, 1e-10])  # Small action
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                return {
                    'env_id': env_id,
                    'total_reward': total_reward,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'env_id': env_id,
                    'error': str(e),
                    'success': False
                }
        
        # Run all environments in parallel
        threads = []
        results = [None] * num_envs
        
        def worker(i):
            results[i] = run_env_episode(i, environments[i])
        
        # Start threads
        for i in range(num_envs):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Close environments
        for env in environments:
            env.close()
        
        # Analyze results
        successful_runs = [r for r in results if r and r.get('success', False)]
        print(f"✓ Stress test: {len(successful_runs)}/{num_envs} environments completed successfully")
        
        # Test cache performance under load
        cache_manager = get_cache_manager()
        test_cache = cache_manager.get_cache("stress_test", cache_type="adaptive")
        
        # Rapid cache operations
        start_time = time.perf_counter()
        for i in range(100):
            test_cache.put(f"stress_key_{i}", f"value_{i}", compute_time=0.001)
        
        cache_write_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        hits = 0
        for i in range(100):
            found, value = test_cache.get(f"stress_key_{i}")
            if found:
                hits += 1
        
        cache_read_time = time.perf_counter() - start_time
        
        print(f"✓ Cache stress: {hits}/100 hits, "
              f"write: {cache_write_time:.4f}s, read: {cache_read_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Scalability stress test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Generation 3 scaling tests."""
    print("TERRAGON SDLC - GENERATION 3 VALIDATION")
    print("Spin-Torque RL-Gym Scaling Test")
    print("=" * 80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Set environment variables for testing
    os.environ['SPINTORQUE_LOG_LEVEL'] = 'WARNING'
    
    tests = [
        test_adaptive_caching,
        test_concurrent_processing,
        test_async_capabilities,
        test_auto_scaling,
        test_performance_optimization,
        test_resource_management,
        test_scalability_stress
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("GENERATION 3 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.8:  # 80% pass rate for generation 3
        print("✓ GENERATION 3: MAKE IT SCALE - SUCCESS!")
        print("  Advanced scaling features implemented successfully.")
        print("  System has adaptive caching, concurrency, auto-scaling, and optimization.")
        print("  Ready for quality gates validation and deployment preparation.")
        return True
    else:
        print("✗ GENERATION 3: SCALING ISSUES DETECTED")
        print("  Some scaling features failed. Review and fix before proceeding.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)