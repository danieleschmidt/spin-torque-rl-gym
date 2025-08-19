#!/usr/bin/env python3
"""
Production health check script for Spin-Torque RL-Gym.
Returns exit code 0 for healthy, 1 for unhealthy.
"""

import sys
import time
import traceback
from typing import Dict, Any

try:
    import gymnasium as gym
    import spin_torque_gym
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def check_basic_imports() -> bool:
    """Check that basic imports work."""
    try:
        # Test core imports
        import spin_torque_gym.envs
        import spin_torque_gym.devices
        import spin_torque_gym.physics
        import spin_torque_gym.utils
        return True
    except Exception as e:
        print(f"Import check failed: {e}")
        return False


def check_environment_creation() -> bool:
    """Check that environments can be created."""
    try:
        env = gym.make('SpinTorque-v0', max_steps=3)
        env.close()
        return True
    except Exception as e:
        print(f"Environment creation failed: {e}")
        return False


def check_basic_operations() -> bool:
    """Check that basic operations work."""
    try:
        # Create environment
        env = gym.make('SpinTorque-v0', max_steps=2)
        
        # Test reset
        obs, info = env.reset()
        if obs is None or not isinstance(obs, np.ndarray):
            raise ValueError("Invalid observation from reset")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if obs is None or not isinstance(obs, np.ndarray):
            raise ValueError("Invalid observation from step")
        
        if not isinstance(reward, (int, float)):
            raise ValueError("Invalid reward type")
        
        env.close()
        return True
    except Exception as e:
        print(f"Basic operations failed: {e}")
        return False


def check_performance_systems() -> bool:
    """Check that performance optimization systems are working."""
    try:
        from spin_torque_gym.utils.performance_optimization import get_optimizer
        
        optimizer = get_optimizer()
        report = optimizer.get_performance_report()
        
        if not isinstance(report, dict):
            raise ValueError("Invalid performance report")
        
        return True
    except Exception as e:
        print(f"Performance systems check failed: {e}")
        return False


def check_monitoring_systems() -> bool:
    """Check that monitoring systems are working."""
    try:
        from spin_torque_gym.utils.monitoring import EnvironmentMonitor
        
        monitor = EnvironmentMonitor(log_level="ERROR")
        health_report = monitor.get_health_report()
        
        if not isinstance(health_report, dict):
            raise ValueError("Invalid health report")
        
        return True
    except Exception as e:
        print(f"Monitoring systems check failed: {e}")
        return False


def run_comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive health check."""
    start_time = time.time()
    
    checks = {
        'basic_imports': check_basic_imports,
        'environment_creation': check_environment_creation,
        'basic_operations': check_basic_operations,
        'performance_systems': check_performance_systems,
        'monitoring_systems': check_monitoring_systems
    }
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks.items():
        try:
            start = time.time()
            passed = check_func()
            duration = time.time() - start
            
            results[check_name] = {
                'passed': passed,
                'duration': duration
            }
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            results[check_name] = {
                'passed': False,
                'error': str(e),
                'duration': time.time() - start
            }
            all_passed = False
    
    total_duration = time.time() - start_time
    
    health_status = {
        'timestamp': time.time(),
        'overall_health': 'healthy' if all_passed else 'unhealthy',
        'total_duration': total_duration,
        'checks': results,
        'version': getattr(spin_torque_gym, '__version__', 'unknown'),
        'environment': 'production'
    }
    
    return health_status


def main():
    """Main health check function."""
    try:
        print("🔍 Running production health check...")
        
        health_status = run_comprehensive_health_check()
        
        # Print results
        print(f"Health Status: {health_status['overall_health']}")
        print(f"Total Duration: {health_status['total_duration']:.3f}s")
        print(f"Version: {health_status['version']}")
        
        # Print individual check results
        for check_name, result in health_status['checks'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result['duration']
            print(f"  {check_name}: {status} ({duration:.3f}s)")
            
            if 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Return appropriate exit code
        if health_status['overall_health'] == 'healthy':
            print("🎯 Health check completed successfully")
            sys.exit(0)
        else:
            print("⚠️ Health check failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Health check script failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()