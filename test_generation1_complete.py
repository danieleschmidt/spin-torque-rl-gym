#!/usr/bin/env python3
"""
Comprehensive test script for Generation 1: MAKE IT WORK

This script validates the basic functionality of the SpinTorque RL-Gym
environment implementation across all major components.
"""

import sys
import traceback
import warnings

import gymnasium as gym
import numpy as np


def test_environment_creation():
    """Test basic environment creation and initialization."""
    print("=" * 60)
    print("Testing Environment Creation...")
    print("=" * 60)

    try:
        # Test basic environment creation
        env = gym.make('SpinTorque-v0')
        print("✓ Basic SpinTorque-v0 environment created successfully")

        # Test with custom parameters
        env_custom = gym.make('SpinTorque-v0',
                             device_type='stt_mram',
                             max_steps=50,
                             temperature=300.0,
                             action_mode='continuous')
        print("✓ Custom environment created successfully")

        # Test different device types
        env_sot = gym.make('SpinTorque-v0', device_type='sot_mram')
        print("✓ SOT-MRAM environment created successfully")

        env_vcma = gym.make('SpinTorque-v0', device_type='vcma_mram')
        print("✓ VCMA-MRAM environment created successfully")

        env_skyrmion = gym.make('SpinTorque-v0', device_type='skyrmion')
        print("✓ Skyrmion environment created successfully")

        return True

    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_environment_spaces():
    """Test environment observation and action spaces."""
    print("\n" + "=" * 60)
    print("Testing Environment Spaces...")
    print("=" * 60)

    try:
        env = gym.make('SpinTorque-v0')

        # Test observation space
        obs_space = env.observation_space
        print(f"✓ Observation space: {obs_space}")

        # Test action space
        action_space = env.action_space
        print(f"✓ Action space: {action_space}")

        # Test discrete action mode
        env_discrete = gym.make('SpinTorque-v0', action_mode='discrete')
        print(f"✓ Discrete action space: {env_discrete.action_space}")

        # Test dictionary observation mode
        env_dict = gym.make('SpinTorque-v0', observation_mode='dict')
        print(f"✓ Dictionary observation space: {env_dict.observation_space}")

        return True

    except Exception as e:
        print(f"✗ Environment spaces test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_environment_loop():
    """Test basic environment reset-step loop."""
    print("\n" + "=" * 60)
    print("Testing Basic Environment Loop...")
    print("=" * 60)

    try:
        env = gym.make('SpinTorque-v0', max_steps=10)

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Reset successful - obs shape: {obs.shape}, info keys: {list(info.keys())}")

        # Test step loop
        total_reward = 0
        for step in range(5):
            # Random action
            action = env.action_space.sample()

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            print(f"  Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break

            obs = next_obs

        print(f"✓ Environment loop completed - Total reward: {total_reward:.3f}")
        env.close()

        return True

    except Exception as e:
        print(f"✗ Environment loop test failed: {e}")
        traceback.print_exc()
        return False


def test_physics_solver():
    """Test physics solver functionality."""
    print("\n" + "=" * 60)
    print("Testing Physics Solver...")
    print("=" * 60)

    try:
        from spin_torque_gym.physics.simple_solver import SimpleLLGSSolver

        # Create solver
        solver = SimpleLLGSSolver(method='euler', timeout=0.1)
        print("✓ SimpleLLGSSolver created successfully")

        # Test solving with simple parameters
        device_params = {
            'damping': 0.01,
            'saturation_magnetization': 800e3,
            'uniaxial_anisotropy': 1e6,
            'volume': 1e-24,
            'easy_axis': np.array([0, 0, 1]),
            'polarization': 0.7
        }

        # Initial magnetization
        m_initial = np.array([1, 0, 0])

        # Define simple current function
        def current_func(t):
            return 1e6 if t < 1e-10 else 0.0

        # Solve
        result = solver.solve(
            m_initial=m_initial,
            time_span=(0, 2e-10),
            device_params=device_params,
            current_func=current_func,
            thermal_noise=False
        )

        print(f"✓ Solver completed - Success: {result['success']}")
        print(f"  Final magnetization: {result['m'][-1]}")
        print(f"  Solution time: {result.get('solve_time', 0):.4f}s")

        # Test solver info
        solver_info = solver.get_solver_info()
        print(f"✓ Solver info: {solver_info}")

        return True

    except Exception as e:
        print(f"✗ Physics solver test failed: {e}")
        traceback.print_exc()
        return False


def test_device_models():
    """Test device model functionality."""
    print("\n" + "=" * 60)
    print("Testing Device Models...")
    print("=" * 60)

    try:
        from spin_torque_gym.devices import DeviceFactory

        factory = DeviceFactory()
        print("✓ DeviceFactory created successfully")

        # Test STT-MRAM device
        device_params = factory.get_default_parameters('stt_mram')
        stt_device = factory.create_device('stt_mram', device_params)
        print(f"✓ STT-MRAM device created: {stt_device}")

        # Test device functionality
        magnetization = np.array([0, 0, 1])
        applied_field = np.array([0, 0, 0])

        h_eff = stt_device.compute_effective_field(magnetization, applied_field)
        print(f"✓ Effective field computed: {h_eff}")

        resistance = stt_device.compute_resistance(magnetization)
        print(f"✓ Resistance computed: {resistance:.2f} Ω")

        # Test other device types
        sot_device = factory.create_device('sot_mram', factory.get_default_parameters('sot_mram'))
        print(f"✓ SOT-MRAM device created: {sot_device}")

        vcma_device = factory.create_device('vcma_mram', factory.get_default_parameters('vcma_mram'))
        print(f"✓ VCMA-MRAM device created: {vcma_device}")

        skyrmion_device = factory.create_device('skyrmion', factory.get_default_parameters('skyrmion'))
        print(f"✓ Skyrmion device created: {skyrmion_device}")

        return True

    except Exception as e:
        print(f"✗ Device models test failed: {e}")
        traceback.print_exc()
        return False


def test_reward_functions():
    """Test reward function functionality."""
    print("\n" + "=" * 60)
    print("Testing Reward Functions...")
    print("=" * 60)

    try:
        from spin_torque_gym.rewards import CompositeReward

        # Test composite reward with default components
        reward_components = {
            'success': {
                'weight': 10.0,
                'function': lambda obs, action, next_obs, info: 10.0 if info.get('is_success', False) else 0.0
            },
            'energy': {
                'weight': -0.1,
                'function': lambda obs, action, next_obs, info: -info.get('step_energy', 0.0) / 1e-12
            },
            'progress': {
                'weight': 1.0,
                'function': lambda obs, action, next_obs, info: info.get('alignment_improvement', 0.0)
            }
        }

        reward_fn = CompositeReward(reward_components)
        print("✓ CompositeReward created successfully")

        # Test reward computation
        test_info = {
            'is_success': True,
            'step_energy': 1e-13,
            'alignment_improvement': 0.1
        }

        reward = reward_fn.compute(None, None, None, test_info)
        print(f"✓ Reward computed: {reward:.3f}")

        # Test component statistics
        stats = reward_fn.get_component_statistics()
        print(f"✓ Component statistics: {list(stats.keys())}")

        return True

    except Exception as e:
        print(f"✗ Reward functions test failed: {e}")
        traceback.print_exc()
        return False


def test_material_database():
    """Test material database functionality."""
    print("\n" + "=" * 60)
    print("Testing Material Database...")
    print("=" * 60)

    try:
        from spin_torque_gym.physics import MaterialDatabase

        # Create database
        db = MaterialDatabase()
        print("✓ MaterialDatabase created successfully")

        # Test material retrieval
        cofeb = db.get_material('CoFeB')
        print(f"✓ CoFeB material retrieved: Ms={cofeb.saturation_magnetization:.0f} A/m")

        # Test temperature adjustment
        temp_props = db.get_temperature_adjusted_properties('CoFeB', 350.0)
        print(f"✓ Temperature-adjusted properties: Ms={temp_props['saturation_magnetization']:.0f} A/m")

        # Test material list
        materials = db.list_materials()
        print(f"✓ Available materials: {materials}")

        return True

    except Exception as e:
        print(f"✗ Material database test failed: {e}")
        traceback.print_exc()
        return False


def test_complete_episode():
    """Test a complete training episode."""
    print("\n" + "=" * 60)
    print("Testing Complete Episode...")
    print("=" * 60)

    try:
        # Create environment
        env = gym.make('SpinTorque-v0',
                      max_steps=20,
                      success_threshold=0.8,
                      temperature=300.0)

        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"✓ Episode started - Initial alignment: {info.get('current_alignment', 0.0):.3f}")

        total_reward = 0
        episode_data = []

        # Run episode
        for step in range(20):
            # Simple policy: apply moderate current toward target
            if hasattr(env.observation_space, 'shape'):
                # Vector observation
                target = obs[3:6]  # Target magnetization
                current_mag = obs[0:3]  # Current magnetization

                # Simple alignment-based policy
                alignment = np.dot(current_mag, target)
                if alignment < 0.8:
                    action = np.array([1e6, 1e-9])  # Apply current
                else:
                    action = np.array([0.0, 1e-12])  # No current
            else:
                # Random action as fallback
                action = env.action_space.sample()

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            episode_data.append({
                'step': step,
                'reward': reward,
                'alignment': info.get('current_alignment', 0.0),
                'energy': info.get('step_energy', 0.0),
                'success': info.get('is_success', False)
            })

            print(f"  Step {step:2d}: reward={reward:6.3f}, alignment={info.get('current_alignment', 0.0):.3f}, success={info.get('is_success', False)}")

            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break

            obs = next_obs

        print(f"✓ Episode completed - Total reward: {total_reward:.3f}")

        # Analyze episode
        analysis = env.analyze_episode()
        print(f"✓ Episode analysis: {analysis}")

        # Get performance stats
        perf_stats = env.get_performance_stats()
        print(f"✓ Performance stats keys: {list(perf_stats.keys())}")

        env.close()

        return True

    except Exception as e:
        print(f"✗ Complete episode test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Generation 1 tests."""
    print("TERRAGON SDLC - GENERATION 1 VALIDATION")
    print("Spin-Torque RL-Gym Basic Functionality Test")
    print("=" * 80)

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    tests = [
        test_environment_creation,
        test_environment_spaces,
        test_basic_environment_loop,
        test_physics_solver,
        test_device_models,
        test_reward_functions,
        test_material_database,
        test_complete_episode
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
    print("GENERATION 1 TEST SUMMARY")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("✓ GENERATION 1: MAKE IT WORK - COMPLETED SUCCESSFULLY!")
        print("  All basic functionality is working correctly.")
        print("  Ready to proceed to Generation 2: MAKE IT ROBUST")
        return True
    else:
        print("✗ GENERATION 1: BASIC FUNCTIONALITY ISSUES DETECTED")
        print("  Some tests failed. Please review and fix issues before proceeding.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
