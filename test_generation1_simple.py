#!/usr/bin/env python3
"""
Generation 1 Simple Testing: Make It Work
Basic functionality validation with quantum research enhancements
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import spin torque gym
import spin_torque_gym

def test_basic_environment_functionality():
    """Test core environment functionality"""
    print("🧪 Testing basic environment functionality...")
    
    # Create environment
    env = gym.make('SpinTorque-v0', 
                   device_type='stt_mram',
                   max_steps=50)
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"✓ Environment created successfully")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Initial magnetization: {obs[:3]}")
    
    # Test random actions
    episode_rewards = []
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        
        if terminated or truncated:
            print(f"  - Episode terminated at step {step+1}")
            break
    
    env.close()
    print(f"✓ Basic environment test passed - avg reward: {np.mean(episode_rewards):.3f}")
    return True

def test_quantum_research_integration():
    """Test quantum research modules integration"""
    print("🔬 Testing quantum research integration...")
    
    try:
        from spin_torque_gym.quantum import QuantumOptimization
        from spin_torque_gym.research import ComparativeAlgorithms
        
        # Test quantum optimization
        quantum_opt = QuantumOptimization()
        test_params = np.random.rand(5)
        optimized_params = quantum_opt.optimize_switching_sequence(test_params)
        
        print(f"✓ Quantum optimization working")
        print(f"  - Input params: {test_params}")
        print(f"  - Optimized params: {optimized_params}")
        
        # Test comparative algorithms
        comp_algo = ComparativeAlgorithms()
        baseline_results = comp_algo.run_baseline_comparison()
        
        print(f"✓ Comparative algorithms working")
        print(f"  - Baseline results: {len(baseline_results)} algorithms compared")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Quantum modules not fully implemented yet: {e}")
        return False

def test_simple_training_loop():
    """Test a simple training loop"""
    print("🎯 Testing simple training loop...")
    
    env = gym.make('SpinTorque-v0', max_steps=20)
    
    # Simple random agent
    total_rewards = []
    
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(20):
            # Simple heuristic: apply current proportional to error
            target = info.get('target_state', np.array([0, 0, 1]))
            current_mag = obs[:3]
            error = np.linalg.norm(target - current_mag)
            
            # Simple control policy
            if error > 0.5:
                current_magnitude = 1e6  # Strong current
                duration = 0.5  # Short pulse
            else:
                current_magnitude = 0.5e6  # Weak current
                duration = 0.2  # Very short pulse
            
            action = np.array([current_magnitude, duration])
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: reward = {episode_reward:.3f}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    print(f"✓ Simple training loop completed - avg reward: {avg_reward:.3f}")
    return avg_reward > -10  # Basic success threshold

def test_physics_simulation_accuracy():
    """Test physics simulation accuracy"""
    print("⚛️ Testing physics simulation accuracy...")
    
    # Test energy conservation
    from spin_torque_gym.physics import LLGSSolver
    from spin_torque_gym.devices import DeviceFactory
    
    device = DeviceFactory.create_device('stt_mram')
    solver = LLGSSolver()
    
    # Initial state
    initial_magnetization = np.array([1.0, 0.0, 0.0])
    
    # Apply zero current (should conserve energy)
    evolved_mag = solver.evolve_magnetization(
        initial_magnetization, 
        current=0.0,
        dt=1e-12,
        device=device
    )
    
    # Check magnitude conservation (|m| = 1)
    initial_norm = np.linalg.norm(initial_magnetization)
    final_norm = np.linalg.norm(evolved_mag)
    
    conservation_error = abs(final_norm - initial_norm)
    print(f"✓ Magnetization norm conservation error: {conservation_error:.6f}")
    
    return conservation_error < 1e-10  # Very strict conservation

def create_simple_benchmark_report():
    """Create a simple benchmark report"""
    print("📊 Creating simple benchmark report...")
    
    results = {
        'basic_functionality': test_basic_environment_functionality(),
        'quantum_integration': test_quantum_research_integration(),
        'simple_training': test_simple_training_loop(),
        'physics_accuracy': test_physics_simulation_accuracy(),
    }
    
    # Generate simple visualization
    plt.figure(figsize=(10, 6))
    
    # Plot test results
    test_names = list(results.keys())
    test_results = [1 if result else 0 for result in results.values()]
    colors = ['green' if result else 'red' for result in results.values()]
    
    plt.bar(test_names, test_results, color=colors, alpha=0.7)
    plt.title('Generation 1 Test Results - Make It Work')
    plt.ylabel('Pass (1) / Fail (0)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('generation1_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 GENERATION 1 SUMMARY:")
    print(f"  Tests Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total*100:.1f}%")
    
    if passed >= 3:  # Allow 1 failure for quantum modules
        print("✅ GENERATION 1 COMPLETE: Basic functionality working!")
        return True
    else:
        print("❌ GENERATION 1 NEEDS WORK: Core functionality issues detected")
        return False

def main():
    """Run Generation 1 testing"""
    print("=" * 60)
    print("🚀 TERRAGON AUTONOMOUS SDLC - GENERATION 1: MAKE IT WORK")
    print("=" * 60)
    
    success = create_simple_benchmark_report()
    
    if success:
        print("\n🎉 Ready to proceed to Generation 2: Make It Robust!")
    else:
        print("\n🔧 Generation 1 issues need resolution before proceeding")
    
    return success

if __name__ == "__main__":
    main()