#!/usr/bin/env python3
"""Simple RL training test for Generation 1"""

import gymnasium as gym
import numpy as np


def simple_random_policy(obs):
    """Random policy for testing"""
    return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1)])

def main():
    print("🧪 Testing Simple RL Training - Generation 1")

    # Create environment
    env = gym.make('SpinTorque-v0', device_type='stt_mram')

    # Test episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    steps = 0

    for step in range(50):  # Short test episode
        action = simple_random_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}")

        if done or truncated:
            print(f"Episode finished at step {step}")
            break

    print(f"✅ Episode completed: {steps} steps, total reward: {total_reward:.3f}")
    env.close()

    # Test device creation
    from spin_torque_gym.devices import create_device
    device = create_device('stt_mram')
    print(f"✅ Device created: {device.__class__.__name__}")

    print("🎉 Generation 1 Simple Implementation: SUCCESS")

if __name__ == "__main__":
    main()
