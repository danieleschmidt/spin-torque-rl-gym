"""Demo: train a SimplePolicyGradient on SpinTorqueEnv."""

from spin_torque_rl.env import SpinTorqueEnv
from spin_torque_rl.policy import SimplePolicyGradient
import numpy as np


def run_demo(n_episodes: int = 50) -> float:
    """Train the policy for n_episodes and return the average episode reward."""
    env = SpinTorqueEnv()
    policy = SimplePolicyGradient()

    rewards = policy.train(env, n_episodes=n_episodes)
    avg_reward = float(np.mean(rewards))
    return avg_reward


if __name__ == "__main__":
    avg = run_demo()
    print(f"Average reward over 50 episodes: {avg:.4f}")
