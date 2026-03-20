"""Tests for spin_torque_rl package."""

import numpy as np
import pytest

from spin_torque_rl.env import SpinTorqueEnv
from spin_torque_rl.policy import SimplePolicyGradient
from spin_torque_rl.demo import run_demo


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:

    def test_env_observation_space(self):
        env = SpinTorqueEnv()
        obs_space = env.observation_space
        assert obs_space.shape == (2,)
        assert obs_space.low.tolist() == [-1.0, -1.0]
        assert obs_space.high.tolist() == [1.0, 1.0]
        assert str(obs_space.dtype) == "float32"

    def test_env_action_space(self):
        env = SpinTorqueEnv()
        action_space = env.action_space
        assert action_space.n == 4
        for a in range(4):
            assert action_space.contains(a)

    def test_env_reset_returns_valid_state(self):
        env = SpinTorqueEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        # State must be on unit circle (within tolerance)
        assert abs(np.linalg.norm(obs) - 1.0) < 1e-5

    def test_env_step_returns_tuple(self):
        env = SpinTorqueEnv()
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_env_step_terminates_after_20_steps(self):
        env = SpinTorqueEnv()
        env.reset()
        terminated = False
        for i in range(20):
            _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True

    def test_env_reward_closer_is_higher(self):
        """Reward should be higher (less negative) when state is closer to target."""
        env = SpinTorqueEnv(target_state=[1.0, 0.0])
        # Manually set state close to target
        env.reset()
        env._state = np.array([1.0, 0.0], dtype=np.float32)
        _, reward_close, _, _, _ = env.step(0)

        env.reset()
        env._state = np.array([-1.0, 0.0], dtype=np.float32)
        _, reward_far, _, _, _ = env.step(0)

        # Close state should yield higher reward (less negative distance)
        assert reward_close > reward_far

    def test_env_state_normalized(self):
        """After each step, the magnetization magnitude should remain ~1."""
        env = SpinTorqueEnv()
        obs, _ = env.reset()
        assert abs(np.linalg.norm(obs) - 1.0) < 1e-4

        for action in range(4):
            env.reset()
            for _ in range(5):
                obs, _, terminated, _, _ = env.step(action)
                assert abs(np.linalg.norm(obs) - 1.0) < 1e-4, (
                    f"State magnitude {np.linalg.norm(obs):.6f} not ~1 for action {action}"
                )
                if terminated:
                    break

    def test_env_reward_is_negative_distance(self):
        """Reward = -||state - target||, so it should always be <= 0."""
        env = SpinTorqueEnv()
        env.reset()
        for _ in range(5):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert reward <= 0.0
            if terminated:
                break

    def test_env_custom_target(self):
        env = SpinTorqueEnv(target_state=[0.0, 1.0])
        np.testing.assert_array_almost_equal(env.target_state, [0.0, 1.0])


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestPolicy:

    def test_policy_forward_output_shape(self):
        policy = SimplePolicyGradient()
        obs = np.array([1.0, 0.0], dtype=np.float32)
        probs = policy.forward(obs)
        assert probs.shape == (4,)

    def test_policy_forward_sums_to_one(self):
        policy = SimplePolicyGradient()
        obs = np.array([0.5, -0.5], dtype=np.float32)
        probs = policy.forward(obs)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_policy_forward_all_positive(self):
        policy = SimplePolicyGradient()
        obs = np.array([1.0, 0.0], dtype=np.float32)
        probs = policy.forward(obs)
        assert (probs >= 0).all()

    def test_policy_select_action_valid(self):
        policy = SimplePolicyGradient()
        obs = np.array([1.0, 0.0], dtype=np.float32)
        action, log_prob = policy.select_action(obs)
        assert 0 <= action <= 3
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0  # log of probability in [0,1] is <= 0

    def test_policy_train_runs(self):
        """Training should run without errors and return a reward list."""
        env = SpinTorqueEnv()
        policy = SimplePolicyGradient()
        rewards = policy.train(env, n_episodes=5)
        assert isinstance(rewards, list)
        assert len(rewards) == 5
        for r in rewards:
            assert isinstance(r, float)

    def test_policy_update_runs(self):
        """Update should not raise errors."""
        policy = SimplePolicyGradient()
        log_probs = [-0.5, -1.0, -0.3]
        rewards = [-1.0, -0.5, -0.2]
        policy.update(log_probs, rewards, lr=0.01)


# ---------------------------------------------------------------------------
# Demo test
# ---------------------------------------------------------------------------

class TestDemo:

    def test_demo_runs(self):
        """run_demo should complete and return a float."""
        avg_reward = run_demo(n_episodes=5)
        assert isinstance(avg_reward, float)
        assert avg_reward <= 0.0  # reward is always negative distance
