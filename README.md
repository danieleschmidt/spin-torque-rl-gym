# spin-torque-rl-gym

Gymnasium environment for spin-torque magnetic device control using reinforcement learning. State is 2D magnetization vector, actions are current pulses.

## Install

```bash
pip install numpy gymnasium
```

## Usage

```python
from spin_torque_rl.env import SpinTorqueEnv

env = SpinTorqueEnv()
obs, info = env.reset()
```

## Environment Details

- **Observation space:** `Box(-1, 1, shape=(2,), dtype=float32)` — 2D magnetization vector `[mx, my]` on the unit circle
- **Action space:** `Discrete(4)` — current pulse directions: `+x`, `-x`, `+y`, `-y`
- **Reward:** `-||state - target||` (negative distance; higher = closer to target)
- **Episode length:** 20 steps

## Training Example

```python
from spin_torque_rl.env import SpinTorqueEnv
from spin_torque_rl.policy import SimplePolicyGradient

env = SpinTorqueEnv()
policy = SimplePolicyGradient()
episode_rewards = policy.train(env, n_episodes=100)
print(f"Final avg reward: {sum(episode_rewards[-10:]) / 10:.4f}")
```

## Physics

The magnetization update follows a simplified Landau-Lifshitz-Gilbert (LLG) model:

```
m += dt * (pulse - damping * m)
m = m / ||m||   # normalize to unit circle
```

With `dt=0.1`, `damping=0.1`, `pulse_amplitude=0.5`.
