"""SpinTorqueEnv: Gymnasium environment for spin-torque magnetic device control."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SpinTorqueEnv(gym.Env):
    """
    Gymnasium environment simulating a spin-torque magnetic device.

    The state is a 2D magnetization vector [mx, my] normalized to the unit circle.
    Actions are discrete current pulse directions that drive the magnetization.
    The goal is to reach a target magnetization state.

    Observation space: Box(-1, 1, shape=(2,), dtype=float32)
    Action space: Discrete(4) — +x, -x, +y, -y pulses
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_state=None,
        max_steps: int = 20,
        dt: float = 0.1,
        damping: float = 0.1,
        pulse_amplitude: float = 0.5,
    ):
        super().__init__()

        self.target_state = np.array(
            target_state if target_state is not None else [1.0, 0.0],
            dtype=np.float32,
        )
        self.max_steps = max_steps
        self.dt = dt
        self.damping = damping
        self.pulse_amplitude = pulse_amplitude

        # Pulse directions: 0=+x, 1=-x, 2=+y, 3=-y
        self._pulse_directions = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._state: np.ndarray = np.zeros(2, dtype=np.float32)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Random angle on the unit circle
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._state = np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        self._step_count = 0

        return self._state.copy(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._state = self._apply_pulse(action)
        self._step_count += 1

        reward = float(-np.linalg.norm(self._state - self.target_state))

        terminated = self._step_count >= self.max_steps
        truncated = False

        return self._state.copy(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _apply_pulse(self, action: int) -> np.ndarray:
        """Simplified LLG-like magnetization update."""
        pulse = self._pulse_directions[action] * self.pulse_amplitude
        m = self._state + self.dt * (pulse - self.damping * self._state)
        norm = np.linalg.norm(m)
        if norm > 0:
            m = m / norm
        return m.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def render(self):
        pass  # No rendering implemented

    def close(self):
        pass
