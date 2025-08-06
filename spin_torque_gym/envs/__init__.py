"""Gymnasium environments for spintronic device control.

This module provides RL environments for training agents to control
various spintronic devices including STT-MRAM, SOT-MRAM, and others.
"""

from gymnasium.envs.registration import register
from .spin_torque_env import SpinTorqueEnv
from .array_env import SpinTorqueArrayEnv
from .skyrmion_env import SkyrmionRacetrackEnv

# Register environments with Gymnasium
register(
    id='SpinTorque-v0',
    entry_point='spin_torque_gym.envs:SpinTorqueEnv',
    max_episode_steps=100,
    kwargs={'device_type': 'stt_mram'}
)

register(
    id='SpinTorqueArray-v0', 
    entry_point='spin_torque_gym.envs:SpinTorqueArrayEnv',
    max_episode_steps=200,
    kwargs={'array_size': (4, 4), 'device_type': 'stt_mram'}
)

register(
    id='SkyrmionRacetrack-v0',
    entry_point='spin_torque_gym.envs:SkyrmionRacetrackEnv',
    max_episode_steps=150,
    kwargs={'track_length': 1000e-9, 'n_skyrmions': 1}
)

__all__ = [
    "SpinTorqueEnv",
    "SpinTorqueArrayEnv", 
    "SkyrmionRacetrackEnv"
]