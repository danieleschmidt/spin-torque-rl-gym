"""
Spin-Torque RL-Gym: Reinforcement Learning for Spintronic Device Control

A Gymnasium-compatible environment for training RL agents to control
spin-torque devices in neuromorphic computing applications.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"

from gymnasium.envs.registration import register

# Register main environment
register(
    id='SpinTorque-v0',
    entry_point='spin_torque_gym.envs:SpinTorqueEnv',
    max_episode_steps=1000,
)

# Register multi-device environment  
register(
    id='SpinTorqueArray-v0',
    entry_point='spin_torque_gym.envs:SpinTorqueArrayEnv',
    max_episode_steps=2000,
)

# Register skyrmion environment
register(
    id='SkyrmionRacetrack-v0', 
    entry_point='spin_torque_gym.envs:SkyrmionRacetrackEnv',
    max_episode_steps=1500,
)