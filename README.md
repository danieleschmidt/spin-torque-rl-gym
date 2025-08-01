# Spin-Torque RL-Gym

A Gymnasium-compatible environment that simulates discrete magnetic states for RL agents to learn optimal programming of spin-torque devices. Mirrors experimental protocols from recent physical device papers, enabling AI-driven discovery of efficient magnetization switching sequences.

## Overview

Spin-Torque RL-Gym provides a realistic simulation environment where reinforcement learning agents learn to control spin-torque devices used in neuromorphic computing and magnetic memory (MRAM). By modeling the complex dynamics of magnetic switching, agents discover optimal pulse sequences that minimize energy consumption while ensuring reliable state transitions.

## Key Features

- **Physically Accurate**: Based on Landau-Lifshitz-Gilbert-Slonczewski equations
- **Gymnasium Compatible**: Drop-in replacement for standard RL environments
- **Multi-Device Support**: STT-MRAM, SOT-MRAM, VCMA, and skyrmion devices
- **Hardware Validated**: Calibrated against experimental device data
- **Energy Aware**: Tracks switching energy and thermal effects
- **Visualization**: Real-time magnetization dynamics visualization

## Installation

```bash
# Basic installation
pip install spin-torque-rl-gym

# With visualization support
pip install spin-torque-rl-gym[viz]

# With JAX acceleration
pip install spin-torque-rl-gym[jax]

# Development installation
git clone https://github.com/yourusername/spin-torque-rl-gym
cd spin-torque-rl-gym
pip install -e ".[dev]"
```

## Quick Start

### Basic RL Training

```python
import gymnasium as gym
import spin_torque_gym
from stable_baselines3 import PPO

# Create environment
env = gym.make(
    'SpinTorque-v0',
    device_type='stt_mram',
    target_states=[0, 1, 0, 1],  # Target magnetization pattern
    max_steps=100
)

# Train RL agent
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    verbose=1
)

model.learn(total_timesteps=1_000_000)

# Test learned policy
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    if done:
        print(f"Success! Energy used: {info['total_energy']:.2f} pJ")
        break
```

### Custom Device Configuration

```python
from spin_torque_gym.devices import CustomMTJ

# Define custom magnetic tunnel junction
custom_device = CustomMTJ(
    size=(50e-9, 100e-9, 2e-9),  # 50×100×2 nm³
    ms=800e3,  # Saturation magnetization (A/m)
    damping=0.01,  # Gilbert damping
    polarization=0.7,  # Spin polarization
    resistance_ap=10e3,  # Anti-parallel resistance (Ω)
    resistance_p=5e3,   # Parallel resistance (Ω)
    thermal_stability=60  # E_b/k_B T
)

# Create environment with custom device
env = gym.make(
    'SpinTorque-v0',
    device=custom_device,
    temperature=300,  # Kelvin
    include_thermal_fluctuations=True
)
```

## Architecture

```
spin-torque-rl-gym/
├── spin_torque_gym/
│   ├── envs/
│   │   ├── spin_torque_env.py    # Main Gymnasium environment
│   │   ├── multi_device_env.py   # Multi-device control
│   │   ├── array_env.py          # Crossbar array control
│   │   └── skyrmion_env.py       # Skyrmion manipulation
│   ├── physics/
│   │   ├── llgs_solver.py        # LLG-Slonczewski solver
│   │   ├── thermal_model.py      # Thermal fluctuations
│   │   ├── materials.py          # Material parameters
│   │   └── quantum_effects.py    # Quantum corrections
│   ├── devices/
│   │   ├── stt_mram.py          # STT-MRAM device
│   │   ├── sot_mram.py          # SOT-MRAM device
│   │   ├── vcma_mram.py         # VCMA-MRAM device
│   │   └── skyrmion.py          # Skyrmion racetrack
│   ├── rewards/
│   │   ├── energy_reward.py      # Energy efficiency
│   │   ├── speed_reward.py       # Switching speed
│   │   ├── reliability_reward.py # Success rate
│   │   └── composite_reward.py   # Multi-objective
│   ├── visualization/
│   │   ├── magnetization_viz.py  # 3D magnetization
│   │   ├── phase_diagram.py     # Switching diagrams
│   │   └── trajectory_plot.py   # State trajectories
│   └── benchmarks/
│       ├── protocols.py          # Standard protocols
│       ├── baselines.py          # Baseline controllers
│       └── metrics.py            # Evaluation metrics
├── examples/
├── configs/
└── experiments/
```

## Physics Simulation

### Landau-Lifshitz-Gilbert-Slonczewski Dynamics

```python
from spin_torque_gym.physics import LLGSSolver

class SpinDynamics:
    """Simulates magnetization dynamics under spin-torque"""
    
    def __init__(self, device):
        self.device = device
        self.solver = LLGSSolver(
            method='rk45',
            atol=1e-9,
            rtol=1e-6
        )
    
    def step(self, current, dt):
        """Evolve magnetization for time dt under applied current"""
        # Effective field components
        h_eff = self.compute_effective_field()
        
        # Spin torque terms
        tau_stt = self.compute_stt_torque(current)
        tau_flt = self.compute_fieldlike_torque(current)
        
        # LLG-S equation
        dm_dt = -self.device.gamma * cross(self.m, h_eff)
        dm_dt += self.device.alpha * cross(self.m, dm_dt)
        dm_dt += tau_stt + tau_flt
        
        # Thermal fluctuations
        if self.include_thermal:
            dm_dt += self.thermal_field()
        
        # Integrate
        self.m = self.solver.step(self.m, dm_dt, dt)
        
        # Normalize
        self.m = self.m / norm(self.m)
        
        return self.m
```

### Energy Landscape

```python
from spin_torque_gym.physics import EnergyLandscape

# Analyze device energy landscape
landscape = EnergyLandscape(device)

# Find stable states
stable_states = landscape.find_minima()
print(f"Found {len(stable_states)} stable states")

# Compute minimum energy paths
mep = landscape.minimum_energy_path(
    initial_state=stable_states[0],
    final_state=stable_states[1],
    method='nudged_elastic_band'
)

# Energy barriers
barrier = landscape.energy_barrier(stable_states[0], stable_states[1])
print(f"Energy barrier: {barrier / device.kb / device.T:.1f} k_B T")

# Switching phase diagram
landscape.plot_phase_diagram(
    current_range=(-1e6, 1e6),  # A/cm²
    field_range=(-100, 100),     # mT
    save_path='phase_diagram.png'
)
```

## RL Environment Details

### Observation Space

```python
from spin_torque_gym.envs import SpinTorqueEnv

env = SpinTorqueEnv()

# Observation components
observation = {
    'magnetization': [mx, my, mz],        # Current magnetization
    'target': [tx, ty, tz],               # Target magnetization
    'resistance': R,                      # Current resistance
    'temperature': T,                     # Device temperature
    'time_remaining': t_remain,           # Steps remaining
    'energy_used': E_total,               # Cumulative energy
    'last_action': [I_prev, duration]     # Previous current pulse
}

print(f"Observation space: {env.observation_space}")
# Box(low=-1, high=1, shape=(11,), dtype=float32)
```

### Action Space

```python
# Continuous action space
action = {
    'current': I,      # Current magnitude (-I_max to I_max)
    'duration': dt,    # Pulse duration (0 to dt_max)
}

# Or discrete action space
env_discrete = gym.make(
    'SpinTorque-v0',
    action_mode='discrete',
    current_levels=[-2e6, -1e6, 0, 1e6, 2e6],  # A/cm²
    duration_levels=[0.1, 0.5, 1.0, 2.0]        # ns
)

print(f"Action space: {env_discrete.action_space}")
# Discrete(20)  # 5 currents × 4 durations
```

### Reward Function

```python
from spin_torque_gym.rewards import CompositeReward

# Multi-objective reward
reward = CompositeReward(
    components={
        'success': {
            'weight': 10.0,
            'function': lambda s, t: 10.0 if dot(s, t) > 0.9 else 0.0
        },
        'energy': {
            'weight': -0.1,
            'function': lambda E: -E / 1e-12  # Normalize to pJ
        },
        'speed': {
            'weight': 1.0,
            'function': lambda t: 1.0 / (1.0 + t / 1e-9)  # ns scale
        },
        'overshoot': {
            'weight': -5.0,
            'function': lambda m: -max(0, norm(m) - 1.1)
        }
    }
)

env = gym.make('SpinTorque-v0', reward_function=reward)
```

## Advanced Environments

### Multi-Device Array Control

```python
from spin_torque_gym.envs import SpinTorqueArrayEnv

# Control array of devices
array_env = SpinTorqueArrayEnv(
    array_size=(8, 8),
    device_type='stt_mram',
    coupling='dipolar',  # Include dipolar interactions
    target_pattern=checkerboard_pattern
)

# Observation includes all devices
obs = array_env.reset()
print(f"Array observation shape: {obs['magnetization'].shape}")
# (8, 8, 3)  # 8×8 devices, each with 3D magnetization

# Action controls row/column or individual devices
action = {
    'mode': 'row',     # 'row', 'column', 'individual'
    'index': 3,        # Row 3
    'current': 1.5e6,  # A/cm²
    'duration': 0.5    # ns
}
```

### Skyrmion Manipulation

```python
from spin_torque_gym.envs import SkyrmionRacetrackEnv

# Skyrmion racetrack memory
skyrmion_env = SkyrmionRacetrackEnv(
    track_length=1000e-9,  # 1 μm
    track_width=200e-9,    # 200 nm
    skyrmion_radius=20e-9, # 20 nm
    target_positions=[100e-9, 300e-9, 500e-9]
)

# Actions control current gradients
action = {
    'current_x': 1e6,    # Drive current along track
    'current_y': 0,      # Transverse current
    'gradient': 1e12     # Current gradient (A/cm³)
}

# Reward for precise positioning
obs, reward, done, _, info = skyrmion_env.step(action)
if done and info['position_error'] < 5e-9:
    print("Skyrmion positioned successfully!")
```

## Training Strategies

### Curriculum Learning

```python
from spin_torque_gym.utils import CurriculumWrapper

# Start with easy switching, gradually increase difficulty
curriculum_env = CurriculumWrapper(
    env,
    difficulty_schedule={
        0: {'thermal_stability': 30, 'damping': 0.02},
        100000: {'thermal_stability': 60, 'damping': 0.01},
        500000: {'thermal_stability': 100, 'damping': 0.005}
    }
)

# Train with curriculum
model = PPO('MlpPolicy', curriculum_env)
model.learn(total_timesteps=1_000_000)
```

### Sim-to-Real Transfer

```python
from spin_torque_gym.utils import DomainRandomization

# Add domain randomization for sim-to-real
randomized_env = DomainRandomization(
    env,
    parameters={
        'ms': (700e3, 900e3),        # ±12.5% variation
        'damping': (0.008, 0.012),   # ±20% variation
        'polarization': (0.6, 0.8),  # ±14% variation
        'temperature': (250, 350)     # ±50K variation
    }
)

# Train robust policy
robust_model = SAC('MlpPolicy', randomized_env)
robust_model.learn(total_timesteps=2_000_000)
```

## Visualization

### Real-Time Rendering

```python
from spin_torque_gym.visualization import MagnetizationRenderer

# 3D visualization during training
env = gym.make('SpinTorque-v0', render_mode='human')

# Or create custom renderer
renderer = MagnetizationRenderer(
    style='sphere',  # 'arrow', 'sphere', 'color'
    trail_length=50,
    update_rate=30   # fps
)

# Render training progress
for episode in range(100):
    obs, _ = env.reset()
    
    while True:
        action = agent.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        
        # Update visualization
        renderer.update(
            magnetization=obs['magnetization'],
            target=obs['target'],
            current=action['current']
        )
        
        if done:
            break
```

### Analysis Plots

```python
from spin_torque_gym.visualization import TrainingAnalyzer

analyzer = TrainingAnalyzer()

# Plot learning curves
analyzer.plot_learning_curves(
    model,
    metrics=['success_rate', 'energy_per_switch', 'switching_time'],
    save_path='learning_curves.png'
)

# Analyze learned strategies
analyzer.analyze_policy(
    model,
    env,
    num_episodes=100,
    plots=['action_distribution', 'phase_space', 'energy_histogram']
)

# Compare with optimal control
optimal_controller = load_optimal_controller()
analyzer.compare_policies(
    {'RL': model, 'Optimal': optimal_controller},
    test_cases=standard_test_cases,
    save_path='policy_comparison.png'
)
```

## Experimental Validation

### Hardware Protocol Matching

```python
from spin_torque_gym.benchmarks import ExperimentalProtocols

# Load experimental switching protocols
protocols = ExperimentalProtocols()

# Test against published results
test_results = {}
for protocol_name, protocol in protocols.items():
    env.set_device_parameters(protocol['device_params'])
    
    success_rate = evaluate_policy(
        model,
        env,
        protocol['test_conditions'],
        num_trials=1000
    )
    
    test_results[protocol_name] = {
        'success_rate': success_rate,
        'expected': protocol['expected_success_rate'],
        'deviation': abs(success_rate - protocol['expected_success_rate'])
    }

print(f"Average deviation from experiments: {np.mean([r['deviation'] for r in test_results.values()]):.2%}")
```

### Energy Optimization Results

```python
from spin_torque_gym.benchmarks import EnergyBenchmark

benchmark = EnergyBenchmark()

# Compare with state-of-the-art
results = benchmark.evaluate(
    model,
    metrics={
        'write_energy': 'pJ',
        'write_latency': 'ns',
        'ber': 'log10'  # Bit error rate
    }
)

# Generate comparison table
benchmark.generate_comparison_table(
    results,
    baselines=['conventional_pulse', 'precessional_switching', 'vcma_assisted'],
    save_path='energy_comparison.csv'
)
```

## Custom RL Algorithms

### Physics-Informed RL

```python
from spin_torque_gym.algorithms import PhysicsInformedPPO

# PPO with physics knowledge
pi_ppo = PhysicsInformedPPO(
    policy='MlpPolicy',
    env=env,
    physics_loss_weight=0.1,
    physics_constraints={
        'energy_conservation': True,
        'angular_momentum': True,
        'thermal_limit': 400  # K
    }
)

# Custom loss includes physics violations
def physics_loss(trajectory):
    violations = 0
    for state, action, next_state in trajectory:
        # Check physical constraints
        if violates_energy_conservation(state, action, next_state):
            violations += 1
        if exceeds_thermal_limit(state, action):
            violations += 1
    return violations / len(trajectory)

pi_ppo.learn(total_timesteps=1_000_000)
```

### Hierarchical Control

```python
from spin_torque_gym.algorithms import HierarchicalController

# High-level planner + low-level controller
hier_controller = HierarchicalController(
    high_level_policy='DQN',
    low_level_policy='TD3',
    env=env
)

# High-level decides switching strategy
# Low-level executes precise control

@hier_controller.high_level_action
def switching_strategy(state):
    """Decide switching approach based on state"""
    if is_easy_switch(state):
        return 'direct'
    elif is_hard_switch(state):
        return 'precessional'
    else:
        return 'multi_pulse'

@hier_controller.low_level_action
def execute_strategy(strategy, state):
    """Execute the chosen strategy"""
    if strategy == 'direct':
        return direct_switch_action(state)
    elif strategy == 'precessional':
        return precessional_switch_action(state)
    else:
        return multi_pulse_action(state)

hier_controller.learn(total_timesteps=2_000_000)
```

## Extensions

### Quantum Effects

```python
from spin_torque_gym.physics import QuantumSpinTorque

# Include quantum corrections
quantum_env = gym.make(
    'SpinTorque-v0',
    physics_model=QuantumSpinTorque(
        include_berry_phase=True,
        include_spin_pumping=True,
        quantum_temperature=10  # K
    )
)

# Quantum tunneling of magnetization
quantum_env.set_tunneling_parameters(
    attempt_frequency=1e9,  # Hz
    barrier_shape='symmetric'
)
```

### Multi-Physics Coupling

```python
from spin_torque_gym.physics import CoupledPhysics

# Couple magnetic, thermal, and mechanical
coupled_env = gym.make(
    'SpinTorque-v0',
    physics_model=CoupledPhysics(
        magnetic_model='llgs',
        thermal_model='fourier',
        mechanical_model='elastic',
        coupling_strength={
            'magnetoelastic': 0.1,
            'magnetothermal': 0.05
        }
    )
)
```

## Performance Optimization

### JAX Acceleration

```python
from spin_torque_gym.jax import JaxSpinTorqueEnv
import jax

# JAX-accelerated environment
jax_env = JaxSpinTorqueEnv(
    device_type='stt_mram',
    batch_size=1024  # Simulate 1024 environments in parallel
)

# Vectorized training
@jax.jit
def batch_step(states, actions):
    return jax.vmap(jax_env.step)(states, actions)

# 100x speedup for large batches
states = jax_env.reset(1024)
for _ in range(100):
    actions = policy(states)
    states, rewards, dones, infos = batch_step(states, actions)
```

## Configuration

### Environment Configuration

```yaml
# config/env_config.yaml
device:
  type: "stt_mram"
  geometry:
    shape: "ellipse"
    major_axis: 100e-9
    minor_axis: 50e-9
    thickness: 2e-9
  material:
    name: "CoFeB"
    ms: 800e3
    exchange: 20e-12
    anisotropy: 1e6
    
physics:
  temperature: 300
  include_thermal: true
  include_dipolar: false
  timestep: 1e-12
  
training:
  max_current: 2e6  # A/cm²
  max_duration: 5e-9  # 5 ns
  success_threshold: 0.95
  energy_weight: 0.1
```

## Troubleshooting

### Common Issues

```python
from spin_torque_gym.diagnostics import EnvironmentDiagnostics

diag = EnvironmentDiagnostics(env)

# Check environment stability
stability = diag.check_stability(num_steps=10000)
if not stability.is_stable:
    print(f"Unstable dynamics detected: {stability.issue}")
    # Reduce timestep or adjust parameters
    env.physics.timestep *= 0.1

# Analyze reward distribution
reward_stats = diag.analyze_rewards(num_episodes=100)
if reward_stats.variance > 100:
    print("High reward variance - consider reward normalization")
    
# Debug action effects
diag.visualize_action_effects(
    state=env.reset()[0],
    action_samples=100,
    save_path='action_effects.png'
)
```

## Citation

```bibtex
@software{spin_torque_rl_gym,
  title={Spin-Torque RL-Gym: Reinforcement Learning for Spintronic Device Control},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/spin-torque-rl-gym}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Spintronics research community for device physics
- Gymnasium team for the RL framework
- Experimental collaborators for validation data

## Resources

- [Documentation](https://spin-torque-rl-gym.readthedocs.io)
- [Tutorials](https://github.com/yourusername/spin-torque-rl-gym/tutorials)
- [Device Database](https://spin-torque-rl-gym.github.io/devices)
- [Benchmark Results](https://spin-torque-rl-gym.github.io/benchmarks)
