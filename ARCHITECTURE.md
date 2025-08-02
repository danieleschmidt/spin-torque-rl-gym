# Spin-Torque RL-Gym Architecture

## System Overview

Spin-Torque RL-Gym provides a physically-accurate Gymnasium environment for training reinforcement learning agents to control spintronic devices. The system bridges quantum mechanical spin dynamics with modern RL frameworks to enable AI-driven discovery of optimal magnetization switching protocols.

## Architecture Principles

### 1. Physics-First Design
- All simulations based on fundamental Landau-Lifshitz-Gilbert-Slonczewski equations
- Quantum mechanical effects (thermal fluctuations, Berry phase) included where relevant
- Material parameters calibrated against experimental data

### 2. Modular Component Architecture
- Cleanly separated physics simulation, RL environment, and visualization layers
- Pluggable device models (STT-MRAM, SOT-MRAM, VCMA, skyrmions)
- Configurable reward functions for multi-objective optimization

### 3. Performance Optimization
- JAX acceleration for batch training (100x speedup)
- Efficient numerical solvers with adaptive timestep
- Memory-efficient state representation

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Training Scripts  │  Benchmarks  │  Analysis Tools  │  Examples │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      RL Environment Layer                       │
├─────────────────────────────────────────────────────────────────┤
│          Gymnasium Interface          │    Visualization        │
│  ┌─────────────────────────────────┐   │  ┌───────────────────┐  │
│  │  SpinTorqueEnv                  │   │  │ MagnetizationViz  │  │
│  │ ┌─────────────┬─────────────────┤   │  │ PhaseSpace Plot   │  │
│  │ │Obs Space    │ Action Space    │   │  │ Energy Landscape  │  │
│  │ │- Magnetizat.│ - Current       │   │  │ Training Progress │  │
│  │ │- Target     │ - Duration      │   │  └───────────────────┘  │
│  │ │- Energy     │ - Field         │   │                        │
│  │ └─────────────┴─────────────────┤   │                        │
│  └─────────────────────────────────┘   │                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        Physics Engine                           │
├─────────────────────────────────────────────────────────────────┤
│     Dynamics Solver     │    Device Models    │   Reward Func   │
│  ┌─────────────────────┐│  ┌─────────────────┐│ ┌──────────────┐ │
│  │ LLGS Integrator     ││  │ STT-MRAM        ││ │ Energy       │ │
│  │ - RK45/DOP853       ││  │ SOT-MRAM        ││ │ Speed        │ │
│  │ - Adaptive timestep ││  │ VCMA-MRAM       ││ │ Reliability  │ │
│  │ - Error control     ││  │ Skyrmion Track  ││ │ Multi-obj    │ │
│  └─────────────────────┘│  └─────────────────┘│ └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      Foundation Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Material Database  │  Numerical Methods  │    Utilities        │
│  ┌─────────────────│  ┌─────────────────────│  ┌──────────────── │
│  │ CoFeB/MgO       │  │ Vector Operations   │  │ Config Parser   │
│  │ Pt/Co           │  │ Energy Minimization │  │ Data Logging    │
│  │ W/CoFeB         │  │ Thermal Noise       │  │ Metrics         │
│  │ Synthetic AFM   │  │ Random Fields       │  │ Checkpointing   │
│  └─────────────────│  └─────────────────────│  └──────────────── │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

### Training Loop Data Flow

```
Environment Reset
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Initial State Generation                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Random magnetization + target pattern + device parameters   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RL Agent                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Policy Network: obs → (current, duration, field)            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Physics Simulation                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. Apply control pulse (current, field)                     │ │
│  │ 2. Integrate LLGS equations for duration                    │ │
│  │ 3. Add thermal fluctuations                                 │ │
│  │ 4. Update magnetization state                               │ │
│  │ 5. Calculate energy consumed                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Reward Calculation                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Success: dot(m_final, m_target) > threshold                 │ │
│  │ Energy: -energy_consumed / normalization                    │ │
│  │ Speed: 1.0 / (switching_time + epsilon)                    │ │
│  │ Stability: -|magnetization_overshoot|                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    State Transition                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Update observation with new magnetization                   │ │
│  │ Check termination conditions                                │ │
│  │ Return (obs, reward, done, truncated, info)                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Physics Engine (`spin_torque_gym/physics/`)

**Landau-Lifshitz-Gilbert-Slonczewski Solver**
- Integrates magnetization dynamics under applied currents and fields
- Supports multiple numerical methods (RK45, DOP853, Euler)
- Adaptive timestep with error control for stability

**Thermal Fluctuation Model**
- Gaussian white noise representing thermal effects
- Temperature-dependent noise amplitude: σ = √(2αkT/γμ₀MsV)
- Correlated noise across magnetization components

**Material Parameter Database**
- Experimentally validated parameters for common materials
- Temperature and field dependence of material constants
- Interface effects for multilayer structures

### 2. Device Models (`spin_torque_gym/devices/`)

**STT-MRAM Device**
```python
class STTMRAMDevice:
    """Spin-transfer torque MRAM with in-plane or perpendicular anisotropy"""
    
    def __init__(self, geometry, material, thermal_stability):
        self.volume = geometry.calculate_volume()
        self.demagnetization = geometry.calculate_demag_factors()
        self.spin_torque_efficiency = material.polarization
        self.thermal_barrier = thermal_stability * k_B * temperature
    
    def compute_effective_field(self, magnetization, applied_field):
        """Total effective field including anisotropy, exchange, demag"""
        h_anis = self.anisotropy_field(magnetization)
        h_demag = self.demagnetization_field(magnetization)
        h_thermal = self.thermal_field()
        return applied_field + h_anis + h_demag + h_thermal
    
    def compute_spin_torque(self, current, magnetization):
        """Slonczewski spin-transfer torque terms"""
        p_hat = self.polarization_direction
        tau_stt = (self.beta * current / self.volume) * cross(magnetization, cross(magnetization, p_hat))
        tau_fl = (self.beta_prime * current / self.volume) * cross(magnetization, p_hat)
        return tau_stt + tau_fl
```

### 3. RL Environment Interface (`spin_torque_gym/envs/`)

**Observation Space Design**
```python
observation_space = gym.spaces.Box(
    low=-1.0, high=1.0, 
    shape=(11,),  # [mx, my, mz, tx, ty, tz, R/R0, T/T0, t_rem, E_norm, I_prev]
    dtype=np.float32
)
```

**Action Space Options**
- Continuous: Current magnitude and duration
- Discrete: Predefined current levels and pulse widths
- Multi-discrete: Separate discretization for each action dimension

**Reward Function Architecture**
```python
class CompositeReward:
    def __init__(self, components: Dict[str, Dict]):
        self.components = components
    
    def compute(self, state, action, next_state, info):
        total_reward = 0.0
        for name, config in self.components.items():
            component_reward = config['function'](state, action, next_state, info)
            total_reward += config['weight'] * component_reward
        return total_reward
```

### 4. Visualization System (`spin_torque_gym/visualization/`)

**Real-time Magnetization Rendering**
- 3D arrow plots showing magnetization vectors
- Phase space trajectories on Bloch sphere
- Energy landscape visualization with switching paths

**Training Progress Monitoring**
- Success rate vs. episode plots
- Energy efficiency improvements over time
- Action distribution analysis

## Performance Considerations

### Memory Management
- Efficient state representation using float32
- Lazy loading of material parameters
- Configurable trajectory history length

### Computational Optimization
- JAX just-in-time compilation for physics kernels
- Vectorized operations for batch training
- GPU acceleration for large device arrays

### Numerical Stability
- Automatic timestep adaptation based on magnetization dynamics
- Normalized magnetization vectors maintained at unit length
- Energy conservation monitoring for solver validation

## Extensibility Points

### Adding New Device Types
1. Implement device class inheriting from `BaseSpintronicDevice`
2. Define geometry, material parameters, and physics methods
3. Register device in device factory
4. Add corresponding test cases

### Custom Reward Functions
1. Inherit from `BaseReward` class
2. Implement `compute()` method with desired objectives
3. Register in reward function registry
4. Validate with known optimal solutions

### Advanced Physics Models
1. Add new solver to `physics/solvers/` directory
2. Implement required interface methods
3. Add configuration options
4. Benchmark against existing solvers

## Quality Assurance

### Testing Strategy
- Unit tests for all physics calculations
- Integration tests for complete training loops
- Validation against experimental switching data
- Performance benchmarks for different configurations

### Code Quality
- Type hints throughout codebase
- Comprehensive docstrings with physics equations
- Linting with ruff and mypy
- Pre-commit hooks for code formatting

This architecture ensures maintainable, extensible, and physically accurate simulation of spintronic devices for RL training while providing excellent performance for large-scale experiments.