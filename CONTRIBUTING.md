# Contributing to Spin-Torque RL-Gym

Thank you for your interest in contributing to Spin-Torque RL-Gym! This project aims to advance reinforcement learning research in spintronics and neuromorphic computing.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/spin-torque-rl-gym.git
   cd spin-torque-rl-gym
   ```

2. **Development Setup**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

## 📋 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Add comprehensive tests for new functionality
- Update documentation as needed
- Ensure physics simulations remain accurate

### 3. Quality Checks
```bash
# Linting and formatting
ruff check .
ruff format .

# Type checking
mypy spin_torque_gym

# Security scanning
bandit -r spin_torque_gym
safety check

# Test coverage
pytest --cov=spin_torque_gym --cov-report=html
```

### 4. Submit Pull Request
- Write clear, descriptive commit messages
- Include tests and documentation
- Reference related issues
- Ensure all CI checks pass

## 🎯 Contribution Areas

### Core Physics
- **Magnetic Dynamics**: Improve LLG-S solvers
- **Thermal Effects**: Better stochastic models
- **Quantum Corrections**: Add quantum tunneling
- **Multi-Physics**: Couple electrical, thermal, mechanical

### RL Environments
- **New Device Types**: VCMA, SOT, domain walls
- **Array Environments**: Crossbar arrays, memristor networks
- **Custom Rewards**: Energy-aware, reliability-based
- **Observation Spaces**: Add noise, partial observability

### Algorithms
- **Physics-Informed RL**: Constraint-aware training
- **Hierarchical Control**: Multi-level optimization
- **Transfer Learning**: Sim-to-real adaptation
- **Multi-Objective**: Pareto-optimal solutions

### Validation
- **Experimental Data**: Device characterization
- **Benchmark Protocols**: Standard test cases
- **Uncertainty Quantification**: Model confidence
- **Hardware Validation**: Real device comparison

## 🔬 Physics Standards

### Accuracy Requirements
- **Energy Conservation**: Within 1% per simulation
- **Angular Momentum**: Properly conserved
- **Thermal Equilibrium**: Detailed balance maintained
- **Unit Consistency**: SI units throughout

### Reference Materials
- [LLG-S Equations](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.54.9353)
- [Thermal Fluctuations](https://doi.org/10.1063/1.1968418)
- [Spin Transfer Torque](https://doi.org/10.1063/1.1819531)

## 🧪 Testing Guidelines

### Test Categories
```python
# Unit tests - individual components
tests/unit/test_physics_solver.py
tests/unit/test_device_models.py

# Integration tests - component interactions  
tests/integration/test_environment.py
tests/integration/test_training.py

# Physics tests - validation against theory
tests/physics/test_energy_conservation.py
tests/physics/test_thermal_equilibrium.py

# Benchmark tests - performance validation
tests/benchmarks/test_switching_protocols.py
tests/benchmarks/test_energy_efficiency.py
```

### Test Standards
- **Coverage**: Minimum 85% for new code
- **Physics**: Validate against analytical solutions
- **Performance**: No >5% regression in speed
- **Reproducibility**: Fixed random seeds

## 📚 Documentation

### Code Documentation
```python
def compute_effective_field(self, magnetization: np.ndarray) -> np.ndarray:
    """Compute effective magnetic field.
    
    Includes exchange, anisotropy, demagnetization, and applied fields.
    
    Args:
        magnetization: Unit magnetization vector [mx, my, mz]
        
    Returns:
        Effective field in Tesla [Hx, Hy, Hz]
        
    Raises:
        ValueError: If magnetization is not unit vector
        
    References:
        Aharoni, A. (2000). Introduction to micromagnetics
    """
```

### Examples and Tutorials
- Jupyter notebooks with clear explanations
- Progressive difficulty from basic to advanced
- Real-world use cases and applications
- Performance optimization tips

## 🔒 Security Considerations

- **No Secrets**: Never commit API keys or credentials
- **Input Validation**: Sanitize all user inputs
- **Dependency Scanning**: Regular security audits
- **Code Review**: Security-focused PR reviews

## 🐛 Bug Reports

### Issue Template
```markdown
**Environment Information**
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.5]
- Spin-Torque RL-Gym Version: [e.g., 0.1.0]

**Bug Description**
Clear description of the issue

**Minimal Reproduction**
```python
# Minimal code to reproduce the bug
```

**Expected Behavior**
What should happen

**Actual Behavior**  
What actually happens

**Physics Context**
If physics-related, include device parameters and expected physical behavior
```

## 🎖️ Recognition

Contributors are recognized in:
- [AUTHORS.md](AUTHORS.md) file
- Release notes for significant contributions
- Academic publications when appropriate
- Conference presentations and demos

## 📞 Getting Help

- **Discussions**: [GitHub Discussions](https://github.com/terragon-labs/spin-torque-rl-gym/discussions)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/terragon-labs/spin-torque-rl-gym/issues)
- **Email**: daniel@terragonlabs.com for sensitive issues

## 🏛️ Governance

This project follows a **Benevolent Dictator** model with community input:

- **Maintainer**: Daniel Schmidt (@dschmidt-terragon)
- **Core Contributors**: Physics and ML domain experts
- **Community**: All contributors and users

### Decision Process
1. **Discussion**: Community input via issues/discussions
2. **Proposal**: Detailed RFC for major changes
3. **Review**: Technical and physics validation
4. **Decision**: Maintainer final approval
5. **Implementation**: Community contribution

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for helping advance RL research in spintronics!** 🧲🤖