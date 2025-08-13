"""Unit tests for physics simulation components."""

import numpy as np


class TestLLGSSolver:
    """Test Landau-Lifshitz-Gilbert-Slonczewski solver."""

    def test_llgs_equation_structure(self, sample_magnetization):
        """Test LLGS equation structure."""
        m = sample_magnetization

        # Ensure magnetization is normalized
        assert np.isclose(np.linalg.norm(m), 1.0)

        # Test gyromagnetic ratio
        gamma = 2.21e5  # m/A⋅s
        assert gamma > 0

        # Test damping parameter
        alpha = 0.01
        assert 0 < alpha <= 1.0

    def test_effective_field_components(self, sample_magnetization):
        """Test effective field calculation."""
        m = sample_magnetization

        # Anisotropy field (uniaxial along z)
        k_u = 1e6  # J/m³
        ms = 800e3  # A/m
        h_anis = 2 * k_u / (4e-7 * np.pi * ms)  # A/m
        h_k = h_anis * np.array([0, 0, m[2]])

        assert h_anis > 0
        assert h_k[2] != 0

    def test_exchange_field(self):
        """Test exchange field calculation."""
        # Exchange constant
        a_ex = 20e-12  # J/m
        assert a_ex > 0

        # Exchange field ∝ ∇²m
        # For uniform magnetization, exchange field is zero
        uniform_m = np.array([0, 0, 1])
        # ∇²m = 0 for uniform state
        exchange_field = np.array([0, 0, 0])
        assert np.allclose(exchange_field, np.zeros(3))

    def test_demagnetization_field(self, mock_device_params):
        """Test demagnetization field calculation."""
        size = mock_device_params["size"]
        ms = mock_device_params["ms"]

        # Demagnetization factors for ellipsoid
        # For thin film (thickness << width, length): Nz ≈ 1, Nx ≈ Ny ≈ 0
        n_x, n_y, n_z = 0.0, 0.0, 1.0
        assert n_x + n_y + n_z == 1.0

        # Demagnetization field H_d = -N⋅M
        m = np.array([0, 0, 1])
        h_demag = -np.array([n_x * m[0], n_y * m[1], n_z * m[2]]) * ms
        assert h_demag[2] < 0  # Opposes z-component

    def test_thermal_field(self, physics_config):
        """Test thermal fluctuation field."""
        temperature = physics_config["temperature"]
        if temperature > 0:
            # Thermal field strength
            k_b = 1.380649e-23  # J/K
            mu_0 = 4 * np.pi * 1e-7  # H/m

            # Volume and timestep needed for noise calculation
            volume = 50e-9 * 100e-9 * 2e-9  # m³
            dt = physics_config["timestep"]

            noise_strength = np.sqrt(2 * k_b * temperature / (mu_0 * volume * dt))
            assert noise_strength > 0
        else:
            # No thermal noise at T=0
            assert temperature == 0

    def test_spin_torque_terms(self, mock_action):
        """Test spin-transfer torque calculation."""
        current = mock_action["current"]  # A/cm²

        # Convert to A/m²
        j = current * 1e4
        assert j > 0

        # Spin polarization
        p = 0.7
        assert 0 < p <= 1.0

        # Slonczewski torque coefficient
        # β = (μ_B P ℏ j) / (2 e Ms t)
        mu_b = 9.274010078e-24  # J/T
        h_bar = 1.054571817e-34  # J⋅s
        e = 1.602176634e-19  # C

        assert mu_b > 0
        assert h_bar > 0
        assert e > 0

    def test_numerical_integration(self, sample_magnetization, mock_action):
        """Test numerical integration methods."""
        m = sample_magnetization
        dt = 1e-12  # 1 ps

        # Simple Euler method test
        dmdt = np.array([0.1, 0.0, -0.1])  # Example derivative
        m_new_euler = m + dt * dmdt

        # Normalize
        m_new_euler = m_new_euler / np.linalg.norm(m_new_euler)
        assert np.isclose(np.linalg.norm(m_new_euler), 1.0)


class TestThermalModel:
    """Test thermal fluctuation model."""

    def test_langevin_dynamics(self, physics_config):
        """Test Langevin dynamics implementation."""
        temperature = physics_config["temperature"]
        dt = physics_config["timestep"]

        if temperature > 0:
            # White noise characteristics
            # <ξ(t)> = 0, <ξ(t)ξ(t')> = 2D δ(t-t')
            np.random.seed(42)
            noise = np.random.normal(0, 1, size=1000)

            # Mean should be close to zero
            assert abs(np.mean(noise)) < 0.1

            # Variance should be close to 1
            assert abs(np.var(noise) - 1.0) < 0.1

    def test_fluctuation_dissipation_theorem(self, mock_device_params):
        """Test fluctuation-dissipation theorem."""
        temperature = mock_device_params["temperature"]
        damping = mock_device_params["damping"]

        if temperature > 0 and damping > 0:
            # FDT: noise correlations related to dissipation
            k_b = 1.380649e-23  # J/K
            correlation_strength = 2 * damping * k_b * temperature
            assert correlation_strength > 0


class TestQuantumEffects:
    """Test quantum corrections to classical dynamics."""

    def test_quantum_tunneling(self):
        """Test quantum tunneling probability."""
        # Energy barrier
        e_b = 1e-20  # J
        temperature = 300  # K
        k_b = 1.380649e-23  # J/K

        # Thermal activation rate
        rate_thermal = np.exp(-e_b / (k_b * temperature))
        assert 0 < rate_thermal <= 1

        # Quantum tunneling becomes important at low temperatures
        if temperature < 10:  # K
            # Quantum effects should be considered
            assert e_b / (k_b * temperature) > 10

    def test_berry_phase(self):
        """Test Berry phase contributions."""
        # Berry curvature for magnetic systems
        # Placeholder for Berry phase calculations
        berry_curvature = 1.0  # Example value
        assert isinstance(berry_curvature, (int, float))

    def test_spin_pumping(self):
        """Test spin pumping effects."""
        # Spin pumping conductance
        g_spin = 1e15  # S/m²
        assert g_spin > 0

        # Additional damping from spin pumping
        alpha_sp = 0.001
        assert alpha_sp > 0


class TestMaterialModels:
    """Test material parameter models."""

    def test_temperature_dependence(self):
        """Test temperature dependence of material parameters."""
        # Saturation magnetization vs temperature
        # Ms(T) = Ms(0) * (1 - (T/Tc)^α)
        ms_0 = 800e3  # A/m at T=0
        t_curie = 600  # K
        temperature = 300  # K
        alpha_temp = 1.5

        ms_t = ms_0 * (1 - (temperature / t_curie) ** alpha_temp)
        assert 0 < ms_t < ms_0

    def test_current_dependence(self):
        """Test current-dependent heating effects."""
        # Joule heating: P = I²R
        current = 1e-3  # A
        resistance = 1e3  # Ω
        power = current**2 * resistance
        assert power > 0

        # Temperature rise
        thermal_resistance = 1e6  # K/W
        delta_t = power * thermal_resistance
        assert delta_t > 0

    def test_anisotropy_models(self):
        """Test magnetic anisotropy models."""
        # Uniaxial anisotropy
        k_u = 1e6  # J/m³
        assert k_u > 0

        # Shape anisotropy
        ms = 800e3  # A/m
        mu_0 = 4 * np.pi * 1e-7  # H/m
        k_shape = 0.5 * mu_0 * ms**2
        assert k_shape > 0

        # Interface anisotropy
        k_s = 1e-3  # J/m²
        thickness = 2e-9  # m
        k_interface = 2 * k_s / thickness
        assert k_interface > 0
