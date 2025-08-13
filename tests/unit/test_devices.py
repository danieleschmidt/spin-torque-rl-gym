"""Comprehensive unit tests for device models."""


import numpy as np
import pytest

from spin_torque_gym.devices import DeviceFactory
from spin_torque_gym.devices.skyrmion_device import SkyrmionDevice
from spin_torque_gym.devices.sot_mram import SOTMRAMDevice
from spin_torque_gym.devices.stt_mram import STTMRAMDevice
from spin_torque_gym.devices.vcma_mram import VCMAMRAMDevice


class TestSTTMRAMDevice:
    """Test STT-MRAM device model."""

    @pytest.fixture
    def device_params(self):
        """STT-MRAM device parameters."""
        return {
            'volume': 1e-24,  # m³
            'saturation_magnetization': 800e3,  # A/m
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,  # J/m³
            'polarization': 0.7,
            'easy_axis': np.array([0, 0, 1]),
            'reference_magnetization': np.array([0, 0, 1]),
            'resistance_parallel': 1e3,  # Ω
            'resistance_antiparallel': 2e3  # Ω
        }

    @pytest.fixture
    def stt_device(self, device_params):
        """Create STT-MRAM device instance."""
        return STTMRAMDevice(device_params)

    def test_device_initialization(self, stt_device, device_params):
        """Test device initialization with parameters."""
        assert stt_device.volume == device_params['volume']
        assert stt_device.saturation_magnetization == device_params['saturation_magnetization']
        assert np.allclose(stt_device.reference_magnetization, device_params['reference_magnetization'])

    def test_parameter_validation(self, device_params):
        """Test parameter validation."""
        # Test missing required parameter
        incomplete_params = device_params.copy()
        del incomplete_params['volume']

        with pytest.raises(ValueError, match="Missing required parameter: volume"):
            STTMRAMDevice(incomplete_params)

        # Test invalid parameter values
        invalid_params = device_params.copy()
        invalid_params['volume'] = -1e-24

        with pytest.raises(ValueError, match="Volume must be positive"):
            STTMRAMDevice(invalid_params)

        invalid_params['volume'] = 1e-24
        invalid_params['damping'] = 1.5

        with pytest.raises(ValueError, match="Damping must be between 0 and 1"):
            STTMRAMDevice(invalid_params)

    def test_effective_field_computation(self, stt_device):
        """Test effective magnetic field calculation."""
        magnetization = np.array([0, 0, 1])
        applied_field = np.array([100, 0, 0])  # A/m

        h_eff = stt_device.compute_effective_field(magnetization, applied_field)

        # Should include applied field and anisotropy contributions
        assert h_eff.shape == (3,)
        assert np.linalg.norm(h_eff) > 0

        # Applied field should be included
        assert np.allclose(h_eff[:1], applied_field[:1])

    def test_resistance_calculation(self, stt_device):
        """Test TMR resistance calculation."""
        # Parallel configuration
        m_parallel = np.array([0, 0, 1])
        r_parallel = stt_device.compute_resistance(m_parallel)
        assert np.isclose(r_parallel, 1e3, rtol=1e-2)

        # Anti-parallel configuration
        m_antiparallel = np.array([0, 0, -1])
        r_antiparallel = stt_device.compute_resistance(m_antiparallel)
        assert np.isclose(r_antiparallel, 2e3, rtol=1e-2)

        # Intermediate angle (90 degrees)
        m_perpendicular = np.array([1, 0, 0])
        r_perpendicular = stt_device.compute_resistance(m_perpendicular)
        assert 1e3 < r_perpendicular < 2e3

    def test_magnetization_validation(self, stt_device):
        """Test magnetization vector validation."""
        # Valid unit vector
        valid_mag = np.array([0, 0, 1])
        normalized = stt_device.validate_magnetization(valid_mag)
        assert np.allclose(np.linalg.norm(normalized), 1.0)

        # Non-unit vector (should be normalized)
        non_unit = np.array([0, 0, 2])
        normalized = stt_device.validate_magnetization(non_unit)
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        assert np.allclose(normalized, np.array([0, 0, 1]))

        # Zero vector (should raise error)
        with pytest.raises(ValueError):
            stt_device.validate_magnetization(np.array([0, 0, 0]))

    def test_string_representation(self, stt_device):
        """Test string representation."""
        repr_str = repr(stt_device)
        assert "STTMRAMDevice" in repr_str
        assert "volume" in repr_str
        assert "Ms" in repr_str


class TestSOTMRAMDevice:
    """Test SOT-MRAM device model."""

    @pytest.fixture
    def device_params(self):
        """SOT-MRAM device parameters."""
        return {
            'volume': 1e-24,  # m³
            'saturation_magnetization': 800e3,  # A/m
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,  # J/m³
            'easy_axis': np.array([0, 0, 1]),
            'spin_hall_angle': 0.1,
            'heavy_metal_thickness': 5e-9,  # m
            'heavy_metal_resistivity': 2e-7,  # Ω⋅m
            'interface_transparency': 0.5,
            'field_like_efficiency': 0.1,
            'damping_like_efficiency': 0.2
        }

    @pytest.fixture
    def sot_device(self, device_params):
        """Create SOT-MRAM device instance."""
        return SOTMRAMDevice(device_params)

    def test_device_initialization(self, sot_device, device_params):
        """Test device initialization."""
        assert sot_device.spin_hall_angle == device_params['spin_hall_angle']
        assert sot_device.heavy_metal_thickness == device_params['heavy_metal_thickness']
        assert hasattr(sot_device, 'j_s_efficiency')
        assert hasattr(sot_device, 'tau_dl_factor')
        assert hasattr(sot_device, 'tau_fl_factor')

    def test_parameter_validation(self, device_params):
        """Test SOT parameter validation."""
        # Test unrealistic spin Hall angle
        invalid_params = device_params.copy()
        invalid_params['spin_hall_angle'] = 1.5

        # Should create device but issue warning
        with pytest.warns(UserWarning, match="Spin Hall angle > 1.0 is physically unrealistic"):
            SOTMRAMDevice(invalid_params)

    def test_effective_field_computation(self, sot_device):
        """Test effective field including all contributions."""
        magnetization = np.array([0, 0, 1])
        applied_field = np.array([100, 0, 0])

        h_eff = sot_device.compute_effective_field(magnetization, applied_field)

        assert h_eff.shape == (3,)
        assert np.linalg.norm(h_eff) > 0

    def test_spin_torque_calculation(self, sot_device):
        """Test spin-orbit torque calculation."""
        current_density = 1e6  # A/m²
        magnetization = np.array([0, 1, 0])  # y-direction
        current_direction = np.array([1, 0, 0])  # x-direction

        tau_dl, tau_fl = sot_device.compute_spin_torque(current_density, magnetization, current_direction)

        # Check torque vectors are perpendicular to magnetization
        assert tau_dl.shape == (3,)
        assert tau_fl.shape == (3,)

        # Damping-like torque should be perpendicular to m
        assert np.abs(np.dot(tau_dl, magnetization)) < 1e-10

    def test_switching_threshold(self, sot_device):
        """Test critical current calculation."""
        thresholds = sot_device.get_switching_threshold()

        assert 'critical_current_density' in thresholds
        assert 'critical_field' in thresholds
        assert 'damping_like_efficiency' in thresholds
        assert 'field_like_efficiency' in thresholds

        assert thresholds['critical_current_density'] > 0
        assert thresholds['critical_field'] > 0

    def test_power_consumption(self, sot_device):
        """Test power consumption calculation."""
        current_density = 1e6  # A/m²
        pulse_duration = 1e-9  # s
        magnetization = np.array([0, 0, 1])

        energy = sot_device.compute_power_consumption(current_density, pulse_duration, magnetization)

        assert energy > 0
        assert isinstance(energy, float)

    def test_estimate_switching_time(self, sot_device):
        """Test switching time estimation."""
        # Get critical current to ensure we're above threshold
        j_c = sot_device.get_switching_threshold()['critical_current_density']
        current_density = 2 * j_c  # Use 2x critical current for fast switching
        temperature = 300.0  # K

        switching_time = sot_device.estimate_switching_time(current_density, temperature)

        assert switching_time > 0
        assert switching_time < 1e-6  # Should be reasonable for high current


class TestVCMAMRAMDevice:
    """Test VCMA-MRAM device model."""

    @pytest.fixture
    def device_params(self):
        """VCMA-MRAM device parameters."""
        return {
            'volume': 1e-24,  # m³
            'saturation_magnetization': 800e3,  # A/m
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,  # J/m³
            'easy_axis': np.array([0, 0, 1]),
            'vcma_coefficient': 100e-6,  # J/(V⋅m)
            'dielectric_thickness': 1e-9,  # m
            'dielectric_constant': 25.0,
            'breakdown_voltage': 2.0,  # V
            'leakage_resistance': 1e12  # Ω
        }

    @pytest.fixture
    def vcma_device(self, device_params):
        """Create VCMA-MRAM device instance."""
        return VCMAMRAMDevice(device_params)

    def test_device_initialization(self, vcma_device, device_params):
        """Test device initialization."""
        assert vcma_device.vcma_coefficient == device_params['vcma_coefficient']
        assert vcma_device.dielectric_thickness == device_params['dielectric_thickness']
        assert hasattr(vcma_device, 'capacitance')
        assert hasattr(vcma_device, 'base_anisotropy')

    def test_parameter_validation(self, device_params):
        """Test VCMA parameter validation."""
        # Test negative VCMA coefficient
        invalid_params = device_params.copy()
        invalid_params['vcma_coefficient'] = -100e-6

        with pytest.warns(UserWarning, match="Negative VCMA coefficient"):
            VCMAMRAMDevice(invalid_params)

    def test_effective_anisotropy_calculation(self, vcma_device):
        """Test voltage-modified anisotropy calculation."""
        # Zero voltage
        k_eff_0 = vcma_device._compute_effective_anisotropy(0.0)
        assert k_eff_0 == vcma_device.base_anisotropy

        # Positive voltage
        k_eff_pos = vcma_device._compute_effective_anisotropy(1.0)
        assert k_eff_pos != k_eff_0

        # Voltage clamping at breakdown
        k_eff_high = vcma_device._compute_effective_anisotropy(10.0)  # Above breakdown
        k_eff_breakdown = vcma_device._compute_effective_anisotropy(vcma_device.breakdown_voltage)
        assert k_eff_high == k_eff_breakdown

    def test_switching_probability(self, vcma_device):
        """Test switching probability calculation."""
        voltage = 1.5  # V
        pulse_duration = 1e-9  # s
        temperature = 300.0  # K

        prob = vcma_device.compute_switching_probability(voltage, pulse_duration, temperature)

        assert 0 <= prob <= 1

        # Higher voltage should give higher probability
        prob_high = vcma_device.compute_switching_probability(1.8, pulse_duration, temperature)
        assert prob_high >= prob

        # Zero voltage should give low probability
        prob_zero = vcma_device.compute_switching_probability(0.0, pulse_duration, temperature)
        assert prob_zero < prob

    def test_power_consumption(self, vcma_device):
        """Test VCMA power consumption."""
        voltage = 1.0  # V
        pulse_duration = 1e-9  # s

        energy = vcma_device.compute_power_consumption(voltage, pulse_duration)

        assert energy > 0

        # Higher voltage should consume more energy
        energy_high = vcma_device.compute_power_consumption(1.5, pulse_duration)
        assert energy_high > energy

        # Zero voltage should consume no energy
        energy_zero = vcma_device.compute_power_consumption(0.0, pulse_duration)
        assert energy_zero == 0.0

    def test_switching_threshold(self, vcma_device):
        """Test switching threshold calculation."""
        thresholds = vcma_device.get_switching_threshold()

        assert 'critical_voltage' in thresholds
        assert 'thermal_switching_voltage' in thresholds
        assert 'breakdown_voltage' in thresholds

        assert thresholds['critical_voltage'] > 0
        assert thresholds['thermal_switching_voltage'] >= 0
        assert thresholds['breakdown_voltage'] == vcma_device.breakdown_voltage

    def test_energy_barrier_calculation(self, vcma_device):
        """Test energy barrier calculation."""
        magnetization = np.array([0, 0, 1])  # Easy axis

        # Zero voltage
        barrier_0 = vcma_device.compute_energy_barrier(magnetization, 0.0)
        assert barrier_0 >= 0

        # With voltage
        barrier_v = vcma_device.compute_energy_barrier(magnetization, 1.0)
        assert barrier_v != barrier_0

    def test_estimate_switching_time(self, vcma_device):
        """Test switching time estimation."""
        voltage = 1.5  # V
        temperature = 300.0  # K

        switching_time = vcma_device.estimate_switching_time(voltage, temperature)

        assert switching_time > 0

        # Higher voltage should give faster switching
        switching_time_high = vcma_device.estimate_switching_time(1.8, temperature)
        assert switching_time_high <= switching_time

    def test_leakage_current(self, vcma_device):
        """Test leakage current calculation."""
        voltage = 1.0  # V

        current = vcma_device.compute_leakage_current(voltage)
        assert current > 0

        # Higher voltage should give higher current
        current_high = vcma_device.compute_leakage_current(1.5)
        assert current_high > current

        # Zero voltage should give zero current
        current_zero = vcma_device.compute_leakage_current(0.0)
        assert current_zero == 0.0


class TestSkyrmionDevice:
    """Test skyrmion device model."""

    @pytest.fixture
    def device_params(self):
        """Skyrmion device parameters."""
        return {
            'volume': 1e-24,  # m³
            'saturation_magnetization': 800e3,  # A/m
            'exchange_constant': 2e-11,  # J/m
            'dmi_constant': 3e-3,  # J/m²
            'skyrmion_radius': 20e-9,  # m
            'track_width': 200e-9,  # m
            'spin_hall_angle': 0.1,
            'interface_transparency': 0.5,
            'pinning_strength': 0.1,
            'damping': 0.3,
            'uniaxial_anisotropy': 1e6  # J/m³
        }

    @pytest.fixture
    def skyrmion_device(self, device_params):
        """Create skyrmion device instance."""
        return SkyrmionDevice(device_params)

    def test_device_initialization(self, skyrmion_device, device_params):
        """Test device initialization."""
        assert skyrmion_device.dmi_constant == device_params['dmi_constant']
        assert skyrmion_device.skyrmion_radius == device_params['skyrmion_radius']
        assert skyrmion_device.track_width == device_params['track_width']
        assert hasattr(skyrmion_device, 'exchange_length')
        assert hasattr(skyrmion_device, 'magnus_coefficient')

    def test_parameter_validation(self, device_params):
        """Test skyrmion parameter validation."""
        # Test zero DMI (should warn)
        no_dmi_params = device_params.copy()
        no_dmi_params['dmi_constant'] = 0

        with pytest.warns(UserWarning, match="Zero DMI constant"):
            SkyrmionDevice(no_dmi_params)

    def test_effective_field_computation(self, skyrmion_device):
        """Test effective field computation for skyrmion."""
        position = np.array([50e-9, 100e-9])  # m
        applied_field = np.array([100, 0, 0])  # A/m

        h_eff = skyrmion_device.compute_effective_field(position, applied_field=applied_field)

        assert h_eff.shape == (3,)
        assert np.linalg.norm(h_eff) > 0

    def test_skyrmion_velocity_calculation(self, skyrmion_device):
        """Test skyrmion velocity under current."""
        current_density = np.array([1e6, 0])  # A/m²
        position = np.array([50e-9, 100e-9])  # m

        velocity = skyrmion_device.compute_skyrmion_velocity(current_density, position)

        assert velocity.shape == (2,)
        assert np.linalg.norm(velocity) > 0

        # Zero current should give zero velocity
        velocity_zero = skyrmion_device.compute_skyrmion_velocity(np.array([0, 0]), position)
        assert np.allclose(velocity_zero, np.array([0, 0]))

    def test_skyrmion_hall_angle(self, skyrmion_device):
        """Test skyrmion Hall angle calculation."""
        hall_angle = skyrmion_device._compute_skyrmion_hall_angle()

        assert 0 < hall_angle < np.pi/2
        assert np.degrees(hall_angle) >= 5  # Minimum clipping
        assert np.degrees(hall_angle) <= 45  # Maximum clipping

    def test_stability_calculation(self, skyrmion_device):
        """Test skyrmion stability calculation."""
        position = np.array([50e-9, 100e-9])  # m
        temperature = 300.0  # K

        stability = skyrmion_device.compute_skyrmion_stability(position, temperature)

        assert 0 <= stability <= 1

        # Higher temperature should reduce stability
        stability_high_T = skyrmion_device.compute_skyrmion_stability(position, 400.0)
        assert stability_high_T <= stability

    def test_skyrmion_energy(self, skyrmion_device):
        """Test skyrmion energy calculation."""
        energy = skyrmion_device._compute_skyrmion_energy()

        assert isinstance(energy, float)
        # Total energy can be positive or negative depending on DMI strength

    def test_resistance_calculation(self, skyrmion_device):
        """Test resistance with skyrmion configuration."""
        # Single skyrmion
        skyrmion_positions = [np.array([50e-9, 100e-9])]

        resistance = skyrmion_device.compute_resistance(skyrmion_positions)

        assert resistance > 1.0  # Minimum resistance

        # Multiple skyrmions should affect resistance
        multiple_skyrmions = [np.array([50e-9, 100e-9]), np.array([150e-9, 100e-9])]
        resistance_multi = skyrmion_device.compute_resistance(multiple_skyrmions)
        assert resistance_multi != resistance

    def test_power_consumption(self, skyrmion_device):
        """Test power consumption for skyrmion manipulation."""
        current_density = np.array([1e6, 0])  # A/m²
        pulse_duration = 1e-9  # s
        skyrmion_positions = [np.array([50e-9, 100e-9])]

        energy = skyrmion_device.compute_power_consumption(current_density, pulse_duration, skyrmion_positions)

        assert energy > 0

        # Zero current should consume no energy
        energy_zero = skyrmion_device.compute_power_consumption(np.array([0, 0]), pulse_duration, skyrmion_positions)
        assert energy_zero == 0.0

    def test_position_validation(self, skyrmion_device):
        """Test skyrmion position validation."""
        # Valid position
        valid_pos = np.array([50e-9, 100e-9])
        validated = skyrmion_device.validate_position(valid_pos)
        assert np.allclose(validated, valid_pos)

        # Position outside track
        invalid_pos = np.array([50e-9, 300e-9])  # y > track_width
        with pytest.raises(ValueError, match="outside track width"):
            skyrmion_device.validate_position(invalid_pos)

        # Wrong dimensions
        with pytest.raises(ValueError, match="must be a 2D vector"):
            skyrmion_device.validate_position(np.array([1, 2, 3]))

    def test_motion_time_estimation(self, skyrmion_device):
        """Test motion time estimation."""
        initial_pos = np.array([50e-9, 100e-9])
        target_pos = np.array([150e-9, 100e-9])
        current_density = 1e6  # A/m²

        time_estimate = skyrmion_device.estimate_motion_time(initial_pos, target_pos, current_density)

        assert time_estimate > 0
        assert time_estimate < np.inf

        # Zero current should give infinite time
        time_zero = skyrmion_device.estimate_motion_time(initial_pos, target_pos, 0.0)
        assert time_zero == np.inf


class TestDeviceFactory:
    """Test device factory functionality."""

    @pytest.fixture
    def factory(self):
        """Create device factory instance."""
        return DeviceFactory()

    def test_available_devices(self, factory):
        """Test getting available device types."""
        devices = factory.get_available_devices()

        expected_devices = ['stt_mram', 'sot_mram', 'vcma_mram', 'skyrmion', 'skyrmion_track']

        for expected in expected_devices:
            assert expected in devices

    def test_device_creation(self, factory):
        """Test creating different device types."""
        # STT-MRAM
        stt_params = {
            'volume': 1e-24,
            'saturation_magnetization': 800e3,
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,
            'polarization': 0.7,
            'easy_axis': np.array([0, 0, 1])
        }
        stt_device = factory.create_device('stt_mram', stt_params)
        assert isinstance(stt_device, STTMRAMDevice)

        # SOT-MRAM
        sot_params = {
            'volume': 1e-24,
            'saturation_magnetization': 800e3,
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,
            'easy_axis': np.array([0, 0, 1]),
            'spin_hall_angle': 0.1
        }
        sot_device = factory.create_device('sot_mram', sot_params)
        assert isinstance(sot_device, SOTMRAMDevice)

        # VCMA-MRAM
        vcma_params = {
            'volume': 1e-24,
            'saturation_magnetization': 800e3,
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,
            'easy_axis': np.array([0, 0, 1]),
            'vcma_coefficient': 100e-6
        }
        vcma_device = factory.create_device('vcma_mram', vcma_params)
        assert isinstance(vcma_device, VCMAMRAMDevice)

        # Skyrmion
        skyrmion_params = {
            'volume': 1e-24,
            'saturation_magnetization': 800e3,
            'exchange_constant': 2e-11,
            'dmi_constant': 3e-3,
            'skyrmion_radius': 20e-9
        }
        skyrmion_device = factory.create_device('skyrmion', skyrmion_params)
        assert isinstance(skyrmion_device, SkyrmionDevice)

    def test_invalid_device_type(self, factory):
        """Test creating invalid device type."""
        with pytest.raises(ValueError, match="Unknown device type"):
            factory.create_device('invalid_device', {})

    def test_device_info_retrieval(self, factory):
        """Test retrieving device information."""
        # Create a device and get its info
        params = {
            'volume': 1e-24,
            'saturation_magnetization': 800e3,
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,
            'polarization': 0.7,
            'easy_axis': np.array([0, 0, 1])
        }
        device = factory.create_device('stt_mram', params)

        # Test that device has required methods
        assert hasattr(device, 'compute_effective_field')
        assert hasattr(device, 'compute_resistance')
        assert hasattr(device, 'validate_magnetization')

        # Test calling methods
        magnetization = np.array([0, 0, 1])
        applied_field = np.array([100, 0, 0])

        h_eff = device.compute_effective_field(magnetization, applied_field)
        resistance = device.compute_resistance(magnetization)

        assert h_eff.shape == (3,)
        assert isinstance(resistance, float)
        assert resistance > 0
