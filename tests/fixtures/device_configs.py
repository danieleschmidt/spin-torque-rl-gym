"""Test fixtures for device configurations."""

import numpy as np

# Standard STT-MRAM device configurations
STT_MRAM_CONFIGS = {
    "standard": {
        "device_type": "stt_mram",
        "geometry": {
            "shape": "ellipse",
            "major_axis": 100e-9,  # m
            "minor_axis": 50e-9,   # m
            "thickness": 2e-9,     # m
        },
        "material": {
            "name": "CoFeB",
            "ms": 800e3,           # A/m
            "exchange": 20e-12,    # J/m
            "anisotropy": 1e6,     # J/m³
            "damping": 0.01,
            "polarization": 0.7,
        },
        "electrical": {
            "resistance_p": 5e3,   # Ω
            "resistance_ap": 10e3, # Ω
            "area": 50e-9 * 100e-9, # m²
        },
        "thermal": {
            "temperature": 300,    # K
            "thermal_stability": 60, # E_b/k_B T
        }
    },

    "low_damping": {
        "device_type": "stt_mram",
        "geometry": {
            "shape": "ellipse",
            "major_axis": 80e-9,
            "minor_axis": 40e-9,
            "thickness": 1.5e-9,
        },
        "material": {
            "name": "CoFeB",
            "ms": 750e3,
            "exchange": 18e-12,
            "anisotropy": 1.2e6,
            "damping": 0.005,      # Lower damping
            "polarization": 0.8,   # Higher polarization
        },
        "electrical": {
            "resistance_p": 8e3,
            "resistance_ap": 16e3,
            "area": 80e-9 * 40e-9,
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 80,
        }
    },

    "high_thermal_stability": {
        "device_type": "stt_mram",
        "geometry": {
            "shape": "ellipse",
            "major_axis": 120e-9,
            "minor_axis": 60e-9,
            "thickness": 2.5e-9,
        },
        "material": {
            "name": "CoFeB",
            "ms": 850e3,
            "exchange": 22e-12,
            "anisotropy": 1.5e6,   # Higher anisotropy
            "damping": 0.015,
            "polarization": 0.65,
        },
        "electrical": {
            "resistance_p": 4e3,
            "resistance_ap": 8e3,
            "area": 120e-9 * 60e-9,
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 120, # Higher stability
        }
    }
}

# SOT-MRAM device configurations
SOT_MRAM_CONFIGS = {
    "standard": {
        "device_type": "sot_mram",
        "geometry": {
            "shape": "rectangle",
            "width": 100e-9,
            "length": 200e-9,
            "thickness": 1.5e-9,
        },
        "material": {
            "name": "CoFeB/Pt",
            "ms": 800e3,
            "exchange": 20e-12,
            "anisotropy": 0.8e6,   # Lower anisotropy
            "damping": 0.02,
            "polarization": 0.6,
        },
        "sot_parameters": {
            "spin_hall_angle": 0.1,
            "spin_hall_conductivity": 2e6,  # S/m
            "field_like_efficiency": 0.05,
            "damping_like_efficiency": 0.15,
        },
        "electrical": {
            "resistance_p": 10e3,
            "resistance_ap": 20e3,
            "area": 100e-9 * 200e-9,
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 40,
        }
    }
}

# VCMA-MRAM device configurations
VCMA_MRAM_CONFIGS = {
    "standard": {
        "device_type": "vcma_mram",
        "geometry": {
            "shape": "circle",
            "diameter": 80e-9,
            "thickness": 1.2e-9,
        },
        "material": {
            "name": "CoFeB/MgO",
            "ms": 700e3,
            "exchange": 18e-12,
            "anisotropy": 0.5e6,   # Base anisotropy
            "damping": 0.008,
            "polarization": 0.75,
        },
        "vcma_parameters": {
            "vcma_coefficient": 100e-15,  # J/V·m
            "breakdown_voltage": 2.0,     # V
            "capacitance": 1e-15,         # F
        },
        "electrical": {
            "resistance_p": 15e3,
            "resistance_ap": 30e3,
            "area": np.pi * (40e-9)**2,
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 30,
        }
    }
}

# Skyrmion device configurations
SKYRMION_CONFIGS = {
    "racetrack": {
        "device_type": "skyrmion",
        "geometry": {
            "track_length": 1000e-9,  # m
            "track_width": 200e-9,    # m
            "thickness": 0.8e-9,      # m
        },
        "material": {
            "name": "Pt/Co/MgO",
            "ms": 600e3,
            "exchange": 15e-12,
            "anisotropy": 1.2e6,
            "damping": 0.03,
            "dmi": 2e-3,              # J/m² (Dzyaloshinskii-Moriya)
        },
        "skyrmion_parameters": {
            "radius": 20e-9,          # m
            "mobility": 1e-12,        # m²/A·s
            "magnus_force_factor": 0.4,
            "pinning_strength": 0.1,  # Dimensionless
        },
        "positions": {
            "memory_positions": [100e-9, 300e-9, 500e-9, 700e-9, 900e-9],
            "read_position": 50e-9,
            "write_position": 950e-9,
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 50,
        }
    }
}

# Test scenarios for different device configurations
TEST_SCENARIOS = {
    "easy_switching": {
        "description": "Easy switching scenario for initial training",
        "device_modifications": {
            "thermal_stability": 20,
            "damping": 0.02,
            "temperature": 77,  # Low temperature
        },
        "target_alignment": 0.9,
        "max_steps": 50,
    },

    "standard_switching": {
        "description": "Standard switching scenario",
        "device_modifications": {},  # Use default parameters
        "target_alignment": 0.95,
        "max_steps": 100,
    },

    "hard_switching": {
        "description": "Challenging switching scenario",
        "device_modifications": {
            "thermal_stability": 100,
            "damping": 0.005,
            "temperature": 400,  # High temperature
        },
        "target_alignment": 0.98,
        "max_steps": 200,
    },

    "ultra_fast_switching": {
        "description": "Ultra-fast switching with time constraints",
        "device_modifications": {
            "damping": 0.001,
        },
        "target_alignment": 0.95,
        "max_steps": 20,  # Very limited time
        "time_penalty_weight": 2.0,
    },

    "energy_efficient": {
        "description": "Energy-efficient switching scenario",
        "device_modifications": {},
        "target_alignment": 0.9,
        "max_steps": 150,
        "energy_penalty_weight": 1.0,
        "current_limit": 1e6,  # A/cm² (reduced current limit)
    },

    "thermal_robust": {
        "description": "High-temperature operation",
        "device_modifications": {
            "temperature": 450,  # High temperature
            "thermal_stability": 40,  # Reduced stability at high T
        },
        "target_alignment": 0.92,
        "max_steps": 120,
        "include_thermal_fluctuations": True,
    }
}

# Experimental validation data
EXPERIMENTAL_BENCHMARKS = {
    "IBM_2016": {
        "reference": "IBM Research, Nature 2016",
        "device_params": STT_MRAM_CONFIGS["standard"],
        "switching_current": 2.5e6,  # A/cm²
        "switching_time": 10e-9,     # s
        "success_rate": 0.95,
        "write_energy": 5e-12,       # J
        "read_disturb_rate": 1e-6,
    },

    "Intel_2018": {
        "reference": "Intel, IEDM 2018",
        "device_params": STT_MRAM_CONFIGS["low_damping"],
        "switching_current": 1.8e6,
        "switching_time": 5e-9,
        "success_rate": 0.98,
        "write_energy": 3e-12,
        "read_disturb_rate": 5e-7,
    },

    "TSMC_2020": {
        "reference": "TSMC, VLSI 2020",
        "device_params": SOT_MRAM_CONFIGS["standard"],
        "switching_current": 1.2e6,
        "switching_time": 1e-9,
        "success_rate": 0.99,
        "write_energy": 1e-12,
        "read_disturb_rate": 1e-8,
    }
}

def get_device_config(device_type: str, variant: str = "standard") -> dict:
    """Get device configuration by type and variant.
    
    Args:
        device_type: Device type ("stt_mram", "sot_mram", "vcma_mram", "skyrmion")
        variant: Configuration variant ("standard", "low_damping", etc.)
    
    Returns:
        Device configuration dictionary
    """
    config_map = {
        "stt_mram": STT_MRAM_CONFIGS,
        "sot_mram": SOT_MRAM_CONFIGS,
        "vcma_mram": VCMA_MRAM_CONFIGS,
        "skyrmion": SKYRMION_CONFIGS,
    }

    if device_type not in config_map:
        raise ValueError(f"Unknown device type: {device_type}")

    if variant not in config_map[device_type]:
        available = list(config_map[device_type].keys())
        raise ValueError(f"Unknown variant '{variant}' for {device_type}. Available: {available}")

    return config_map[device_type][variant].copy()


def get_test_scenario(scenario_name: str) -> dict:
    """Get test scenario configuration.
    
    Args:
        scenario_name: Name of the test scenario
    
    Returns:
        Test scenario configuration dictionary
    """
    if scenario_name not in TEST_SCENARIOS:
        available = list(TEST_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    return TEST_SCENARIOS[scenario_name].copy()


def get_experimental_benchmark(benchmark_name: str) -> dict:
    """Get experimental benchmark data.
    
    Args:
        benchmark_name: Name of the experimental benchmark
    
    Returns:
        Experimental benchmark data dictionary
    """
    if benchmark_name not in EXPERIMENTAL_BENCHMARKS:
        available = list(EXPERIMENTAL_BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {available}")

    return EXPERIMENTAL_BENCHMARKS[benchmark_name].copy()
