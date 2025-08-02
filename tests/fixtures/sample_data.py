"""Sample data fixtures for testing."""

import numpy as np
from typing import Dict, List, Tuple


def generate_magnetization_trajectory(
    initial: np.ndarray,
    target: np.ndarray,
    num_steps: int = 100,
    noise_level: float = 0.0
) -> np.ndarray:
    """Generate sample magnetization trajectory.
    
    Args:
        initial: Initial magnetization vector [3]
        target: Target magnetization vector [3]
        num_steps: Number of trajectory steps
        noise_level: Noise level (0.0 = no noise)
    
    Returns:
        Trajectory array [num_steps, 3]
    """
    trajectory = np.zeros((num_steps, 3))
    
    # Spherical linear interpolation (SLERP)
    dot_product = np.dot(initial, target)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
    
    if abs(dot_product) > 0.9995:  # Vectors are nearly parallel
        # Use linear interpolation
        for i in range(num_steps):
            t = i / (num_steps - 1)
            trajectory[i] = (1 - t) * initial + t * target
            trajectory[i] = trajectory[i] / np.linalg.norm(trajectory[i])
    else:
        # Use spherical interpolation
        omega = np.arccos(abs(dot_product))
        sin_omega = np.sin(omega)
        
        for i in range(num_steps):
            t = i / (num_steps - 1)
            coeff1 = np.sin((1 - t) * omega) / sin_omega
            coeff2 = np.sin(t * omega) / sin_omega
            trajectory[i] = coeff1 * initial + coeff2 * target
            trajectory[i] = trajectory[i] / np.linalg.norm(trajectory[i])
    
    # Add noise if requested
    if noise_level > 0:
        for i in range(num_steps):
            noise = np.random.normal(0, noise_level, 3)
            trajectory[i] = trajectory[i] + noise
            trajectory[i] = trajectory[i] / np.linalg.norm(trajectory[i])
    
    return trajectory


def generate_switching_protocol(
    protocol_type: str = "constant",
    duration: float = 10e-9,
    current_magnitude: float = 2e6,
    num_steps: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample switching protocol.
    
    Args:
        protocol_type: Type of protocol ("constant", "pulse", "precessional", "optimized")
        duration: Total duration in seconds
        current_magnitude: Current magnitude in A/cm²
        num_steps: Number of time steps
    
    Returns:
        Tuple of (time_array, current_array)
    """
    time_array = np.linspace(0, duration, num_steps)
    current_array = np.zeros(num_steps)
    
    if protocol_type == "constant":
        current_array[:] = current_magnitude
    
    elif protocol_type == "pulse":
        # Square pulse for first half
        pulse_end = num_steps // 2
        current_array[:pulse_end] = current_magnitude
    
    elif protocol_type == "precessional":
        # Oscillating current for precessional switching
        frequency = 1e9  # Hz
        current_array = current_magnitude * np.sin(2 * np.pi * frequency * time_array)
    
    elif protocol_type == "optimized":
        # Example optimized protocol with multiple phases
        phase1_end = num_steps // 4
        phase2_end = 3 * num_steps // 4
        
        # Phase 1: High current pulse
        current_array[:phase1_end] = current_magnitude * 1.5
        
        # Phase 2: Moderate current
        current_array[phase1_end:phase2_end] = current_magnitude * 0.8
        
        # Phase 3: Low current for stabilization
        current_array[phase2_end:] = current_magnitude * 0.3
    
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")
    
    return time_array, current_array


def generate_experimental_data(
    num_devices: int = 100,
    num_trials_per_device: int = 10,
    base_success_rate: float = 0.95,
    variation: float = 0.1
) -> Dict[str, np.ndarray]:
    """Generate mock experimental data for validation.
    
    Args:
        num_devices: Number of test devices
        num_trials_per_device: Number of trials per device
        base_success_rate: Base success rate
        variation: Device-to-device variation
    
    Returns:
        Dictionary with experimental data arrays
    """
    np.random.seed(42)  # For reproducible test data
    
    # Device-to-device variation in success rate
    device_success_rates = np.random.normal(
        base_success_rate, 
        variation * base_success_rate, 
        num_devices
    )
    device_success_rates = np.clip(device_success_rates, 0.1, 1.0)
    
    # Generate trial results
    trial_results = []
    switching_times = []
    energies = []
    
    for device_idx in range(num_devices):
        device_sr = device_success_rates[device_idx]
        
        for trial in range(num_trials_per_device):
            # Success/failure
            success = np.random.random() < device_sr
            trial_results.append(success)
            
            # Switching time (log-normal distribution)
            if success:
                time = np.random.lognormal(np.log(2e-9), 0.5)  # ~2ns mean
            else:
                time = np.inf  # Failed to switch
            switching_times.append(time)
            
            # Energy consumption (also log-normal)
            if success:
                energy = np.random.lognormal(np.log(1e-12), 0.3)  # ~1pJ mean
            else:
                energy = np.random.lognormal(np.log(5e-12), 0.5)  # Higher for failures
            energies.append(energy)
    
    return {
        "success": np.array(trial_results, dtype=bool),
        "switching_time": np.array(switching_times),
        "energy": np.array(energies),
        "device_success_rates": device_success_rates,
        "num_devices": num_devices,
        "num_trials_per_device": num_trials_per_device
    }


def generate_phase_diagram_data(
    current_range: Tuple[float, float] = (-3e6, 3e6),
    field_range: Tuple[float, float] = (-200e-3, 200e-3),
    resolution: Tuple[int, int] = (50, 40)
) -> Dict[str, np.ndarray]:
    """Generate sample phase diagram data.
    
    Args:
        current_range: Range of current density (A/cm²)
        field_range: Range of magnetic field (T)
        resolution: Resolution (current_points, field_points)
    
    Returns:
        Dictionary with phase diagram data
    """
    current_points, field_points = resolution
    
    current_array = np.linspace(current_range[0], current_range[1], current_points)
    field_array = np.linspace(field_range[0], field_range[1], field_points)
    
    # Create meshgrid
    current_mesh, field_mesh = np.meshgrid(current_array, field_array)
    
    # Simple switching model: switching occurs when |I| + α|H| > threshold
    alpha = 5e6  # Conversion factor A/cm² per Tesla
    threshold = 1.5e6  # A/cm²
    
    effective_drive = np.abs(current_mesh) + alpha * np.abs(field_mesh)
    switching_probability = 1 / (1 + np.exp(-(effective_drive - threshold) / (0.2 * threshold)))
    
    return {
        "current_array": current_array,
        "field_array": field_array,
        "current_mesh": current_mesh,
        "field_mesh": field_mesh,
        "switching_probability": switching_probability
    }


def generate_energy_landscape_data(
    theta_resolution: int = 100,
    phi_resolution: int = 50,
    anisotropy_type: str = "uniaxial"
) -> Dict[str, np.ndarray]:
    """Generate sample energy landscape data.
    
    Args:
        theta_resolution: Resolution in theta direction
        phi_resolution: Resolution in phi direction  
        anisotropy_type: Type of anisotropy ("uniaxial", "biaxial", "cubic")
    
    Returns:
        Dictionary with energy landscape data
    """
    theta_array = np.linspace(0, 2*np.pi, theta_resolution)
    phi_array = np.linspace(0, np.pi, phi_resolution)
    
    theta_mesh, phi_mesh = np.meshgrid(theta_array, phi_array)
    
    # Convert to Cartesian coordinates
    mx = np.sin(phi_mesh) * np.cos(theta_mesh)
    my = np.sin(phi_mesh) * np.sin(theta_mesh)
    mz = np.cos(phi_mesh)
    
    # Energy calculation based on anisotropy type
    k_u = 1e6  # J/m³
    volume = 50e-9 * 100e-9 * 2e-9  # m³
    
    if anisotropy_type == "uniaxial":
        # E = -K_u cos²(φ) where φ is angle from easy axis (z)
        energy = -k_u * volume * mz**2
    
    elif anisotropy_type == "biaxial":
        # E = -K_1 cos²(φ) + K_2 cos⁴(φ)
        k_2 = 0.1 * k_u  # Smaller fourth-order term
        energy = -k_u * volume * mz**2 + k_2 * volume * mz**4
    
    elif anisotropy_type == "cubic":
        # E = K_c (mx²my² + my²mz² + mz²mx²)
        k_c = 0.1 * k_u
        energy = k_c * volume * (mx**2 * my**2 + my**2 * mz**2 + mz**2 * mx**2)
    
    else:
        raise ValueError(f"Unknown anisotropy type: {anisotropy_type}")
    
    return {
        "theta_array": theta_array,
        "phi_array": phi_array,
        "theta_mesh": theta_mesh,
        "phi_mesh": phi_mesh,
        "mx": mx,
        "my": my,
        "mz": mz,
        "energy": energy,
        "anisotropy_type": anisotropy_type
    }


def generate_training_history(
    num_episodes: int = 1000,
    learning_rate: float = 0.001,
    noise_level: float = 0.1
) -> Dict[str, np.ndarray]:
    """Generate sample training history data.
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate (affects convergence speed)
        noise_level: Noise level in learning curve
    
    Returns:
        Dictionary with training history data
    """
    np.random.seed(42)
    
    episodes = np.arange(num_episodes)
    
    # Learning curve with exponential improvement + noise
    asymptotic_reward = 10.0
    initial_reward = -5.0
    learning_speed = learning_rate * 1000
    
    base_rewards = initial_reward + (asymptotic_reward - initial_reward) * (
        1 - np.exp(-learning_speed * episodes / num_episodes)
    )
    
    # Add noise
    noise = np.random.normal(0, noise_level * asymptotic_reward, num_episodes)
    rewards = base_rewards + noise
    
    # Success rate (sigmoid improvement)
    success_rates = 1 / (1 + np.exp(-(episodes - num_episodes/2) / (num_episodes/10)))
    success_rates = 0.1 + 0.85 * success_rates  # Scale to 0.1-0.95 range
    
    # Energy efficiency (exponential improvement)
    initial_energy = 10e-12  # J
    final_energy = 1e-12     # J
    energies = initial_energy * np.exp(-learning_speed * episodes / num_episodes)
    energies = np.maximum(energies, final_energy)
    
    # Switching time (also improves)
    initial_time = 20e-9  # s
    final_time = 2e-9     # s
    switching_times = initial_time * np.exp(-learning_speed * episodes / num_episodes)
    switching_times = np.maximum(switching_times, final_time)
    
    return {
        "episodes": episodes,
        "rewards": rewards,
        "success_rates": success_rates,
        "energies": energies,
        "switching_times": switching_times,
        "base_rewards": base_rewards,
        "learning_rate": learning_rate
    }