"""Configuration management for Spin-Torque RL-Gym.

This module provides centralized configuration management with support for
environment variables, JSON/YAML config files, and runtime parameter overrides.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    temperature: float = 300.0  # Kelvin
    solver_method: str = 'RK45'
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9
    max_timestep: float = 1e-12
    include_thermal: bool = True
    include_quantum: bool = False
    random_seed: Optional[int] = None


@dataclass  
class DeviceConfig:
    """Device configuration."""
    device_type: str = 'stt_mram'
    volume: float = 1e-24  # m³
    saturation_magnetization: float = 800e3  # A/m
    damping: float = 0.01
    uniaxial_anisotropy: float = 1e6  # J/m³
    polarization: float = 0.7
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training environment configuration."""
    max_steps: int = 100
    max_current: float = 2e6  # A/m²
    max_duration: float = 5e-9  # seconds
    success_threshold: float = 0.9
    action_mode: str = 'continuous'  # 'continuous' or 'discrete'
    observation_mode: str = 'vector'  # 'vector' or 'dict'
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'success': 10.0,
        'energy': -0.1,
        'progress': 1.0,
        'stability': -2.0
    })


@dataclass
class ComputeConfig:
    """Computational resource configuration."""
    use_jax: bool = False
    gpu_device: int = -1
    num_cores: int = 0  # 0 = auto-detect
    memory_limit_gb: float = 8.0
    enable_profiling: bool = False


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_level: str = 'INFO'
    log_dir: str = './logs'
    enable_tensorboard: bool = False
    tensorboard_dir: str = './runs'
    save_episodes: bool = True
    episode_dir: str = './episodes'
    wandb_project: Optional[str] = None
    mlflow_uri: Optional[str] = None


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    render_mode: Optional[str] = None
    animation_fps: int = 30
    save_animations: bool = False
    matplotlib_backend: str = 'Agg'
    output_dir: str = './visualizations'


@dataclass
class SpinTorqueConfig:
    """Complete configuration for Spin-Torque RL-Gym."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    debug_mode: bool = False
    strict_mode: bool = False
    safe_mode: bool = False


class ConfigManager:
    """Configuration manager with multiple source support."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = SpinTorqueConfig()
        self._env_prefix = 'SPIN_TORQUE_'
        
        # Load configuration from multiple sources
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from all sources in priority order."""
        # 1. Load defaults (already in dataclass)
        
        # 2. Load from config file if provided
        if self.config_path and self.config_path.exists():
            self._load_from_file(self.config_path)
        
        # 3. Load from environment variables (highest priority)
        self._load_from_env()
        
        # 4. Validate configuration
        self._validate_config()
    
    def _load_from_file(self, config_path: Path) -> None:
        """Load configuration from JSON or YAML file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Update configuration with file data
            self._update_config_from_dict(data)
            
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # Physics config
            f'{self._env_prefix}TEMPERATURE': ('physics', 'temperature', float),
            f'{self._env_prefix}SOLVER_METHOD': ('physics', 'solver_method', str),
            f'{self._env_prefix}SOLVER_RTOL': ('physics', 'solver_rtol', float),
            f'{self._env_prefix}SOLVER_ATOL': ('physics', 'solver_atol', float),
            f'{self._env_prefix}MAX_TIMESTEP': ('physics', 'max_timestep', float),
            f'{self._env_prefix}INCLUDE_THERMAL': ('physics', 'include_thermal', self._parse_bool),
            f'{self._env_prefix}INCLUDE_QUANTUM': ('physics', 'include_quantum', self._parse_bool),
            f'{self._env_prefix}SEED': ('physics', 'random_seed', int),
            
            # Device config
            f'{self._env_prefix}DEVICE_TYPE': ('device', 'device_type', str),
            f'{self._env_prefix}VOLUME': ('device', 'volume', float),
            f'{self._env_prefix}SATURATION_MAGNETIZATION': ('device', 'saturation_magnetization', float),
            f'{self._env_prefix}DAMPING': ('device', 'damping', float),
            f'{self._env_prefix}ANISOTROPY': ('device', 'uniaxial_anisotropy', float),
            f'{self._env_prefix}POLARIZATION': ('device', 'polarization', float),
            
            # Training config  
            f'{self._env_prefix}MAX_STEPS': ('training', 'max_steps', int),
            f'{self._env_prefix}MAX_CURRENT': ('training', 'max_current', float),
            f'{self._env_prefix}MAX_DURATION': ('training', 'max_duration', float),
            f'{self._env_prefix}SUCCESS_THRESHOLD': ('training', 'success_threshold', float),
            f'{self._env_prefix}ACTION_MODE': ('training', 'action_mode', str),
            f'{self._env_prefix}OBSERVATION_MODE': ('training', 'observation_mode', str),
            
            # Compute config
            f'{self._env_prefix}USE_JAX': ('compute', 'use_jax', self._parse_bool),
            f'{self._env_prefix}GPU_DEVICE': ('compute', 'gpu_device', int),
            f'{self._env_prefix}NUM_CORES': ('compute', 'num_cores', int),
            f'{self._env_prefix}MEMORY_LIMIT_GB': ('compute', 'memory_limit_gb', float),
            f'{self._env_prefix}ENABLE_PROFILING': ('compute', 'enable_profiling', self._parse_bool),
            
            # Logging config
            f'{self._env_prefix}LOG_LEVEL': ('logging', 'log_level', str),
            f'{self._env_prefix}LOG_DIR': ('logging', 'log_dir', str),
            f'{self._env_prefix}ENABLE_TENSORBOARD': ('logging', 'enable_tensorboard', self._parse_bool),
            f'{self._env_prefix}TENSORBOARD_DIR': ('logging', 'tensorboard_dir', str),
            f'{self._env_prefix}SAVE_EPISODES': ('logging', 'save_episodes', self._parse_bool),
            f'{self._env_prefix}EPISODE_DIR': ('logging', 'episode_dir', str),
            f'{self._env_prefix}WANDB_PROJECT': ('logging', 'wandb_project', str),
            f'{self._env_prefix}MLFLOW_URI': ('logging', 'mlflow_uri', str),
            
            # Visualization config
            f'{self._env_prefix}RENDER_MODE': ('visualization', 'render_mode', str),
            f'{self._env_prefix}ANIMATION_FPS': ('visualization', 'animation_fps', int),
            f'{self._env_prefix}SAVE_ANIMATIONS': ('visualization', 'save_animations', self._parse_bool),
            f'{self._env_prefix}MPL_BACKEND': ('visualization', 'matplotlib_backend', str),
            f'{self._env_prefix}VIZ_OUTPUT_DIR': ('visualization', 'output_dir', str),
            
            # Global settings
            f'{self._env_prefix}DEBUG_MODE': ('debug_mode', bool),
            f'{self._env_prefix}STRICT_MODE': ('strict_mode', bool), 
            f'{self._env_prefix}SAFE_MODE': ('safe_mode', bool),
        }
        
        for env_var, (section, key, dtype) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    parsed_value = dtype(value)
                    if isinstance(section, str):
                        # Nested config (e.g., 'physics', 'temperature')
                        section_obj = getattr(self.config, section)
                        setattr(section_obj, key, parsed_value)
                    else:
                        # Top-level config (e.g., 'debug_mode')
                        setattr(self.config, section, parsed_value)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string."""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary data."""
        for section_name, section_data in data.items():
            if hasattr(self.config, section_name):
                section_obj = getattr(self.config, section_name)
                
                # Update section attributes
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                        elif hasattr(section_obj, 'custom_params'):
                            # Store unknown parameters in custom_params
                            section_obj.custom_params[key] = value
                else:
                    # Direct assignment for non-nested configs
                    setattr(self.config, section_name, section_data)
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Physics validation
        if self.config.physics.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.config.physics.solver_rtol <= 0 or self.config.physics.solver_atol <= 0:
            raise ValueError("Solver tolerances must be positive")
        
        # Device validation
        if self.config.device.volume <= 0:
            raise ValueError("Device volume must be positive")
        if self.config.device.saturation_magnetization <= 0:
            raise ValueError("Saturation magnetization must be positive")
        if not 0 <= self.config.device.damping <= 1:
            raise ValueError("Damping must be between 0 and 1")
        if not 0 <= self.config.device.polarization <= 1:
            raise ValueError("Polarization must be between 0 and 1")
        
        # Training validation
        if self.config.training.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        if self.config.training.max_current <= 0:
            raise ValueError("Max current must be positive")
        if self.config.training.max_duration <= 0:
            raise ValueError("Max duration must be positive")
        if not 0 <= self.config.training.success_threshold <= 1:
            raise ValueError("Success threshold must be between 0 and 1")
        
        # Create directories if they don't exist
        for dir_path in [
            self.config.logging.log_dir,
            self.config.logging.tensorboard_dir,
            self.config.logging.episode_dir,
            self.config.visualization.output_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_config(self) -> SpinTorqueConfig:
        """Get the complete configuration object."""
        return self.config
    
    def get_section(self, section_name: str) -> Any:
        """Get a specific configuration section."""
        if not hasattr(self.config, section_name):
            raise ValueError(f"Unknown config section: {section_name}")
        return getattr(self.config, section_name)
    
    def update_config(
        self,
        updates: Dict[str, Any],
        validate: bool = True
    ) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates (nested structure supported)
            validate: Whether to validate after updates
        """
        self._update_config_from_dict(updates)
        
        if validate:
            self._validate_config()
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        
        # Convert config to dictionary
        config_dict = asdict(self.config)
        
        # Save based on file extension
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(config_path={self.config_path})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"SpinTorque Configuration:\n{yaml.dump(asdict(self.config), default_flow_style=False)}"


# Global configuration manager instance
_config_manager = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> SpinTorqueConfig:
    """Get global configuration instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        Configuration object
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.get_config()


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration.
    
    Args:
        updates: Configuration updates
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    _config_manager.update_config(updates)


def reset_config() -> None:
    """Reset global configuration manager."""
    global _config_manager
    _config_manager = None


# Convenience functions for specific config sections
def get_physics_config() -> PhysicsConfig:
    """Get physics configuration."""
    return get_config().physics


def get_device_config() -> DeviceConfig:
    """Get device configuration."""
    return get_config().device


def get_training_config() -> TrainingConfig:
    """Get training configuration."""
    return get_config().training


def get_compute_config() -> ComputeConfig:
    """Get compute configuration."""
    return get_config().compute


def get_logging_config() -> LoggingConfig:
    """Get logging configuration.""" 
    return get_config().logging


def get_visualization_config() -> VisualizationConfig:
    """Get visualization configuration."""
    return get_config().visualization