"""Composite reward function for multi-objective optimization.

This module implements a flexible composite reward function that combines
multiple reward components with configurable weights for multi-objective
optimization in spintronic device control.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


class CompositeReward:
    """Composite reward function combining multiple objectives."""

    def __init__(self, components: Dict[str, Dict[str, Any]]):
        """Initialize composite reward function.
        
        Args:
            components: Dictionary of reward components, each containing:
                - 'weight': Weight for this component
                - 'function': Callable reward function
                - 'normalize': Optional normalization method
                - 'clip': Optional clipping bounds
        """
        self.components = {}
        self.total_weight = 0.0

        for name, config in components.items():
            self._add_component(name, config)

    def _add_component(self, name: str, config: Dict[str, Any]) -> None:
        """Add a reward component."""
        if 'weight' not in config:
            raise ValueError(f"Component '{name}' missing required 'weight'")
        if 'function' not in config:
            raise ValueError(f"Component '{name}' missing required 'function'")

        weight = float(config['weight'])
        function = config['function']

        if not callable(function):
            raise ValueError(f"Component '{name}' function must be callable")

        # Validate function signature
        try:
            # Test with dummy arguments
            test_result = function(None, None, None, {})
            if not isinstance(test_result, (int, float, np.number)):
                raise ValueError(f"Component '{name}' function must return a number")
        except Exception as e:
            warnings.warn(f"Could not validate function for component '{name}': {e}")

        self.components[name] = {
            'weight': weight,
            'function': function,
            'normalize': config.get('normalize', None),
            'clip': config.get('clip', None),
            'history': []
        }

        self.total_weight += abs(weight)

    def compute(
        self,
        observation: Optional[Any],
        action: Optional[Any],
        next_observation: Optional[Any],
        info: Dict[str, Any]
    ) -> float:
        """Compute composite reward.
        
        Args:
            observation: Current state observation
            action: Action taken
            next_observation: Next state observation  
            info: Additional information dictionary
            
        Returns:
            Total composite reward
        """
        total_reward = 0.0
        component_rewards = {}

        for name, config in self.components.items():
            try:
                # Compute component reward
                component_reward = config['function'](observation, action, next_observation, info)

                # Apply normalization if specified
                if config['normalize'] is not None:
                    component_reward = self._apply_normalization(
                        component_reward, config['normalize'], name
                    )

                # Apply clipping if specified
                if config['clip'] is not None:
                    component_reward = self._apply_clipping(
                        component_reward, config['clip']
                    )

                # Weight and add to total
                weighted_reward = config['weight'] * component_reward
                total_reward += weighted_reward

                # Store for analysis
                component_rewards[name] = component_reward
                config['history'].append(component_reward)

                # Limit history length
                if len(config['history']) > 1000:
                    config['history'] = config['history'][-500:]

            except Exception as e:
                warnings.warn(f"Error computing reward component '{name}': {e}")
                component_rewards[name] = 0.0

        # Store component breakdown in info if possible
        if hasattr(info, 'update'):
            info.update({
                'reward_components': component_rewards,
                'total_reward': total_reward
            })

        return float(total_reward)

    def _apply_normalization(
        self,
        value: float,
        normalize_method: Union[str, Dict[str, float]],
        component_name: str
    ) -> float:
        """Apply normalization to reward component."""
        if isinstance(normalize_method, str):
            if normalize_method == 'running_mean':
                # Normalize by running mean
                history = self.components[component_name]['history']
                if len(history) > 0:
                    mean_val = np.mean(history)
                    return value - mean_val
                return value

            elif normalize_method == 'running_std':
                # Normalize by running standard deviation
                history = self.components[component_name]['history']
                if len(history) > 1:
                    std_val = np.std(history)
                    mean_val = np.mean(history)
                    return (value - mean_val) / (std_val + 1e-8)
                return value

            elif normalize_method == 'unit_range':
                # Normalize to [0, 1] based on historical min/max
                history = self.components[component_name]['history']
                if len(history) > 0:
                    min_val = np.min(history)
                    max_val = np.max(history)
                    if max_val > min_val:
                        return (value - min_val) / (max_val - min_val)
                return value

            else:
                warnings.warn(f"Unknown normalization method: {normalize_method}")
                return value

        elif isinstance(normalize_method, dict):
            # Explicit normalization parameters
            if 'mean' in normalize_method and 'std' in normalize_method:
                mean = normalize_method['mean']
                std = normalize_method['std']
                return (value - mean) / (std + 1e-8)

            elif 'min' in normalize_method and 'max' in normalize_method:
                min_val = normalize_method['min']
                max_val = normalize_method['max']
                if max_val > min_val:
                    return (value - min_val) / (max_val - min_val)
                return value

            else:
                warnings.warn(f"Invalid normalization config: {normalize_method}")
                return value

        else:
            warnings.warn(f"Invalid normalization type: {type(normalize_method)}")
            return value

    def _apply_clipping(self, value: float, clip_bounds: tuple) -> float:
        """Apply clipping to reward component."""
        if len(clip_bounds) != 2:
            warnings.warn(f"Clip bounds must be tuple of length 2, got {len(clip_bounds)}")
            return value

        min_val, max_val = clip_bounds
        return np.clip(value, min_val, max_val)

    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each reward component."""
        stats = {}

        for name, config in self.components.items():
            history = config['history']

            if len(history) > 0:
                stats[name] = {
                    'mean': float(np.mean(history)),
                    'std': float(np.std(history)),
                    'min': float(np.min(history)),
                    'max': float(np.max(history)),
                    'weight': config['weight'],
                    'count': len(history)
                }
            else:
                stats[name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'weight': config['weight'],
                    'count': 0
                }

        return stats

    def reset_history(self, component_name: Optional[str] = None) -> None:
        """Reset component history.
        
        Args:
            component_name: Specific component to reset, or None for all
        """
        if component_name is not None:
            if component_name in self.components:
                self.components[component_name]['history'] = []
            else:
                warnings.warn(f"Component '{component_name}' not found")
        else:
            for config in self.components.values():
                config['history'] = []

    def update_weight(self, component_name: str, new_weight: float) -> None:
        """Update weight of a component.
        
        Args:
            component_name: Name of component to update
            new_weight: New weight value
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found")

        old_weight = self.components[component_name]['weight']
        self.components[component_name]['weight'] = float(new_weight)

        # Update total weight
        self.total_weight = self.total_weight - abs(old_weight) + abs(new_weight)

    def add_component(
        self,
        name: str,
        weight: float,
        function: Callable,
        normalize: Optional[Union[str, Dict]] = None,
        clip: Optional[tuple] = None
    ) -> None:
        """Add a new reward component.
        
        Args:
            name: Component name
            weight: Component weight
            function: Reward function
            normalize: Normalization method
            clip: Clipping bounds
        """
        if name in self.components:
            warnings.warn(f"Component '{name}' already exists, will be replaced")

        config = {
            'weight': weight,
            'function': function,
            'normalize': normalize,
            'clip': clip
        }

        self._add_component(name, config)

    def remove_component(self, name: str) -> None:
        """Remove a reward component.
        
        Args:
            name: Component name to remove
        """
        if name not in self.components:
            warnings.warn(f"Component '{name}' not found")
            return

        weight = self.components[name]['weight']
        del self.components[name]
        self.total_weight -= abs(weight)

    def get_component_names(self) -> list:
        """Get list of component names."""
        return list(self.components.keys())

    def __repr__(self) -> str:
        """String representation."""
        component_info = []
        for name, config in self.components.items():
            component_info.append(f"{name}: weight={config['weight']:.3f}")

        return f"CompositeReward({', '.join(component_info)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"CompositeReward with {len(self.components)} components (total weight: {self.total_weight:.3f})"


# Predefined reward functions for common objectives
def success_reward(observation, action, next_observation, info) -> float:
    """Binary success reward."""
    return 10.0 if info.get('is_success', False) else 0.0


def alignment_reward(observation, action, next_observation, info) -> float:
    """Reward based on target alignment."""
    return info.get('current_alignment', 0.0)


def energy_penalty(observation, action, next_observation, info) -> float:
    """Energy consumption penalty."""
    energy = info.get('step_energy', 0.0)
    return -energy / 1e-12  # Normalize to pJ


def progress_reward(observation, action, next_observation, info) -> float:
    """Reward for alignment improvement."""
    return info.get('alignment_improvement', 0.0)


def stability_penalty(observation, action, next_observation, info) -> float:
    """Penalty for magnetization instability."""
    if next_observation is None:
        return 0.0

    if isinstance(next_observation, dict):
        mag = next_observation.get('magnetization', np.array([0, 0, 1]))
    else:
        mag = next_observation[:3]  # Assume first 3 elements are magnetization

    mag_norm = np.linalg.norm(mag)
    return -max(0, mag_norm - 1.1)  # Penalty for excessive magnitude


def speed_reward(observation, action, next_observation, info) -> float:
    """Reward for fast switching."""
    step_count = info.get('step_count', 1)
    return 1.0 / (1.0 + step_count * 0.1)


# Predefined composite reward configurations
DEFAULT_REWARD_CONFIG = {
    'success': {
        'weight': 10.0,
        'function': success_reward
    },
    'energy': {
        'weight': -0.1,
        'function': energy_penalty
    },
    'progress': {
        'weight': 1.0,
        'function': progress_reward
    },
    'stability': {
        'weight': -2.0,
        'function': stability_penalty
    }
}

ENERGY_OPTIMIZED_CONFIG = {
    'success': {
        'weight': 5.0,
        'function': success_reward
    },
    'energy': {
        'weight': -1.0,
        'function': energy_penalty,
        'normalize': {'mean': 0, 'std': 1e-12}
    },
    'alignment': {
        'weight': 2.0,
        'function': alignment_reward
    }
}

SPEED_OPTIMIZED_CONFIG = {
    'success': {
        'weight': 10.0,
        'function': success_reward
    },
    'speed': {
        'weight': 5.0,
        'function': speed_reward
    },
    'progress': {
        'weight': 2.0,
        'function': progress_reward
    }
}
