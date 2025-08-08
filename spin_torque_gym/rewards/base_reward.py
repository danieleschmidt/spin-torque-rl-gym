"""Base reward function interface for spintronic environments.

This module defines the abstract base class for all reward functions
used in spintronic device control environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import numpy as np


class BaseReward(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, weight: float = 1.0, **kwargs):
        """Initialize base reward function.
        
        Args:
            weight: Weight for this reward component
            **kwargs: Additional parameters for specific reward implementations
        """
        self.weight = weight
        self.kwargs = kwargs
    
    @abstractmethod
    def compute(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        action: Union[np.ndarray, int, float],
        next_observation: Union[np.ndarray, Dict[str, Any]],
        info: Dict[str, Any]
    ) -> float:
        """Compute reward value.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Observation after action
            info: Additional information dictionary
            
        Returns:
            Reward value (unweighted)
        """
        pass
    
    def __call__(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        action: Union[np.ndarray, int, float],
        next_observation: Union[np.ndarray, Dict[str, Any]],
        info: Dict[str, Any]
    ) -> float:
        """Callable interface for reward computation.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Observation after action
            info: Additional information dictionary
            
        Returns:
            Weighted reward value
        """
        return self.weight * self.compute(observation, action, next_observation, info)
    
    def reset(self) -> None:
        """Reset reward function state (if any)."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get reward function information.
        
        Returns:
            Dictionary with reward function parameters and state
        """
        return {
            'type': self.__class__.__name__,
            'weight': self.weight,
            'parameters': self.kwargs
        }


class FunctionBasedReward(BaseReward):
    """Reward function based on a callable function."""
    
    def __init__(self, function, weight: float = 1.0, **kwargs):
        """Initialize function-based reward.
        
        Args:
            function: Callable that takes (obs, action, next_obs, info) and returns reward
            weight: Weight for this reward component
            **kwargs: Additional parameters
        """
        super().__init__(weight, **kwargs)
        self.function = function
    
    def compute(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        action: Union[np.ndarray, int, float],
        next_observation: Union[np.ndarray, Dict[str, Any]],
        info: Dict[str, Any]
    ) -> float:
        """Compute reward using the provided function."""
        return self.function(observation, action, next_observation, info)