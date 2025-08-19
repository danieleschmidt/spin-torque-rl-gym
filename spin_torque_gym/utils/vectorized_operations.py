"""Vectorized operations for high-performance physics computations.

This module implements vectorized versions of key physics operations
for significant performance improvements in batch processing scenarios.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorizedSolver:
    """High-performance vectorized LLGS solver for batch processing."""

    def __init__(self):
        """Initialize vectorized solver."""
        # Physical constants
        self.gamma = 2.21e5  # Gyromagnetic ratio (m/(A·s))
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        
        # Numerical parameters
        self.max_iterations = 1000
        self.tolerance = 1e-6
        
        # Performance tracking
        self.vectorized_operations = 0
        self.batch_count = 0

    def solve_batch(
        self,
        m_initial_batch: np.ndarray,
        t_span: Tuple[float, float],
        device_params_batch: List[Dict[str, Any]],
        dt: float = 1e-12
    ) -> List[Dict[str, Any]]:
        """Solve batch of magnetization dynamics using vectorized operations.
        
        Args:
            m_initial_batch: Initial magnetizations (N, 3) array
            t_span: Time span (t_start, t_end)
            device_params_batch: List of device parameter dictionaries
            dt: Time step size
            
        Returns:
            List of solution dictionaries
        """
        self.batch_count += 1
        batch_size = m_initial_batch.shape[0]
        
        if batch_size == 0:
            return []
        
        t_start, t_end = t_span
        n_steps = max(10, int((t_end - t_start) / dt))
        actual_dt = (t_end - t_start) / n_steps
        
        # Time array
        t_array = np.linspace(t_start, t_end, n_steps + 1)
        
        # Initialize magnetization trajectories (batch_size, n_steps+1, 3)
        m_trajectories = np.zeros((batch_size, n_steps + 1, 3))
        m_trajectories[:, 0, :] = m_initial_batch
        
        # Extract and vectorize device parameters
        params_vectorized = self._vectorize_device_params(device_params_batch)
        
        try:
            # Vectorized time integration
            for i in range(n_steps):
                # Current magnetizations (batch_size, 3)
                m_current = m_trajectories[:, i, :]
                
                # Compute derivatives for all samples simultaneously
                dmdt_batch = self._compute_dmdt_vectorized(
                    m_current, 
                    t_array[i], 
                    params_vectorized,
                    actual_dt
                )
                
                # Update magnetizations
                m_next = m_current + actual_dt * dmdt_batch
                
                # Normalize all magnetizations
                m_next = self._normalize_batch(m_next)
                
                m_trajectories[:, i + 1, :] = m_next
            
            # Convert to individual result dictionaries
            results = []
            for j in range(batch_size):
                results.append({
                    't': t_array.copy(),
                    'm': m_trajectories[j, :, :].copy(),
                    'success': True,
                    'message': 'Vectorized integration completed',
                    'n_steps': n_steps,
                    'vectorized': True
                })
            
            self.vectorized_operations += 1
            logger.debug(f"Vectorized batch solve completed: {batch_size} problems")
            
            return results
            
        except Exception as e:
            logger.error(f"Vectorized solve failed: {e}")
            # Return fallback results
            return [
                {
                    't': np.array([t_start, t_end]),
                    'm': np.array([m_initial_batch[j], m_initial_batch[j]]),
                    'success': False,
                    'message': f'Vectorized solve failed: {e}',
                    'vectorized': False
                }
                for j in range(batch_size)
            ]

    def solve_single(
        self,
        m_initial: np.ndarray,
        t_span: Tuple[float, float],
        device_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Solve single problem using vectorized operations."""
        # Convert to batch of size 1
        m_batch = m_initial.reshape(1, -1)
        params_batch = [device_params]
        
        results = self.solve_batch(m_batch, t_span, params_batch)
        return results[0]

    def _vectorize_device_params(
        self, 
        device_params_batch: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Vectorize device parameters for batch processing."""
        if not device_params_batch:
            return {}
        
        # Get parameter names from first entry
        param_names = device_params_batch[0].keys()
        vectorized_params = {}
        
        for param_name in param_names:
            # Extract parameter values
            param_values = []
            for params in device_params_batch:
                value = params.get(param_name, 0.0)
                
                # Handle vector parameters (like easy_axis)
                if isinstance(value, np.ndarray):
                    param_values.append(value)
                else:
                    param_values.append(float(value))
            
            # Convert to numpy array
            if isinstance(param_values[0], np.ndarray):
                # Stack vector parameters
                vectorized_params[param_name] = np.stack(param_values, axis=0)
            else:
                # Convert scalar parameters
                vectorized_params[param_name] = np.array(param_values)
        
        return vectorized_params

    def _compute_dmdt_vectorized(
        self,
        m_batch: np.ndarray,
        t: float,
        params_vectorized: Dict[str, np.ndarray],
        dt: float
    ) -> np.ndarray:
        """Compute dm/dt for batch of magnetizations using vectorized operations."""
        batch_size = m_batch.shape[0]
        
        # Extract vectorized parameters with defaults
        alpha = params_vectorized.get('damping', np.full(batch_size, 0.01))
        ms = params_vectorized.get('saturation_magnetization', np.full(batch_size, 800e3))
        k_u = params_vectorized.get('uniaxial_anisotropy', np.full(batch_size, 1e6))
        
        # Handle easy axis (might be vectorized or scalar)
        if 'easy_axis' in params_vectorized:
            easy_axis = params_vectorized['easy_axis']
            if easy_axis.ndim == 1:
                # Single easy axis for all samples
                easy_axis = np.tile(easy_axis, (batch_size, 1))
        else:
            # Default easy axis
            easy_axis = np.tile(np.array([0, 0, 1]), (batch_size, 1))
        
        # Ensure easy axes are normalized
        easy_axis_norms = np.linalg.norm(easy_axis, axis=1, keepdims=True)
        easy_axis = easy_axis / np.maximum(easy_axis_norms, 1e-12)
        
        # Compute effective field vectorized
        h_eff = self._compute_effective_field_vectorized(
            m_batch, params_vectorized, t
        )
        
        # LLGS equation components (vectorized)
        # Precession term: m × H_eff
        precession = np.cross(m_batch, h_eff)
        
        # Damping term: α * m × (m × H_eff)  
        alpha_expanded = alpha.reshape(-1, 1)  # (batch_size, 1)
        damping_term = alpha_expanded * np.cross(m_batch, precession)
        
        # Combine terms
        gamma_eff = self.gamma / (1 + alpha**2)
        gamma_eff_expanded = gamma_eff.reshape(-1, 1)
        
        dmdt = -gamma_eff_expanded * (precession + damping_term)
        
        return dmdt

    def _compute_effective_field_vectorized(
        self,
        m_batch: np.ndarray,
        params_vectorized: Dict[str, np.ndarray],
        t: float
    ) -> np.ndarray:
        """Compute effective field for batch of magnetizations."""
        batch_size = m_batch.shape[0]
        
        # Initialize effective field
        h_eff = np.zeros_like(m_batch)
        
        # Extract parameters
        k_u = params_vectorized.get('uniaxial_anisotropy', np.full(batch_size, 1e6))
        ms = params_vectorized.get('saturation_magnetization', np.full(batch_size, 800e3))
        
        # Easy axis handling
        if 'easy_axis' in params_vectorized:
            easy_axis = params_vectorized['easy_axis']
            if easy_axis.ndim == 1:
                easy_axis = np.tile(easy_axis, (batch_size, 1))
        else:
            easy_axis = np.tile(np.array([0, 0, 1]), (batch_size, 1))
        
        # Normalize easy axes
        easy_axis_norms = np.linalg.norm(easy_axis, axis=1, keepdims=True)
        easy_axis = easy_axis / np.maximum(easy_axis_norms, 1e-12)
        
        # Anisotropy field (vectorized)
        # H_anis = (2 * K_u / (μ₀ * M_s)) * (m · ê) * ê
        h_k_magnitude = (2 * k_u) / (self.mu_0 * ms)  # (batch_size,)
        m_dot_easy = np.sum(m_batch * easy_axis, axis=1)  # (batch_size,)
        
        # Vectorized anisotropy field
        h_anis = (h_k_magnitude * m_dot_easy).reshape(-1, 1) * easy_axis
        h_eff += h_anis
        
        # Simplified demagnetization field (shape anisotropy)
        # Assumes thin film geometry with out-of-plane easy axis
        ms_expanded = ms.reshape(-1, 1)
        h_demag = -ms_expanded * m_batch[:, [2]] * np.array([0, 0, 1])
        h_eff += h_demag
        
        return h_eff

    def _normalize_batch(self, m_batch: np.ndarray) -> np.ndarray:
        """Normalize batch of magnetization vectors."""
        # Compute norms for each vector
        norms = np.linalg.norm(m_batch, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        
        return m_batch / norms

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get vectorized solver performance statistics."""
        return {
            'vectorized_operations': self.vectorized_operations,
            'batch_count': self.batch_count,
            'average_vectorization_ratio': (
                self.vectorized_operations / max(self.batch_count, 1)
            )
        }


class VectorizedMagneticsOperations:
    """Vectorized implementations of common magnetic operations."""

    @staticmethod
    def batch_cross_product(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """Compute cross product for batch of 3D vectors.
        
        Args:
            a_batch: Array of shape (N, 3)
            b_batch: Array of shape (N, 3)
            
        Returns:
            Cross products array of shape (N, 3)
        """
        return np.cross(a_batch, b_batch, axis=1)

    @staticmethod
    def batch_dot_product(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """Compute dot product for batch of 3D vectors.
        
        Args:
            a_batch: Array of shape (N, 3)
            b_batch: Array of shape (N, 3)
            
        Returns:
            Dot products array of shape (N,)
        """
        return np.sum(a_batch * b_batch, axis=1)

    @staticmethod
    def batch_normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize batch of vectors.
        
        Args:
            vectors: Array of shape (N, 3)
            
        Returns:
            Normalized vectors of shape (N, 3)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return vectors / norms

    @staticmethod
    def batch_energy_computation(
        m_batch: np.ndarray,
        params_batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute magnetic energy for batch of states.
        
        Args:
            m_batch: Magnetizations array of shape (N, 3)
            params_batch: Vectorized parameters dictionary
            
        Returns:
            Energies array of shape (N,)
        """
        batch_size = m_batch.shape[0]
        
        # Extract parameters
        k_u = params_batch.get('uniaxial_anisotropy', np.full(batch_size, 1e6))
        volume = params_batch.get('volume', np.full(batch_size, 1e-24))
        
        # Easy axis
        if 'easy_axis' in params_batch:
            easy_axis = params_batch['easy_axis']
            if easy_axis.ndim == 1:
                easy_axis = np.tile(easy_axis, (batch_size, 1))
        else:
            easy_axis = np.tile(np.array([0, 0, 1]), (batch_size, 1))
        
        # Anisotropy energy: -K_u * V * (m · ê)²
        m_dot_easy = VectorizedMagneticsOperations.batch_dot_product(m_batch, easy_axis)
        e_anis = -k_u * volume * m_dot_easy**2
        
        return e_anis

    @staticmethod
    def batch_resistance_computation(
        m_batch: np.ndarray,
        reference_magnetizations: np.ndarray,
        r_p_batch: np.ndarray,
        r_ap_batch: np.ndarray
    ) -> np.ndarray:
        """Compute TMR resistance for batch of states.
        
        Args:
            m_batch: Free layer magnetizations (N, 3)
            reference_magnetizations: Reference magnetizations (N, 3) 
            r_p_batch: Parallel resistances (N,)
            r_ap_batch: Antiparallel resistances (N,)
            
        Returns:
            Resistances array of shape (N,)
        """
        # Compute alignment
        cos_theta = VectorizedMagneticsOperations.batch_dot_product(
            m_batch, reference_magnetizations
        )
        
        # TMR formula: R = R_P * (1 + TMR * (1 - cos θ) / 2)
        tmr = (r_ap_batch - r_p_batch) / r_p_batch
        resistances = r_p_batch * (1 + tmr * (1 - cos_theta) / 2)
        
        # Ensure positive resistance
        return np.maximum(resistances, r_p_batch * 0.5)


def optimize_with_vectorization(func):
    """Decorator to automatically vectorize compatible functions."""
    def wrapper(*args, **kwargs):
        # Check if first argument is a batch (2D array)
        if (len(args) > 0 and 
            isinstance(args[0], np.ndarray) and 
            args[0].ndim == 2 and 
            args[0].shape[1] == 3):
            
            # Use vectorized version if available
            vectorized_name = f"batch_{func.__name__}"
            if hasattr(VectorizedMagneticsOperations, vectorized_name):
                vectorized_func = getattr(VectorizedMagneticsOperations, vectorized_name)
                return vectorized_func(*args, **kwargs)
        
        # Fallback to original function
        return func(*args, **kwargs)
    
    return wrapper


# Performance utility functions
def benchmark_vectorization():
    """Benchmark vectorized vs non-vectorized operations."""
    import time
    
    print("Benchmarking vectorized operations...")
    
    # Test data
    batch_sizes = [1, 10, 100, 1000]
    n_iterations = 1000
    
    results = {}
    
    for batch_size in batch_sizes:
        # Generate random magnetizations
        m_batch = np.random.normal(0, 1, (batch_size, 3))
        m_batch = VectorizedMagneticsOperations.batch_normalize(m_batch)
        
        # Benchmark normalization
        start_time = time.time()
        for _ in range(n_iterations):
            _ = VectorizedMagneticsOperations.batch_normalize(m_batch)
        vectorized_time = time.time() - start_time
        
        # Benchmark individual operations
        start_time = time.time()
        for _ in range(n_iterations):
            for i in range(batch_size):
                m = m_batch[i]
                _ = m / np.linalg.norm(m)
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / vectorized_time if vectorized_time > 0 else 0
        results[batch_size] = {
            'vectorized_time': vectorized_time,
            'sequential_time': sequential_time,
            'speedup': speedup
        }
        
        print(f"Batch size {batch_size:4d}: "
              f"Vectorized: {vectorized_time:.4f}s, "
              f"Sequential: {sequential_time:.4f}s, "
              f"Speedup: {speedup:.1f}x")
    
    return results


if __name__ == "__main__":
    # Run benchmark
    benchmark_results = benchmark_vectorization()
    
    # Test vectorized solver
    solver = VectorizedSolver()
    
    # Test batch solve
    batch_size = 5
    m_initial_batch = np.random.normal(0, 1, (batch_size, 3))
    m_initial_batch = VectorizedMagneticsOperations.batch_normalize(m_initial_batch)
    
    device_params_batch = [
        {
            'damping': 0.01,
            'saturation_magnetization': 800e3,
            'uniaxial_anisotropy': 1e6,
            'easy_axis': np.array([0, 0, 1])
        }
        for _ in range(batch_size)
    ]
    
    results = solver.solve_batch(
        m_initial_batch,
        (0, 1e-9),
        device_params_batch
    )
    
    print(f"\nVectorized solver test:")
    print(f"Batch size: {batch_size}")
    print(f"Results: {len(results)} solutions")
    print(f"Success rate: {sum(r['success'] for r in results) / len(results) * 100:.1f}%")
    
    stats = solver.get_performance_stats()
    print(f"Solver stats: {stats}")
    
    print("\nVectorized operations module ready!")