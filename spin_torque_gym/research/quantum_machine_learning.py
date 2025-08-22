"""
Quantum Machine Learning for Spintronic Device Optimization

Novel implementation combining quantum computing principles with classical RL
for discovering optimal magnetization switching protocols in spintronic devices.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Using NumPy backend for quantum computations.")

from ..physics import LLGSSolver
from ..devices import BaseDevice
from ..utils.performance import PerformanceProfiler
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class QuantumSpinOptimizer:
    """Quantum-inspired optimization for spintronic switching protocols."""
    
    def __init__(
        self,
        device: BaseDevice,
        num_qubits: int = 8,
        use_quantum_annealing: bool = True,
        temperature_schedule: Optional[Dict] = None
    ):
        self.device = device
        self.num_qubits = num_qubits
        self.use_quantum_annealing = use_quantum_annealing
        self.temperature_schedule = temperature_schedule or {
            'initial': 100.0,
            'final': 0.01,
            'steps': 1000
        }
        
        # Initialize quantum state representation
        self.quantum_state = self._initialize_quantum_state()
        self.hamiltonian_cache = {}
        
    def _initialize_quantum_state(self) -> np.ndarray:
        """Initialize superposition state for optimization exploration."""
        if JAX_AVAILABLE:
            key = jax.random.PRNGKey(42)
            state = jax.random.uniform(
                key, 
                (2**self.num_qubits,), 
                dtype=jnp.complex64
            )
            return state / jnp.linalg.norm(state)
        else:
            state = np.random.uniform(0, 1, 2**self.num_qubits) + \
                   1j * np.random.uniform(0, 1, 2**self.num_qubits)
            return state / np.linalg.norm(state)
    
    def compute_switching_hamiltonian(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        current_density: float
    ) -> np.ndarray:
        """Compute Hamiltonian for quantum optimization of switching paths."""
        cache_key = (
            tuple(initial_state), 
            tuple(target_state), 
            current_density
        )
        
        if cache_key in self.hamiltonian_cache:
            return self.hamiltonian_cache[cache_key]
        
        # Construct problem Hamiltonian
        dim = 2**self.num_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # Energy landscape encoding
        for i in range(dim):
            state_encoding = self._binary_to_magnetization(i)
            energy = self._compute_state_energy(
                state_encoding, 
                initial_state, 
                target_state, 
                current_density
            )
            H[i, i] = energy
        
        # Quantum tunneling terms
        for i in range(dim):
            for j in range(i+1, dim):
                if self._hamming_distance(i, j) == 1:  # Adjacent states
                    tunneling = self._compute_tunneling_amplitude(i, j, current_density)
                    H[i, j] = tunneling
                    H[j, i] = np.conj(tunneling)
        
        self.hamiltonian_cache[cache_key] = H
        return H
    
    def _binary_to_magnetization(self, binary_state: int) -> np.ndarray:
        """Convert binary quantum state to magnetization vector."""
        bits = [(binary_state >> i) & 1 for i in range(self.num_qubits)]
        
        # Map to spherical coordinates
        theta = sum(bits[:self.num_qubits//2]) * np.pi / (self.num_qubits//2)
        phi = sum(bits[self.num_qubits//2:]) * 2*np.pi / (self.num_qubits//2)
        
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
    
    def _compute_state_energy(
        self,
        state: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
        current: float
    ) -> float:
        """Compute energy for a given magnetization state."""
        # Distance from target (minimize)
        target_penalty = -np.dot(state, target)
        
        # Energy barrier consideration
        barrier_height = self.device.compute_energy_barrier(initial, target)
        current_efficiency = current / self.device.critical_current if hasattr(
            self.device, 'critical_current'
        ) else 1.0
        
        switching_cost = barrier_height * np.exp(-current_efficiency)
        
        return target_penalty + 0.1 * switching_cost
    
    def _compute_tunneling_amplitude(
        self, 
        state_i: int, 
        state_j: int, 
        current: float
    ) -> complex:
        """Compute quantum tunneling amplitude between adjacent states."""
        base_tunneling = 0.01  # Base tunneling strength
        current_enhancement = 1.0 + current / 1e6  # Current enhances tunneling
        
        return base_tunneling * current_enhancement * np.exp(
            1j * np.random.uniform(0, 2*np.pi)
        )
    
    def _hamming_distance(self, a: int, b: int) -> int:
        """Compute Hamming distance between binary representations."""
        return bin(a ^ b).count('1')
    
    def optimize_switching_protocol(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        max_current: float = 2e6,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """Find optimal switching protocol using quantum annealing."""
        with PerformanceProfiler("quantum_optimization"):
            H = self.compute_switching_hamiltonian(
                initial_state, target_state, max_current
            )
            
            if self.use_quantum_annealing:
                return self._quantum_annealing_optimization(
                    H, initial_state, target_state, num_steps
                )
            else:
                return self._variational_quantum_optimization(
                    H, initial_state, target_state, num_steps
                )
    
    def _quantum_annealing_optimization(
        self,
        hamiltonian: np.ndarray,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        num_steps: int
    ) -> Dict[str, Any]:
        """Perform quantum annealing to find optimal switching path."""
        state = self.quantum_state.copy()
        energies = []
        
        # Temperature schedule
        temp_initial = self.temperature_schedule['initial']
        temp_final = self.temperature_schedule['final']
        
        best_energy = np.inf
        best_protocol = None
        
        for step in range(num_steps):
            # Linear temperature reduction
            temperature = temp_initial * (temp_final/temp_initial)**(step/num_steps)
            
            # Time evolution under Hamiltonian
            if JAX_AVAILABLE:
                dt = 0.01 / temperature
                evolution = jax.scipy.linalg.expm(-1j * hamiltonian * dt)
                state = evolution @ state
            else:
                from scipy.linalg import expm
                dt = 0.01 / temperature
                evolution = expm(-1j * hamiltonian * dt)
                state = evolution @ state
            
            # Measure energy expectation
            energy = np.real(np.conj(state).T @ hamiltonian @ state)
            energies.append(energy)
            
            # Extract protocol from quantum state
            if energy < best_energy:
                best_energy = energy
                best_protocol = self._extract_protocol_from_state(
                    state, initial_state, target_state
                )
        
        return {
            'protocol': best_protocol,
            'final_energy': best_energy,
            'energy_history': energies,
            'quantum_state': state,
            'success_probability': self._compute_success_probability(state, target_state)
        }
    
    def _variational_quantum_optimization(
        self,
        hamiltonian: np.ndarray,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        num_steps: int
    ) -> Dict[str, Any]:
        """Variational quantum eigensolver approach."""
        # Parameterized quantum circuit
        params = np.random.uniform(0, 2*np.pi, self.num_qubits * 3)  # 3 params per qubit
        
        def cost_function(parameters):
            ansatz_state = self._apply_variational_ansatz(parameters)
            return np.real(np.conj(ansatz_state).T @ hamiltonian @ ansatz_state)
        
        # Classical optimization of quantum parameters
        from scipy.optimize import minimize
        
        result = minimize(
            cost_function,
            params,
            method='COBYLA',
            options={'maxiter': num_steps}
        )
        
        optimal_state = self._apply_variational_ansatz(result.x)
        protocol = self._extract_protocol_from_state(
            optimal_state, initial_state, target_state
        )
        
        return {
            'protocol': protocol,
            'final_energy': result.fun,
            'optimal_parameters': result.x,
            'quantum_state': optimal_state,
            'success_probability': self._compute_success_probability(
                optimal_state, target_state
            )
        }
    
    def _apply_variational_ansatz(self, parameters: np.ndarray) -> np.ndarray:
        """Apply parameterized quantum circuit."""
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0  # Start in |0...0⟩
        
        param_idx = 0
        for qubit in range(self.num_qubits):
            # RY rotation
            theta = parameters[param_idx]
            param_idx += 1
            
            # RZ rotation
            phi = parameters[param_idx]
            param_idx += 1
            
            # Entangling rotation
            gamma = parameters[param_idx]
            param_idx += 1
            
            # Apply rotations (simplified single-qubit operations)
            rotation_matrix = self._get_rotation_matrix(theta, phi, gamma)
            state = self._apply_single_qubit_gate(state, rotation_matrix, qubit)
        
        return state / np.linalg.norm(state)
    
    def _get_rotation_matrix(self, theta: float, phi: float, gamma: float) -> np.ndarray:
        """Generate rotation matrix for variational ansatz."""
        # Simplified rotation combining RY and RZ
        return np.array([
            [np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2)],
            [np.exp(1j*gamma)*np.sin(theta/2), np.exp(1j*(phi+gamma))*np.cos(theta/2)]
        ], dtype=complex)
    
    def _apply_single_qubit_gate(
        self, 
        state: np.ndarray, 
        gate: np.ndarray, 
        qubit: int
    ) -> np.ndarray:
        """Apply single-qubit gate to quantum state."""
        # Simplified implementation - in practice would use tensor products
        new_state = state.copy()
        
        # Apply gate effect through amplitude modification
        for i in range(len(state)):
            if (i >> qubit) & 1:  # If qubit is |1⟩
                new_state[i] *= gate[1, 1]
            else:  # If qubit is |0⟩
                new_state[i] *= gate[0, 0]
        
        return new_state
    
    def _extract_protocol_from_state(
        self,
        quantum_state: np.ndarray,
        initial_state: np.ndarray,
        target_state: np.ndarray
    ) -> List[Dict[str, float]]:
        """Extract classical switching protocol from quantum state."""
        # Find most probable computational basis state
        probabilities = np.abs(quantum_state)**2
        most_probable = np.argmax(probabilities)
        
        # Decode to magnetization trajectory
        optimal_magnetization = self._binary_to_magnetization(most_probable)
        
        # Generate pulse sequence to reach this state
        protocol = []
        
        # Simple protocol generation - interpolate between initial and optimal
        num_pulses = 5
        for i in range(num_pulses):
            t = (i + 1) / num_pulses
            intermediate_m = (1 - t) * initial_state + t * optimal_magnetization
            
            # Determine required current for this transition
            required_current = self._estimate_required_current(
                initial_state if i == 0 else protocol[-1]['target_magnetization'],
                intermediate_m
            )
            
            protocol.append({
                'current': required_current,
                'duration': 1e-9,  # 1 ns pulse
                'target_magnetization': intermediate_m.copy()
            })
        
        return protocol
    
    def _estimate_required_current(
        self, 
        initial_m: np.ndarray, 
        target_m: np.ndarray
    ) -> float:
        """Estimate current density required for magnetization transition."""
        # Simplified model based on switching angle
        angle = np.arccos(np.clip(np.dot(initial_m, target_m), -1, 1))
        
        # Critical current density proportional to switching angle
        if hasattr(self.device, 'critical_current'):
            return self.device.critical_current * (angle / np.pi)
        else:
            return 1e6 * (angle / np.pi)  # Default scaling
    
    def _compute_success_probability(
        self, 
        quantum_state: np.ndarray, 
        target_state: np.ndarray
    ) -> float:
        """Compute probability of successful switching to target state."""
        # Find basis states that correspond to target-like magnetizations
        success_probability = 0.0
        
        for i, amplitude in enumerate(quantum_state):
            state_magnetization = self._binary_to_magnetization(i)
            overlap = np.dot(state_magnetization, target_state)
            
            if overlap > 0.9:  # States close to target
                success_probability += np.abs(amplitude)**2
        
        return success_probability


class QuantumNeuralNetwork:
    """Quantum Neural Network for spintronic pattern recognition and optimization.
    
    Implements parameterized quantum circuits as neural network layers,
    providing quantum advantage for high-dimensional spintronic state classification
    and optimization tasks with exponential speedup potential.
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 4, ansatz_type: str = "strongly_entangling"):
        """Initialize quantum neural network.
        
        Args:
            num_qubits: Number of qubits in the quantum circuit
            num_layers: Number of parameterized layers
            ansatz_type: Type of variational ansatz to use
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_type = ansatz_type
        
        # Calculate parameter count
        if ansatz_type == "strongly_entangling":
            self.num_params = num_layers * num_qubits * 3  # 3 rotations per qubit per layer
        elif ansatz_type == "hardware_efficient":
            self.num_params = num_layers * num_qubits * 2  # 2 rotations per qubit per layer
        else:
            self.num_params = num_layers * num_qubits * 3
        
        # Initialize parameters
        self.parameters = np.random.uniform(0, 2*np.pi, self.num_params)
        
        # Performance metrics
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'quantum_fidelity': [],
            'gradient_norms': []
        }
        
        logger.info(f"Initialized QNN: {num_qubits} qubits, {num_layers} layers, {self.num_params} parameters")
    
    def create_feature_map(self, classical_data: np.ndarray) -> np.ndarray:
        """Create quantum feature map from classical data."""
        # Initialize quantum state in superposition
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0  # Start in |0...0⟩
        
        # Apply Hadamard gates for superposition
        for qubit in range(self.num_qubits):
            state = self._apply_hadamard(state, qubit)
        
        # Encode classical data through rotation angles
        for i, data_point in enumerate(classical_data[:self.num_qubits]):
            # Z-rotation encoding
            angle = np.pi * data_point  # Scale to [0, π]
            state = self._apply_rz(state, angle, i)
        
        return state
    
    def forward(self, classical_input: np.ndarray) -> np.ndarray:
        """Forward pass through quantum neural network."""
        # Create quantum feature map
        state = self.create_feature_map(classical_input)
        
        # Apply variational layers
        param_idx = 0
        for layer in range(self.num_layers):
            if self.ansatz_type == "strongly_entangling":
                layer_param_count = self.num_qubits * 3
            else:
                layer_param_count = self.num_qubits * 2
            
            layer_params = self.parameters[param_idx:param_idx + layer_param_count]
            state = self._apply_variational_layer(state, layer_params)
            param_idx += layer_param_count
        
        # Measurement in computational basis
        probabilities = np.abs(state)**2
        
        # Extract classical features from quantum measurements
        classical_output = self._extract_classical_features(probabilities)
        
        return classical_output
    
    def _apply_variational_layer(self, state: np.ndarray, layer_params: np.ndarray) -> np.ndarray:
        """Apply parameterized variational layer."""
        if self.ansatz_type == "strongly_entangling":
            return self._strongly_entangling_layer(state, layer_params)
        else:
            return self._hardware_efficient_layer(state, layer_params)
    
    def _strongly_entangling_layer(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply strongly entangling ansatz layer."""
        current_state = state.copy()
        
        # Apply single-qubit rotations
        for qubit in range(self.num_qubits):
            # RX, RY, RZ rotations
            rx_angle = params[qubit * 3]
            ry_angle = params[qubit * 3 + 1]
            rz_angle = params[qubit * 3 + 2]
            
            current_state = self._apply_rx(current_state, rx_angle, qubit)
            current_state = self._apply_ry(current_state, ry_angle, qubit)
            current_state = self._apply_rz(current_state, rz_angle, qubit)
        
        # Apply entangling gates (CNOT ladder)
        for qubit in range(self.num_qubits - 1):
            current_state = self._apply_cnot(current_state, qubit, qubit + 1)
        
        return current_state
    
    def _hardware_efficient_layer(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply hardware-efficient ansatz layer."""
        current_state = state.copy()
        
        # Apply RY and RZ rotations
        for qubit in range(self.num_qubits):
            ry_angle = params[qubit * 2]
            rz_angle = params[qubit * 2 + 1]
            
            current_state = self._apply_ry(current_state, ry_angle, qubit)
            current_state = self._apply_rz(current_state, rz_angle, qubit)
        
        # Linear entanglement
        for qubit in range(0, self.num_qubits - 1, 2):
            current_state = self._apply_cnot(current_state, qubit, qubit + 1)
        
        return current_state
    
    def _extract_classical_features(self, probabilities: np.ndarray) -> np.ndarray:
        """Extract classical features from quantum measurement probabilities."""
        features = []
        
        # Marginal probabilities for each qubit
        for qubit in range(self.num_qubits):
            prob_0 = sum(prob for i, prob in enumerate(probabilities) if not (i >> qubit) & 1)
            features.append(prob_0)
        
        # Global measures
        features.append(np.sum(probabilities * np.arange(len(probabilities))))  # Expectation value
        features.append(-np.sum(probabilities * np.log(probabilities + 1e-10)))  # Entropy
        
        return np.array(features)
    
    # Quantum gate implementations (simplified)
    def _apply_hadamard(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Hadamard gate to specified qubit."""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> qubit) & 1:  # Qubit is |1⟩
                partner = i ^ (1 << qubit)
                new_state[i] = (state[i] - state[partner]) / np.sqrt(2)
            else:  # Qubit is |0⟩
                partner = i ^ (1 << qubit)
                new_state[i] = (state[i] + state[partner]) / np.sqrt(2)
        return new_state
    
    def _apply_rx(self, state: np.ndarray, angle: float, qubit: int) -> np.ndarray:
        """Apply RX rotation gate."""
        cos_half = np.cos(angle / 2)
        sin_half = -1j * np.sin(angle / 2)
        
        new_state = state.copy()
        for i in range(len(state)):
            partner = i ^ (1 << qubit)
            if (i >> qubit) & 1:
                new_state[i] = cos_half * state[i] + sin_half * state[partner]
            else:
                new_state[i] = cos_half * state[i] + sin_half * state[partner]
        return new_state
    
    def _apply_ry(self, state: np.ndarray, angle: float, qubit: int) -> np.ndarray:
        """Apply RY rotation gate."""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        new_state = state.copy()
        for i in range(len(state)):
            partner = i ^ (1 << qubit)
            if (i >> qubit) & 1:
                new_state[i] = cos_half * state[i] - sin_half * state[partner]
            else:
                new_state[i] = cos_half * state[i] + sin_half * state[partner]
        return new_state
    
    def _apply_rz(self, state: np.ndarray, angle: float, qubit: int) -> np.ndarray:
        """Apply RZ rotation gate."""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> qubit) & 1:
                new_state[i] *= np.exp(1j * angle / 2)
            else:
                new_state[i] *= np.exp(-1j * angle / 2)
        return new_state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> control) & 1:  # Control is |1⟩
                partner = i ^ (1 << target)
                new_state[i] = state[partner]
        return new_state


class VariationalQuantumClassifier:
    """Variational Quantum Classifier for spintronic state classification."""
    
    def __init__(self, num_qubits: int, num_classes: int, num_layers: int = 3):
        """Initialize variational quantum classifier."""
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Initialize quantum neural network
        self.qnn = QuantumNeuralNetwork(num_qubits, num_layers, "hardware_efficient")
        
        # Classification parameters
        self.classification_weights = np.random.normal(0, 0.1, (num_classes, self.qnn.num_qubits + 2))
        
        logger.info(f"Initialized VQC: {num_qubits} qubits, {num_classes} classes")
    
    def classify(self, input_data: np.ndarray) -> np.ndarray:
        """Classify input using variational quantum circuit."""
        # Get quantum features
        quantum_features = self.qnn.forward(input_data)
        
        # Linear classification layer
        class_scores = np.dot(self.classification_weights, quantum_features)
        
        # Softmax activation
        exp_scores = np.exp(class_scores - np.max(class_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities


class QuantumReinforcementLearning:
    """Hybrid quantum-classical RL for spintronic device control."""
    
    def __init__(
        self,
        quantum_optimizer: QuantumSpinOptimizer,
        classical_policy_size: int = 64,
        quantum_feature_map_depth: int = 3
    ):
        self.quantum_optimizer = quantum_optimizer
        self.classical_policy_size = classical_policy_size
        self.quantum_feature_map_depth = quantum_feature_map_depth
        
        # Initialize hybrid parameters
        self.classical_params = np.random.normal(0, 0.1, classical_policy_size)
        self.quantum_params = np.random.uniform(
            0, 2*np.pi, 
            quantum_optimizer.num_qubits * quantum_feature_map_depth
        )
        
        # Performance tracking
        self.episode_rewards = []
        self.quantum_advantage_scores = []
    
    def quantum_feature_map(self, observation: np.ndarray) -> np.ndarray:
        """Map classical observation to quantum feature space."""
        # Normalize observation
        obs_norm = observation / (np.linalg.norm(observation) + 1e-8)
        
        # Create quantum feature state
        feature_state = self.quantum_optimizer.quantum_state.copy()
        
        # Encode observation through rotations
        param_idx = 0
        for depth in range(self.quantum_feature_map_depth):
            for i, obs_val in enumerate(obs_norm[:self.quantum_optimizer.num_qubits]):
                if param_idx < len(self.quantum_params):
                    # Encode observation value
                    angle = self.quantum_params[param_idx] * obs_val
                    param_idx += 1
                    
                    # Apply encoding rotation (simplified)
                    feature_state = self._apply_feature_encoding(
                        feature_state, angle, i
                    )
        
        return feature_state
    
    def _apply_feature_encoding(
        self, 
        state: np.ndarray, 
        angle: float, 
        qubit: int
    ) -> np.ndarray:
        """Apply feature encoding rotation to quantum state."""
        # Simplified feature encoding through amplitude modulation
        new_state = state.copy()
        
        encoding_factor = np.cos(angle) + 1j * np.sin(angle)
        
        for i in range(len(state)):
            if (i >> qubit) & 1:
                new_state[i] *= encoding_factor
        
        return new_state / np.linalg.norm(new_state)
    
    def hybrid_policy(
        self, 
        observation: np.ndarray, 
        target_state: np.ndarray
    ) -> Dict[str, float]:
        """Generate action using hybrid quantum-classical policy."""
        # Quantum feature extraction
        quantum_features = self.quantum_feature_map(observation[:3])  # Magnetization
        
        # Extract classical features from quantum state
        classical_features = self._quantum_to_classical_features(quantum_features)
        
        # Classical policy network (simplified)
        hidden = np.tanh(
            classical_features[:len(self.classical_params)//2] * 
            self.classical_params[:len(self.classical_params)//2]
        )
        
        output = np.tanh(
            hidden[:len(self.classical_params)//2] * 
            self.classical_params[len(self.classical_params)//2:]
        )
        
        # Generate action
        current_magnitude = output[0] * 2e6  # Scale to max current
        pulse_duration = (output[1] + 1) * 2.5e-9  # Scale to 0-5ns
        
        return {
            'current': current_magnitude,
            'duration': pulse_duration,
            'quantum_confidence': self._compute_quantum_confidence(quantum_features)
        }
    
    def _quantum_to_classical_features(self, quantum_state: np.ndarray) -> np.ndarray:
        """Extract classical features from quantum state."""
        # Measurement-based feature extraction
        features = []
        
        # Amplitude-based features
        features.extend(np.real(quantum_state)[:8])  # Real parts
        features.extend(np.imag(quantum_state)[:8])  # Imaginary parts
        
        # Entanglement-based features
        for i in range(min(4, len(quantum_state)//2)):
            entanglement_measure = np.abs(quantum_state[i] * quantum_state[-(i+1)])
            features.append(entanglement_measure)
        
        return np.array(features)
    
    def _compute_quantum_confidence(self, quantum_state: np.ndarray) -> float:
        """Compute confidence score based on quantum state coherence."""
        # Purity as confidence measure
        density_matrix = np.outer(quantum_state, np.conj(quantum_state))
        purity = np.real(np.trace(density_matrix @ density_matrix))
        
        return purity
    
    def update_policy(
        self, 
        experience_batch: List[Dict[str, Any]],
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """Update hybrid policy parameters based on experience."""
        # Compute gradients (simplified policy gradient)
        classical_grad = np.zeros_like(self.classical_params)
        quantum_grad = np.zeros_like(self.quantum_params)
        
        total_return = 0
        quantum_advantage = 0
        
        for experience in experience_batch:
            reward = experience['reward']
            observation = experience['observation']
            action = experience['action']
            
            total_return += reward
            
            # Estimate quantum advantage
            quantum_features = self.quantum_feature_map(observation[:3])
            classical_baseline = np.mean(observation[:3])  # Simple baseline
            quantum_baseline = np.mean(np.abs(quantum_features)**2)
            
            advantage = quantum_baseline - classical_baseline
            quantum_advantage += advantage
            
            # Simplified gradient computation
            feature_gradient = self._compute_feature_gradient(
                observation, reward, action
            )
            
            classical_grad += feature_gradient[:len(self.classical_params)]
            if len(feature_gradient) > len(self.classical_params):
                quantum_grad += feature_gradient[len(self.classical_params):]
        
        # Update parameters
        self.classical_params += learning_rate * classical_grad
        self.quantum_params += learning_rate * quantum_grad
        
        # Track performance
        avg_return = total_return / len(experience_batch)
        avg_quantum_advantage = quantum_advantage / len(experience_batch)
        
        self.episode_rewards.append(avg_return)
        self.quantum_advantage_scores.append(avg_quantum_advantage)
        
        return {
            'average_return': avg_return,
            'quantum_advantage': avg_quantum_advantage,
            'policy_norm': np.linalg.norm(self.classical_params),
            'quantum_param_norm': np.linalg.norm(self.quantum_params)
        }
    
    def _compute_feature_gradient(
        self, 
        observation: np.ndarray, 
        reward: float, 
        action: Dict[str, float]
    ) -> np.ndarray:
        """Compute gradient of features with respect to parameters."""
        # Simplified gradient computation
        gradient = np.zeros(len(self.classical_params) + len(self.quantum_params))
        
        # Classical gradient (policy gradient estimation)
        quantum_features = self.quantum_feature_map(observation[:3])
        classical_features = self._quantum_to_classical_features(quantum_features)
        
        for i in range(len(self.classical_params)):
            if i < len(classical_features):
                gradient[i] = reward * classical_features[i] * 0.01  # Scale factor
        
        # Quantum gradient (parameter shift rule approximation)
        for i in range(len(self.quantum_params)):
            param_shift = np.pi / 2
            
            # Positive shift
            self.quantum_params[i] += param_shift
            features_plus = self._quantum_to_classical_features(
                self.quantum_feature_map(observation[:3])
            )
            
            # Negative shift
            self.quantum_params[i] -= 2 * param_shift
            features_minus = self._quantum_to_classical_features(
                self.quantum_feature_map(observation[:3])
            )
            
            # Restore parameter
            self.quantum_params[i] += param_shift
            
            # Gradient via parameter shift rule
            gradient[len(self.classical_params) + i] = reward * 0.5 * (
                np.mean(features_plus) - np.mean(features_minus)
            )
        
        return gradient


def create_quantum_enhanced_environment(
    device_type: str = 'stt_mram',
    enable_quantum_optimization: bool = True,
    quantum_params: Optional[Dict] = None
) -> Tuple[Any, QuantumReinforcementLearning]:
    """Factory function to create quantum-enhanced RL environment."""
    from ..envs import SpinTorqueEnv
    from ..devices import DeviceFactory
    
    # Create base environment
    device = DeviceFactory.create(device_type)
    env = SpinTorqueEnv(device_type=device_type)
    
    # Create quantum components
    if enable_quantum_optimization:
        quantum_params = quantum_params or {
            'num_qubits': 8,
            'use_quantum_annealing': True,
            'temperature_schedule': {
                'initial': 50.0,
                'final': 0.1,
                'steps': 500
            }
        }
        
        quantum_optimizer = QuantumSpinOptimizer(device, **quantum_params)
        quantum_rl = QuantumReinforcementLearning(
            quantum_optimizer,
            classical_policy_size=64,
            quantum_feature_map_depth=3
        )
        
        return env, quantum_rl
    else:
        return env, None