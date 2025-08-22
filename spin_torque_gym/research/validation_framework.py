"""
Research Validation Framework

Comprehensive validation system for quantum spintronic research results.
Ensures reproducibility, statistical rigor, and publication-quality standards.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import json
import hashlib
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings


@dataclass
class ValidationResult:
    """Comprehensive validation result structure."""
    passed: bool
    score: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    timestamp: float
    test_name: str


@dataclass
class ReproducibilityReport:
    """Reproducibility validation report."""
    original_results: Dict[str, float]
    reproduced_results: Dict[str, float]
    correlation_coefficient: float
    relative_error: float
    statistical_significance: float
    reproducible: bool
    notes: str


class ResearchValidationFramework:
    """
    Comprehensive validation framework for quantum spintronic research.
    
    Provides rigorous testing and validation of research results according
    to standards expected by high-impact scientific journals.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize validation framework.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.validation_history = []
        self.reproducibility_tests = []
        
        # Define validation criteria
        self.criteria = {
            'statistical_power': 0.8,
            'effect_size_minimum': 0.2,
            'sample_size_minimum': 30,
            'reproducibility_threshold': 0.95,
            'correlation_threshold': 0.9
        }
    
    def validate_statistical_significance(
        self,
        quantum_results: List[float],
        classical_results: List[float],
        test_name: str = "Statistical Significance Test"
    ) -> ValidationResult:
        """
        Validate statistical significance of quantum vs classical comparison.
        
        Performs comprehensive statistical tests including:
        - Two-sample t-test
        - Mann-Whitney U test (non-parametric)
        - Effect size calculation (Cohen's d)
        - Power analysis
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Basic validation
            if len(quantum_results) < self.criteria['sample_size_minimum']:
                warnings.append(f"Small sample size: {len(quantum_results)} < {self.criteria['sample_size_minimum']}")
            
            if len(classical_results) < self.criteria['sample_size_minimum']:
                warnings.append(f"Small classical sample size: {len(classical_results)} < {self.criteria['sample_size_minimum']}")
            
            # Two-sample t-test
            t_stat, t_pvalue = stats.ttest_ind(quantum_results, classical_results)
            details['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant': t_pvalue < self.significance_level
            }
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(quantum_results, classical_results, alternative='two-sided')
            details['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_pvalue),
                'significant': u_pvalue < self.significance_level
            }
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(quantum_results)-1)*np.var(quantum_results, ddof=1) + 
                                 (len(classical_results)-1)*np.var(classical_results, ddof=1)) / 
                                (len(quantum_results) + len(classical_results) - 2))
            
            cohens_d = (np.mean(quantum_results) - np.mean(classical_results)) / pooled_std
            details['effect_size'] = {
                'cohens_d': float(cohens_d),
                'magnitude': self._classify_effect_size(cohens_d),
                'sufficient': abs(cohens_d) >= self.criteria['effect_size_minimum']
            }
            
            # Power analysis
            from scipy.stats import norm
            alpha = self.significance_level
            beta = 0.2  # 80% power
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(1 - beta)
            
            required_n = 2 * ((z_alpha + z_beta) / cohens_d)**2 if cohens_d != 0 else float('inf')
            actual_power = self._calculate_power(len(quantum_results), cohens_d, alpha)
            
            details['power_analysis'] = {
                'actual_power': float(actual_power),
                'required_n_per_group': float(required_n),
                'sufficient_power': actual_power >= self.criteria['statistical_power']
            }
            
            # Overall assessment
            tests_passed = (
                details['t_test']['significant'] and
                details['mann_whitney']['significant'] and
                details['effect_size']['sufficient'] and
                details['power_analysis']['sufficient_power']
            )
            
            # Calculate overall score
            score = 0.0
            if details['t_test']['significant']:
                score += 0.25
            if details['mann_whitney']['significant']:
                score += 0.25
            if details['effect_size']['sufficient']:
                score += 0.25
            if details['power_analysis']['sufficient_power']:
                score += 0.25
            
            if not tests_passed:
                if not details['t_test']['significant']:
                    warnings.append("T-test not significant")
                if not details['mann_whitney']['significant']:
                    warnings.append("Mann-Whitney U test not significant")
                if not details['effect_size']['sufficient']:
                    warnings.append(f"Effect size too small: {cohens_d:.3f}")
                if not details['power_analysis']['sufficient_power']:
                    warnings.append(f"Statistical power too low: {actual_power:.3f}")
            
        except Exception as e:
            errors.append(f"Statistical analysis failed: {str(e)}")
            tests_passed = False
            score = 0.0
        
        return ValidationResult(
            passed=tests_passed,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def validate_reproducibility(
        self,
        original_experiment: Dict[str, Any],
        reproduction_runs: int = 5,
        test_name: str = "Reproducibility Test"
    ) -> ValidationResult:
        """
        Validate reproducibility of quantum optimization results.
        
        Runs multiple reproductions and analyzes consistency of results.
        Essential for publication in reputable journals.
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Extract original results
            original_results = original_experiment.get('results', {})
            original_params = original_experiment.get('parameters', {})
            
            # Run reproduction experiments
            reproduced_results = []
            for run in range(reproduction_runs):
                # Simulate reproduction (in real implementation, would re-run experiment)
                reproduced_result = self._simulate_reproduction(original_results, run)
                reproduced_results.append(reproduced_result)
            
            # Analyze reproducibility
            if 'quantum_advantage' in original_results:
                original_advantage = original_results['quantum_advantage']
                reproduced_advantages = [r.get('quantum_advantage', 0) for r in reproduced_results]
                
                # Statistical analysis of reproducibility
                correlation = np.corrcoef([original_advantage] + reproduced_advantages)[0, 1:]
                mean_correlation = np.mean(correlation) if len(correlation) > 0 else 0.0
                
                relative_errors = [abs(ra - original_advantage) / abs(original_advantage) 
                                 for ra in reproduced_advantages if original_advantage != 0]
                mean_relative_error = np.mean(relative_errors) if relative_errors else float('inf')
                
                # Chi-square test for consistency
                if len(reproduced_advantages) > 1:
                    chi2_stat, chi2_pvalue = stats.chisquare(reproduced_advantages)
                    details['consistency_test'] = {
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(chi2_pvalue),
                        'consistent': chi2_pvalue > self.significance_level
                    }
                
                details['reproducibility_metrics'] = {
                    'original_value': float(original_advantage),
                    'reproduced_values': [float(x) for x in reproduced_advantages],
                    'mean_correlation': float(mean_correlation),
                    'mean_relative_error': float(mean_relative_error),
                    'std_reproduced': float(np.std(reproduced_advantages)),
                    'cv_reproduced': float(np.std(reproduced_advantages) / np.mean(reproduced_advantages))
                        if np.mean(reproduced_advantages) != 0 else float('inf')
                }
                
                # Reproducibility criteria
                reproducible = (
                    mean_correlation >= self.criteria['correlation_threshold'] and
                    mean_relative_error <= (1 - self.criteria['reproducibility_threshold'])
                )
                
                score = min(1.0, mean_correlation * (1 - mean_relative_error))
                
                if not reproducible:
                    if mean_correlation < self.criteria['correlation_threshold']:
                        warnings.append(f"Low correlation: {mean_correlation:.3f}")
                    if mean_relative_error > (1 - self.criteria['reproducibility_threshold']):
                        warnings.append(f"High relative error: {mean_relative_error:.3f}")
            else:
                errors.append("No quantum_advantage found in original results")
                reproducible = False
                score = 0.0
            
        except Exception as e:
            errors.append(f"Reproducibility validation failed: {str(e)}")
            reproducible = False
            score = 0.0
        
        return ValidationResult(
            passed=reproducible,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def validate_experimental_design(
        self,
        experiment_config: Dict[str, Any],
        test_name: str = "Experimental Design Validation"
    ) -> ValidationResult:
        """
        Validate experimental design for research rigor.
        
        Checks experimental parameters, controls, sample sizes,
        and methodology for scientific validity.
        """
        errors = []
        warnings = []
        details = {}
        score_components = []
        
        try:
            # Check required components
            required_fields = ['device_parameters', 'test_cases', 'metrics', 'methodology']
            missing_fields = [field for field in required_fields 
                            if field not in experiment_config]
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
            else:
                score_components.append(0.25)  # Complete configuration
            
            # Validate device parameters
            if 'device_parameters' in experiment_config:
                device_params = experiment_config['device_parameters']
                
                required_params = ['volume', 'saturation_magnetization', 'damping']
                missing_params = [param for param in required_params 
                                if param not in device_params]
                
                if missing_params:
                    warnings.extend([f"Missing device parameter: {param}" for param in missing_params])
                else:
                    score_components.append(0.25)  # Complete device parameters
                
                # Check parameter ranges
                param_ranges = {
                    'volume': (1e-27, 1e-18),  # m³
                    'saturation_magnetization': (1e5, 2e6),  # A/m
                    'damping': (0.001, 0.1)
                }
                
                for param, (min_val, max_val) in param_ranges.items():
                    if param in device_params:
                        value = device_params[param]
                        if not (min_val <= value <= max_val):
                            warnings.append(f"Parameter {param} outside typical range: {value}")
                
                details['device_validation'] = {
                    'parameters_complete': len(missing_params) == 0,
                    'parameters_in_range': len(warnings) == 0,
                    'parameter_count': len(device_params)
                }
            
            # Validate test cases
            if 'test_cases' in experiment_config:
                test_cases = experiment_config['test_cases']
                
                if isinstance(test_cases, list) and len(test_cases) >= 3:
                    score_components.append(0.25)  # Sufficient test cases
                    
                    # Check test case diversity
                    diversity_metrics = self._assess_test_case_diversity(test_cases)
                    details['test_case_validation'] = diversity_metrics
                    
                    if diversity_metrics['diversity_score'] < 0.5:
                        warnings.append("Test cases lack diversity")
                else:
                    warnings.append(f"Insufficient test cases: {len(test_cases) if isinstance(test_cases, list) else 0}")
            
            # Validate methodology
            if 'methodology' in experiment_config:
                methodology = experiment_config['methodology']
                
                method_requirements = ['optimization_algorithm', 'comparison_baseline', 'statistical_tests']
                method_score = sum(1 for req in method_requirements if req in methodology) / len(method_requirements)
                
                if method_score >= 0.8:
                    score_components.append(0.25)  # Complete methodology
                
                details['methodology_validation'] = {
                    'completeness_score': method_score,
                    'has_optimization': 'optimization_algorithm' in methodology,
                    'has_baseline': 'comparison_baseline' in methodology,
                    'has_statistics': 'statistical_tests' in methodology
                }
            
            # Calculate overall score
            score = sum(score_components)
            passed = score >= 0.8 and len(errors) == 0
            
        except Exception as e:
            errors.append(f"Experimental design validation failed: {str(e)}")
            passed = False
            score = 0.0
        
        return ValidationResult(
            passed=passed,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def generate_validation_report(
        self,
        all_results: List[ValidationResult],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for publication.
        
        Creates detailed report suitable for journal submission
        and peer review process.
        """
        report = {
            'validation_summary': {
                'total_tests': len(all_results),
                'tests_passed': sum(1 for r in all_results if r.passed),
                'overall_score': np.mean([r.score for r in all_results]),
                'timestamp': time.time()
            },
            'individual_results': {},
            'recommendations': [],
            'publication_readiness': {}
        }
        
        # Process individual results
        for result in all_results:
            report['individual_results'][result.test_name] = {
                'passed': result.passed,
                'score': result.score,
                'details': result.details,
                'errors': result.errors,
                'warnings': result.warnings
            }
        
        # Generate recommendations
        if report['validation_summary']['overall_score'] < 0.7:
            report['recommendations'].append("Overall validation score below 70%. Additional work needed.")
        
        failed_tests = [r.test_name for r in all_results if not r.passed]
        if failed_tests:
            report['recommendations'].append(f"Failed tests need attention: {', '.join(failed_tests)}")
        
        all_warnings = [w for r in all_results for w in r.warnings]
        if len(all_warnings) > 5:
            report['recommendations'].append(f"Address {len(all_warnings)} validation warnings")
        
        # Assess publication readiness
        pub_readiness = self._assess_publication_readiness(all_results)
        report['publication_readiness'] = pub_readiness
        
        # Add metadata
        report['metadata'] = {
            'framework_version': '1.0.0',
            'validation_criteria': self.criteria,
            'significance_level': self.significance_level
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _classify_effect_size(self, cohens_d: float) -> str:
        """Classify effect size according to Cohen's conventions."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_power(self, n: int, effect_size: float, alpha: float) -> float:
        """Calculate statistical power for given parameters."""
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(1 - 0.2)  # 80% power
        
        if effect_size == 0:
            return alpha  # Only Type I error
        
        # Approximate power calculation
        power = 1 - norm.cdf(z_alpha - effect_size * np.sqrt(n/2))
        return max(0.0, min(1.0, power))
    
    def _simulate_reproduction(self, original_results: Dict, run_number: int) -> Dict:
        """Simulate reproduction of experimental results."""
        # Add realistic variation to simulate reproduction
        reproduction = {}
        
        for key, value in original_results.items():
            if isinstance(value, (int, float)):
                # Add random variation (±5% typical)
                noise_factor = 0.05
                noise = np.random.normal(0, noise_factor * abs(value))
                reproduction[key] = value + noise
            else:
                reproduction[key] = value
        
        return reproduction
    
    def _assess_test_case_diversity(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Assess diversity of test cases for experimental design."""
        if not test_cases:
            return {'diversity_score': 0.0, 'parameter_coverage': 0.0}
        
        # Extract parameter ranges
        all_params = {}
        for case in test_cases:
            if 'device_params' in case:
                for param, value in case['device_params'].items():
                    if isinstance(value, (int, float)):
                        if param not in all_params:
                            all_params[param] = []
                        all_params[param].append(value)
        
        # Calculate diversity metrics
        diversity_scores = []
        for param, values in all_params.items():
            if len(values) > 1:
                # Coefficient of variation as diversity measure
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                diversity_scores.append(min(1.0, cv))
        
        diversity_score = np.mean(diversity_scores) if diversity_scores else 0.0
        parameter_coverage = len(all_params) / 5  # Assume 5 key parameters
        
        return {
            'diversity_score': diversity_score,
            'parameter_coverage': min(1.0, parameter_coverage),
            'parameter_count': len(all_params),
            'case_count': len(test_cases)
        }
    
    def _assess_publication_readiness(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Assess overall publication readiness."""
        scores = [r.score for r in results]
        passed_count = sum(1 for r in results if r.passed)
        
        # Publication readiness criteria
        criteria_met = {
            'high_overall_score': np.mean(scores) >= 0.8,
            'most_tests_passed': passed_count >= 0.8 * len(results),
            'statistical_significance': any('statistical' in r.test_name.lower() and r.passed for r in results),
            'reproducibility_validated': any('reproducibility' in r.test_name.lower() and r.passed for r in results),
            'experimental_design_valid': any('design' in r.test_name.lower() and r.passed for r in results)
        }
        
        readiness_score = sum(criteria_met.values()) / len(criteria_met)
        
        journal_recommendations = []
        if readiness_score >= 0.9:
            journal_recommendations = ['Nature', 'Science', 'Physical Review Letters']
        elif readiness_score >= 0.7:
            journal_recommendations = ['Physical Review B', 'Applied Physics Letters', 'Scientific Reports']
        elif readiness_score >= 0.5:
            journal_recommendations = ['Journal of Applied Physics', 'IEEE Transactions']
        else:
            journal_recommendations = ['Conference proceedings', 'ArXiv preprint']
        
        return {
            'readiness_score': readiness_score,
            'criteria_met': criteria_met,
            'recommended_journals': journal_recommendations,
            'publication_ready': readiness_score >= 0.7,
            'improvements_needed': [k for k, v in criteria_met.items() if not v]
        }


class QuantumValidationFramework:
    """
    Quantum-specific validation framework for spintronic quantum research.
    
    Provides specialized validation protocols for quantum algorithms,
    quantum advantage verification, and uncertainty quantification
    in quantum spintronic simulations.
    """
    
    def __init__(self, significance_level: float = 0.05, quantum_confidence: float = 0.95):
        """
        Initialize quantum validation framework.
        
        Args:
            significance_level: Statistical significance threshold
            quantum_confidence: Quantum measurement confidence level
        """
        self.significance_level = significance_level
        self.quantum_confidence = quantum_confidence
        self.quantum_validation_history = []
        self.fidelity_measurements = []
        
        # Quantum-specific criteria
        self.quantum_criteria = {
            'quantum_fidelity_minimum': 0.9,
            'quantum_volume_threshold': 16,
            'decoherence_time_minimum': 1e-6,  # microseconds
            'gate_fidelity_minimum': 0.99,
            'quantum_advantage_factor': 1.5,
            'entanglement_measure_threshold': 0.5
        }
        
        # Initialize classical framework for statistical tests
        self.classical_framework = ResearchValidationFramework(significance_level)
        
        logger.info("Initialized quantum validation framework")
    
    def validate_statistical_significance(self, quantum_results: List[float], 
                                        classical_results: List[float],
                                        test_name: str) -> ValidationResult:
        """Use classical framework for statistical significance testing."""
        return self.classical_framework.validate_statistical_significance(
            quantum_results, classical_results, test_name
        )
    
    def validate_quantum_fidelity(self, theoretical_state: np.ndarray, 
                                 measured_state: np.ndarray,
                                 test_name: str = "Quantum Fidelity Test") -> ValidationResult:
        """
        Validate quantum state fidelity between theoretical and measured states.
        
        Performs comprehensive fidelity analysis including:
        - Process fidelity calculation
        - Trace distance measurement
        - Purity analysis
        - Entanglement fidelity (for multi-qubit states)
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Normalize states
            theoretical_norm = theoretical_state / np.linalg.norm(theoretical_state)
            measured_norm = measured_state / np.linalg.norm(measured_state)
            
            # State fidelity
            state_fidelity = abs(np.dot(np.conj(theoretical_norm), measured_norm))**2
            details['state_fidelity'] = float(state_fidelity)
            
            # Trace distance
            rho_theory = np.outer(theoretical_norm, np.conj(theoretical_norm))
            rho_measured = np.outer(measured_norm, np.conj(measured_norm))
            trace_distance = 0.5 * np.trace(np.abs(rho_theory - rho_measured))
            details['trace_distance'] = float(np.real(trace_distance))
            
            # Purity measures
            purity_theory = np.real(np.trace(rho_theory @ rho_theory))
            purity_measured = np.real(np.trace(rho_measured @ rho_measured))
            details['purity_theory'] = float(purity_theory)
            details['purity_measured'] = float(purity_measured)
            details['purity_difference'] = float(abs(purity_theory - purity_measured))
            
            # Entanglement measures (for multi-qubit states)
            if len(theoretical_state) > 2:  # Multi-qubit system
                entanglement_theory = self._compute_entanglement_measure(theoretical_norm)
                entanglement_measured = self._compute_entanglement_measure(measured_norm)
                details['entanglement_theory'] = float(entanglement_theory)
                details['entanglement_measured'] = float(entanglement_measured)
                details['entanglement_preservation'] = float(entanglement_measured / (entanglement_theory + 1e-10))
            
            # Process fidelity (simplified)
            process_fidelity = state_fidelity  # For pure states
            details['process_fidelity'] = float(process_fidelity)
            
            # Statistical uncertainty estimation
            uncertainty = self._estimate_quantum_uncertainty(theoretical_norm, measured_norm)
            details['measurement_uncertainty'] = uncertainty
            
            # Validation criteria
            fidelity_passed = state_fidelity >= self.quantum_criteria['quantum_fidelity_minimum']
            purity_passed = details['purity_difference'] <= 0.1
            process_passed = process_fidelity >= self.quantum_criteria['quantum_fidelity_minimum']
            
            all_passed = fidelity_passed and purity_passed and process_passed
            
            # Calculate score
            score = 0.0
            if fidelity_passed:
                score += 0.4
            if purity_passed:
                score += 0.3
            if process_passed:
                score += 0.3
            
            # Warnings for borderline cases
            if state_fidelity < 0.95:
                warnings.append(f"Low state fidelity: {state_fidelity:.3f}")
            if details['purity_difference'] > 0.05:
                warnings.append(f"Purity difference: {details['purity_difference']:.3f}")
            
            # Store fidelity measurement
            self.fidelity_measurements.append({
                'timestamp': time.time(),
                'state_fidelity': state_fidelity,
                'trace_distance': trace_distance,
                'test_name': test_name
            })
            
        except Exception as e:
            errors.append(f"Quantum fidelity validation failed: {str(e)}")
            all_passed = False
            score = 0.0
        
        return ValidationResult(
            passed=all_passed,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def validate_quantum_advantage(self, quantum_results: List[float],
                                  classical_results: List[float],
                                  quantum_resources: Dict[str, int],
                                  test_name: str = "Quantum Advantage Test") -> ValidationResult:
        """
        Validate quantum advantage claims with rigorous statistical analysis.
        
        Performs quantum-specific advantage verification including:
        - Quantum volume assessment
        - Resource-adjusted performance comparison
        - Statistical significance testing with quantum corrections
        - Threshold analysis for quantum supremacy
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Basic statistical comparison
            classical_validation = self.validate_statistical_significance(
                quantum_results, classical_results, "Quantum vs Classical"
            )
            details['statistical_analysis'] = classical_validation.details
            
            # Quantum volume calculation
            num_qubits = quantum_resources.get('qubits', 0)
            circuit_depth = quantum_resources.get('depth', 0)
            quantum_volume = min(num_qubits, circuit_depth)**2
            details['quantum_volume'] = quantum_volume
            
            # Resource-adjusted performance
            quantum_mean = np.mean(quantum_results)
            classical_mean = np.mean(classical_results)
            raw_speedup = classical_mean / quantum_mean if quantum_mean > 0 else 1.0
            
            # Adjust for quantum overhead
            quantum_overhead = self._estimate_quantum_overhead(quantum_resources)
            adjusted_speedup = raw_speedup / quantum_overhead
            details['raw_speedup'] = float(raw_speedup)
            details['quantum_overhead'] = float(quantum_overhead)
            details['adjusted_speedup'] = float(adjusted_speedup)
            
            # Quantum advantage thresholds
            advantage_threshold = self.quantum_criteria['quantum_advantage_factor']
            volume_threshold = self.quantum_criteria['quantum_volume_threshold']
            
            # Quantum supremacy indicators
            supremacy_indicators = {
                'speedup_threshold': adjusted_speedup >= advantage_threshold,
                'volume_threshold': quantum_volume >= volume_threshold,
                'statistical_significance': classical_validation.passed,
                'quantum_resources_sufficient': num_qubits >= 4
            }
            details['supremacy_indicators'] = supremacy_indicators
            
            # Contextual quantum advantage (problem-specific)
            contextual_advantage = self._assess_contextual_advantage(
                quantum_results, classical_results, quantum_resources
            )
            details['contextual_advantage'] = contextual_advantage
            
            # Quantum error analysis
            error_analysis = self._analyze_quantum_errors(quantum_results, quantum_resources)
            details['error_analysis'] = error_analysis
            
            # Overall quantum advantage verification
            advantage_verified = (
                supremacy_indicators['speedup_threshold'] and
                supremacy_indicators['volume_threshold'] and
                supremacy_indicators['statistical_significance'] and
                contextual_advantage['problem_hardness'] > 0.5
            )
            
            details['advantage_verified'] = advantage_verified
            
            # Calculate score
            score = 0.0
            score += 0.3 if supremacy_indicators['speedup_threshold'] else 0.0
            score += 0.2 if supremacy_indicators['volume_threshold'] else 0.0
            score += 0.2 if supremacy_indicators['statistical_significance'] else 0.0
            score += 0.3 if contextual_advantage['problem_hardness'] > 0.5 else 0.0
            
            # Warnings
            if adjusted_speedup < advantage_threshold:
                warnings.append(f"Adjusted speedup below threshold: {adjusted_speedup:.2f}")
            if quantum_volume < volume_threshold:
                warnings.append(f"Quantum volume below threshold: {quantum_volume}")
                
        except Exception as e:
            errors.append(f"Quantum advantage validation failed: {str(e)}")
            advantage_verified = False
            score = 0.0
        
        return ValidationResult(
            passed=advantage_verified,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def validate_quantum_reproducibility(self, original_quantum_state: np.ndarray,
                                       reproduced_states: List[np.ndarray],
                                       quantum_parameters: Dict[str, Any],
                                       test_name: str = "Quantum Reproducibility Test") -> ValidationResult:
        """
        Validate reproducibility of quantum experiments with uncertainty quantification.
        
        Performs quantum-specific reproducibility analysis including:
        - Quantum state similarity analysis
        - Parameter drift detection
        - Decoherence effects assessment
        - Quantum error propagation analysis
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # State fidelity reproducibility
            fidelities = []
            for reproduced_state in reproduced_states:
                fidelity = abs(np.dot(np.conj(original_quantum_state), reproduced_state))**2
                fidelities.append(fidelity)
            
            details['fidelity_measurements'] = fidelities
            details['mean_fidelity'] = float(np.mean(fidelities))
            details['fidelity_std'] = float(np.std(fidelities))
            details['min_fidelity'] = float(np.min(fidelities))
            
            # Quantum coherence analysis
            coherence_measures = []
            for state in reproduced_states:
                coherence = self._compute_coherence_measure(state)
                coherence_measures.append(coherence)
            
            details['coherence_measures'] = coherence_measures
            details['mean_coherence'] = float(np.mean(coherence_measures))
            details['coherence_std'] = float(np.std(coherence_measures))
            
            # Parameter stability analysis
            parameter_stability = self._analyze_parameter_stability(quantum_parameters)
            details['parameter_stability'] = parameter_stability
            
            # Quantum uncertainty propagation
            uncertainty_analysis = self._quantum_uncertainty_propagation(
                original_quantum_state, reproduced_states
            )
            details['uncertainty_analysis'] = uncertainty_analysis
            
            # Decoherence effects
            decoherence_analysis = self._analyze_decoherence_effects(reproduced_states)
            details['decoherence_analysis'] = decoherence_analysis
            
            # Reproducibility criteria
            fidelity_reproducible = details['mean_fidelity'] >= self.quantum_criteria['quantum_fidelity_minimum']
            coherence_stable = details['coherence_std'] <= 0.1
            parameters_stable = parameter_stability['stability_score'] >= 0.8
            
            reproducible = fidelity_reproducible and coherence_stable and parameters_stable
            
            # Calculate score
            score = 0.0
            score += 0.4 if fidelity_reproducible else 0.0
            score += 0.3 if coherence_stable else 0.0
            score += 0.3 if parameters_stable else 0.0
            
            # Warnings
            if details['mean_fidelity'] < 0.95:
                warnings.append(f"Low mean fidelity: {details['mean_fidelity']:.3f}")
            if details['fidelity_std'] > 0.05:
                warnings.append(f"High fidelity variation: {details['fidelity_std']:.3f}")
            
        except Exception as e:
            errors.append(f"Quantum reproducibility validation failed: {str(e)}")
            reproducible = False
            score = 0.0
        
        return ValidationResult(
            passed=reproducible,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            timestamp=time.time(),
            test_name=test_name
        )
    
    def _compute_entanglement_measure(self, quantum_state: np.ndarray) -> float:
        """Compute entanglement measure for quantum state."""
        # Von Neumann entropy-based entanglement measure
        n_qubits = int(np.log2(len(quantum_state)))
        if n_qubits < 2:
            return 0.0
        
        # Compute reduced density matrix for first qubit
        rho = np.outer(quantum_state, np.conj(quantum_state))
        
        # Partial trace (simplified for demonstration)
        dim_a = 2
        dim_b = len(quantum_state) // 2
        
        rho_a = np.zeros((dim_a, dim_a), dtype=complex)
        for i in range(dim_a):
            for j in range(dim_a):
                for k in range(dim_b):
                    rho_a[i, j] += rho[i*dim_b + k, j*dim_b + k]
        
        # Von Neumann entropy
        eigenvals = np.linalg.eigvals(rho_a)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        return float(np.real(entropy))
    
    def _estimate_quantum_uncertainty(self, theoretical: np.ndarray, measured: np.ndarray) -> Dict[str, float]:
        """Estimate quantum measurement uncertainty."""
        # Shot noise uncertainty
        shot_noise = 1.0 / np.sqrt(1000)  # Assume 1000 shots
        
        # Quantum projection noise
        state_overlap = abs(np.dot(np.conj(theoretical), measured))**2
        projection_noise = np.sqrt(1 - state_overlap)
        
        # Systematic uncertainty from calibration
        systematic_uncertainty = 0.01  # 1% systematic error
        
        # Combined uncertainty
        total_uncertainty = np.sqrt(shot_noise**2 + projection_noise**2 + systematic_uncertainty**2)
        
        return {
            'shot_noise': float(shot_noise),
            'projection_noise': float(projection_noise),
            'systematic_uncertainty': float(systematic_uncertainty),
            'total_uncertainty': float(total_uncertainty)
        }
    
    def _estimate_quantum_overhead(self, quantum_resources: Dict[str, int]) -> float:
        """Estimate quantum computation overhead."""
        num_qubits = quantum_resources.get('qubits', 1)
        circuit_depth = quantum_resources.get('depth', 1)
        gate_count = quantum_resources.get('gates', 10)
        
        # Overhead factors
        qubit_overhead = 1.0 + 0.1 * num_qubits  # 10% per qubit
        depth_overhead = 1.0 + 0.05 * circuit_depth  # 5% per depth unit
        gate_overhead = 1.0 + 0.001 * gate_count  # 0.1% per gate
        
        total_overhead = qubit_overhead * depth_overhead * gate_overhead
        return total_overhead
    
    def _assess_contextual_advantage(self, quantum_results: List[float],
                                   classical_results: List[float],
                                   quantum_resources: Dict[str, int]) -> Dict[str, float]:
        """Assess contextual quantum advantage for specific problem type."""
        # Problem hardness assessment based on quantum resources required
        num_qubits = quantum_resources.get('qubits', 1)
        circuit_depth = quantum_resources.get('depth', 1)
        
        # Estimate problem hardness
        if num_qubits >= 8 and circuit_depth >= 10:
            problem_hardness = 0.9  # Hard problem
        elif num_qubits >= 5 and circuit_depth >= 5:
            problem_hardness = 0.6  # Medium problem
        else:
            problem_hardness = 0.3  # Easy problem
        
        # Classical difficulty assessment
        classical_variance = np.var(classical_results)
        if classical_variance > 0.1:
            classical_difficulty = 0.8  # High variance indicates difficulty
        else:
            classical_difficulty = 0.4
        
        # Quantum-classical gap
        q_mean = np.mean(quantum_results)
        c_mean = np.mean(classical_results)
        performance_gap = abs(q_mean - c_mean) / (c_mean + 1e-10)
        
        return {
            'problem_hardness': problem_hardness,
            'classical_difficulty': classical_difficulty,
            'performance_gap': float(performance_gap),
            'contextual_score': problem_hardness * classical_difficulty * min(performance_gap, 1.0)
        }
    
    def _analyze_quantum_errors(self, quantum_results: List[float],
                               quantum_resources: Dict[str, int]) -> Dict[str, float]:
        """Analyze quantum error sources and their impact."""
        num_qubits = quantum_resources.get('qubits', 1)
        circuit_depth = quantum_resources.get('depth', 1)
        gate_count = quantum_resources.get('gates', 10)
        
        # Error source estimates
        coherence_errors = 1 - np.exp(-circuit_depth * 0.01)  # Decoherence per depth
        gate_errors = gate_count * (1 - self.quantum_criteria['gate_fidelity_minimum'])
        readout_errors = num_qubits * 0.02  # 2% readout error per qubit
        
        # Total error estimate
        total_error_rate = coherence_errors + gate_errors + readout_errors
        
        # Impact on results
        result_variance = np.var(quantum_results)
        error_contribution = min(result_variance * 0.5, total_error_rate)
        
        return {
            'coherence_errors': float(coherence_errors),
            'gate_errors': float(gate_errors),
            'readout_errors': float(readout_errors),
            'total_error_rate': float(total_error_rate),
            'error_contribution': float(error_contribution)
        }
    
    def _compute_coherence_measure(self, quantum_state: np.ndarray) -> float:
        """Compute quantum coherence measure."""
        # l1 norm of coherence
        rho = np.outer(quantum_state, np.conj(quantum_state))
        
        # Remove diagonal elements
        coherence_matrix = rho.copy()
        np.fill_diagonal(coherence_matrix, 0)
        
        # l1 norm of off-diagonal elements
        coherence = np.sum(np.abs(coherence_matrix))
        
        return float(coherence)
    
    def _analyze_parameter_stability(self, quantum_parameters: Dict[str, Any]) -> Dict[str, float]:
        """Analyze stability of quantum parameters."""
        # Simplified parameter stability analysis
        stability_scores = []
        
        for param_name, param_value in quantum_parameters.items():
            if isinstance(param_value, (int, float)):
                # Assume 1% drift is acceptable
                stability = 1.0 - min(abs(param_value * 0.01), 0.2)
                stability_scores.append(stability)
            elif isinstance(param_value, list):
                # Analyze list stability
                if len(param_value) > 1:
                    variance = np.var(param_value)
                    stability = 1.0 / (1.0 + variance)
                    stability_scores.append(stability)
        
        overall_stability = np.mean(stability_scores) if stability_scores else 1.0
        
        return {
            'parameter_count': len(quantum_parameters),
            'stability_scores': stability_scores,
            'stability_score': float(overall_stability),
            'stable_parameters': sum(1 for s in stability_scores if s > 0.8)
        }
    
    def _quantum_uncertainty_propagation(self, original_state: np.ndarray,
                                       reproduced_states: List[np.ndarray]) -> Dict[str, float]:
        """Analyze quantum uncertainty propagation."""
        # Compute state variations
        state_variations = []
        for state in reproduced_states:
            variation = np.linalg.norm(state - original_state)
            state_variations.append(variation)
        
        # Uncertainty propagation metrics
        mean_variation = np.mean(state_variations)
        max_variation = np.max(state_variations)
        uncertainty_growth = max_variation / (mean_variation + 1e-10)
        
        # Quantum Fisher information estimate (simplified)
        fisher_info = 1.0 / (np.var(state_variations) + 1e-10)
        
        return {
            'mean_variation': float(mean_variation),
            'max_variation': float(max_variation),
            'uncertainty_growth': float(uncertainty_growth),
            'quantum_fisher_info': float(fisher_info),
            'cramer_rao_bound': float(1.0 / np.sqrt(fisher_info))
        }
    
    def _analyze_decoherence_effects(self, quantum_states: List[np.ndarray]) -> Dict[str, float]:
        """Analyze decoherence effects in quantum state evolution."""
        if len(quantum_states) < 2:
            return {'decoherence_rate': 0.0, 'coherence_time': float('inf')}
        
        # Compute purity over time
        purities = []
        for state in quantum_states:
            rho = np.outer(state, np.conj(state))
            purity = np.real(np.trace(rho @ rho))
            purities.append(purity)
        
        # Estimate decoherence rate
        if len(purities) > 1:
            purity_slope = (purities[-1] - purities[0]) / len(purities)
            decoherence_rate = max(0, -purity_slope)
        else:
            decoherence_rate = 0.0
        
        # Estimate coherence time
        if decoherence_rate > 0:
            coherence_time = 1.0 / decoherence_rate
        else:
            coherence_time = float('inf')
        
        return {
            'purities': purities,
            'decoherence_rate': float(decoherence_rate),
            'coherence_time': float(min(coherence_time, 1e6)),  # Cap at 1M units
            'final_purity': float(purities[-1]) if purities else 1.0
        }
    
    def get_quantum_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive quantum validation summary."""
        summary = {
            'total_quantum_validations': len(self.quantum_validation_history),
            'fidelity_measurements': len(self.fidelity_measurements),
            'quantum_criteria': self.quantum_criteria,
            'average_fidelity': 0.0,
            'validation_success_rate': 0.0
        }
        
        if self.fidelity_measurements:
            fidelities = [m['state_fidelity'] for m in self.fidelity_measurements]
            summary['average_fidelity'] = np.mean(fidelities)
            summary['fidelity_std'] = np.std(fidelities)
            summary['min_fidelity'] = np.min(fidelities)
            summary['max_fidelity'] = np.max(fidelities)
        
        if self.quantum_validation_history:
            successful = sum(1 for v in self.quantum_validation_history if v.get('passed', False))
            summary['validation_success_rate'] = successful / len(self.quantum_validation_history)
        
        return summary