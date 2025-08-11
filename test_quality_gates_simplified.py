#!/usr/bin/env python3
"""Simplified Quality Gates for Spin Torque RL-Gym without external dependencies.

This validates the core quality aspects of the system using only built-in Python.
"""

import ast
import json
import os
import sys
import time
import re
from collections import defaultdict
from pathlib import Path


class QualityGate:
    """Quality gate validation."""
    
    def __init__(self, name: str, critical: bool = True):
        self.name = name
        self.critical = critical
        self.passed = False
        self.score = 0.0
        self.message = ""
        self.details = {}
    
    def run(self):
        """Run the quality gate."""
        try:
            self.passed, self.score, self.message, self.details = self._execute()
        except Exception as e:
            self.passed = False
            self.score = 0.0
            self.message = f"Quality gate failed: {e}"
            self.details = {'error': str(e)}
        
        return self.passed, self.score, self.message, self.details
    
    def _execute(self):
        """Override in subclasses."""
        raise NotImplementedError


class CodeStructureGate(QualityGate):
    """Validate code structure and syntax."""
    
    def __init__(self):
        super().__init__("Code Structure", critical=True)
    
    def _execute(self):
        """Check code structure and syntax."""
        issues = []
        files_checked = 0
        
        # Check Python syntax
        for py_file in Path('.').rglob('*.py'):
            if py_file.is_file() and 'spin_torque_gym' in str(py_file):
                files_checked += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source)
                except SyntaxError as e:
                    issues.append(f"Syntax error in {py_file}: {e}")
                except Exception as e:
                    issues.append(f"Parse error in {py_file}: {e}")
        
        # Calculate score
        if issues:
            passed = False
            score = max(0.0, 1.0 - len(issues) / files_checked)
            message = f"{len(issues)} syntax issues in {files_checked} files"
        else:
            passed = True
            score = 1.0
            message = f"All {files_checked} files have valid syntax"
        
        details = {
            'files_checked': files_checked,
            'syntax_issues': len(issues),
            'issues': issues
        }
        
        return passed, score, message, details


class ImportGate(QualityGate):
    """Validate that core imports work."""
    
    def __init__(self):
        super().__init__("Core Imports", critical=True)
    
    def _execute(self):
        """Test core module imports."""
        import_tests = []
        
        # Test core module imports
        test_imports = [
            ('spin_torque_gym', 'Main package'),
            ('spin_torque_gym.devices.base_device', 'Base device'),
            ('spin_torque_gym.devices.stt_mram', 'STT-MRAM device'),
            ('spin_torque_gym.devices.sot_mram', 'SOT-MRAM device'), 
            ('spin_torque_gym.devices.vcma_mram', 'VCMA-MRAM device'),
            ('spin_torque_gym.devices.skyrmion_device', 'Skyrmion device'),
            ('spin_torque_gym.devices.device_factory', 'Device factory'),
            ('spin_torque_gym.physics.materials', 'Materials database'),
            ('spin_torque_gym.physics.simple_solver', 'Simple physics solver'),
            ('spin_torque_gym.utils.cache', 'Caching utilities'),
            ('spin_torque_gym.utils.performance', 'Performance utilities'),
            ('spin_torque_gym.utils.error_handling', 'Error handling'),
            ('spin_torque_gym.utils.security_validation', 'Security validation'),
            ('spin_torque_gym.utils.advanced_monitoring', 'Advanced monitoring'),
        ]
        
        for module_name, description in test_imports:
            try:
                __import__(module_name)
                import_tests.append({'module': module_name, 'passed': True, 'error': None})
            except Exception as e:
                import_tests.append({'module': module_name, 'passed': False, 'error': str(e)})
        
        # Calculate score
        passed_imports = sum(1 for test in import_tests if test['passed'])
        total_imports = len(import_tests)
        score = passed_imports / total_imports
        passed = score >= 0.9  # 90% of imports must work
        
        if passed:
            message = f"Import test: {passed_imports}/{total_imports} modules imported successfully"
        else:
            failed_modules = [test['module'] for test in import_tests if not test['passed']]
            message = f"Import failures: {', '.join(failed_modules[:3])}"
        
        details = {
            'total_imports': total_imports,
            'passed_imports': passed_imports,
            'import_results': import_tests
        }
        
        return passed, score, message, details


class DeviceInstantiationGate(QualityGate):
    """Test device instantiation without external dependencies."""
    
    def __init__(self):
        super().__init__("Device Creation", critical=True)
    
    def _execute(self):
        """Test device creation."""
        device_tests = []
        
        try:
            from spin_torque_gym.devices.device_factory import DeviceFactory
            
            factory = DeviceFactory()
            available_devices = factory.get_available_devices()
            
            # Test each device type
            for device_type in available_devices:
                try:
                    default_params = factory.get_default_parameters(device_type)
                    device = factory.create_device(device_type, default_params)
                    
                    # Test basic device operations
                    device_info = device.get_device_info()
                    
                    device_tests.append({
                        'device_type': device_type,
                        'passed': True,
                        'error': None
                    })
                    
                except Exception as e:
                    device_tests.append({
                        'device_type': device_type,
                        'passed': False,
                        'error': str(e)
                    })
        
        except Exception as e:
            device_tests.append({
                'device_type': 'factory',
                'passed': False,
                'error': str(e)
            })
        
        # Calculate score
        if device_tests:
            passed_devices = sum(1 for test in device_tests if test['passed'])
            total_devices = len(device_tests)
            score = passed_devices / total_devices
            passed = score >= 0.8  # 80% of devices must instantiate
            
            if passed:
                message = f"Device creation: {passed_devices}/{total_devices} devices created successfully"
            else:
                failed_devices = [test['device_type'] for test in device_tests if not test['passed']]
                message = f"Device creation failures: {', '.join(failed_devices)}"
        else:
            passed = False
            score = 0.0
            message = "No device tests could be run"
        
        details = {
            'device_tests': device_tests,
            'available_devices': available_devices if 'available_devices' in locals() else []
        }
        
        return passed, score, message, details


class DocumentationGate(QualityGate):
    """Check documentation quality."""
    
    def __init__(self):
        super().__init__("Documentation", critical=False)
    
    def _execute(self):
        """Analyze documentation coverage."""
        doc_files = 0
        code_files = 0
        total_functions = 0
        documented_functions = 0
        
        # Count documentation files
        for doc_file in Path('.').rglob('*.md'):
            if doc_file.is_file():
                doc_files += 1
        
        # Analyze Python files for docstrings
        for py_file in Path('.').rglob('*.py'):
            if py_file.is_file() and 'spin_torque_gym' in str(py_file):
                code_files += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not node.name.startswith('_'):  # Skip private
                                total_functions += 1
                                if ast.get_docstring(node):
                                    documented_functions += 1
                
                except Exception:
                    continue
        
        # Calculate documentation score
        docstring_ratio = documented_functions / total_functions if total_functions > 0 else 0
        doc_score = (doc_files * 10 + docstring_ratio * 100) / 110  # Normalize
        
        passed = doc_score >= 0.6
        score = min(1.0, doc_score)
        
        if passed:
            message = f"Documentation: {doc_files} doc files, {documented_functions}/{total_functions} functions documented"
        else:
            message = f"Insufficient documentation: {docstring_ratio:.1%} functions documented"
        
        details = {
            'doc_files': doc_files,
            'code_files': code_files,
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'docstring_ratio': docstring_ratio
        }
        
        return passed, score, message, details


class SecurityGate(QualityGate):
    """Basic security analysis."""
    
    def __init__(self):
        super().__init__("Security", critical=True)
    
    def _execute(self):
        """Perform basic security checks."""
        security_issues = []
        warnings = []
        files_scanned = 0
        
        # Scan Python files for security issues
        for py_file in Path('.').rglob('*.py'):
            if py_file.is_file():
                files_scanned += 1
                try:
                    content = py_file.read_text()
                    
                    # Check for dangerous patterns (excluding test files)
                    if 'test_' not in py_file.name:
                        # Check for eval/exec usage
                        if re.search(r'^\s*eval\s*\(', content, re.MULTILINE):
                            security_issues.append(f"eval() usage in {py_file.name}")
                        
                        if re.search(r'^\s*exec\s*\(', content, re.MULTILINE):
                            security_issues.append(f"exec() usage in {py_file.name}")
                        
                        # Check for shell injection risks
                        if 'subprocess' in content and 'shell=True' in content:
                            warnings.append(f"shell=True in {py_file.name}")
                        
                        # Check for hardcoded secrets
                        if re.search(r'password\s*=\s*["\'][^"\']{8,}["\']', content, re.IGNORECASE):
                            security_issues.append(f"Potential hardcoded password in {py_file.name}")
                
                except Exception:
                    continue
        
        # Test security utilities
        try:
            from spin_torque_gym.utils.security_validation import initialize_security
            initialize_security()
            
            from spin_torque_gym.utils.error_handling import setup_error_handling
            setup_error_handling()
            
        except Exception as e:
            warnings.append(f"Security utilities test failed: {e}")
        
        # Calculate security score
        if security_issues:
            passed = False
            score = max(0.0, 1.0 - len(security_issues) * 0.3)
            message = f"Security issues: {len(security_issues)} critical, {len(warnings)} warnings"
        else:
            passed = True
            score = 1.0 - (len(warnings) * 0.1)
            message = f"Security scan passed: {files_scanned} files checked"
        
        details = {
            'files_scanned': files_scanned,
            'security_issues': security_issues,
            'warnings': warnings
        }
        
        return passed, score, message, details


class FileStructureGate(QualityGate):
    """Validate project file structure."""
    
    def __init__(self):
        super().__init__("File Structure", critical=False)
    
    def _execute(self):
        """Check project structure."""
        expected_files = [
            'README.md',
            'pyproject.toml',
            'spin_torque_gym/__init__.py',
            'spin_torque_gym/devices/__init__.py',
            'spin_torque_gym/physics/__init__.py',
            'spin_torque_gym/envs/__init__.py',
            'spin_torque_gym/utils/__init__.py'
        ]
        
        missing_files = []
        present_files = []
        
        for file_path in expected_files:
            if Path(file_path).exists():
                present_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        # Count total Python files
        python_files = len(list(Path('.').rglob('*.py')))
        
        score = len(present_files) / len(expected_files)
        passed = score >= 0.9
        
        if passed:
            message = f"File structure: {len(present_files)}/{len(expected_files)} key files present, {python_files} total Python files"
        else:
            message = f"Missing files: {', '.join(missing_files[:3])}"
        
        details = {
            'expected_files': len(expected_files),
            'present_files': len(present_files),
            'missing_files': missing_files,
            'total_python_files': python_files
        }
        
        return passed, score, message, details


def run_quality_gates():
    """Execute all quality gates."""
    print("🛡️ SIMPLIFIED QUALITY GATES - SPIN TORQUE RL-GYM")
    print("Comprehensive Quality Assurance (Environment-Independent)")
    print("=" * 70)
    
    gates = [
        CodeStructureGate(),
        ImportGate(),
        DeviceInstantiationGate(),
        SecurityGate(),
        DocumentationGate(),
        FileStructureGate()
    ]
    
    results = {}
    critical_failures = []
    
    # Run each gate
    for gate in gates:
        print(f"\n🔍 Running {gate.name} Quality Gate...")
        print("-" * 50)
        
        passed, score, message, details = gate.run()
        
        results[gate.name] = {
            'passed': passed,
            'score': score,
            'message': message,
            'details': details,
            'critical': gate.critical
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {gate.name}: {message} (Score: {score:.2f})")
        
        if not passed and gate.critical:
            critical_failures.append(gate.name)
        
        # Show key details
        if 'files_checked' in details:
            print(f"   📁 Files analyzed: {details['files_checked']}")
        if 'total_imports' in details:
            print(f"   📦 Modules tested: {details['total_imports']}")
        if 'device_tests' in details:
            print(f"   🔧 Devices tested: {len(details['device_tests'])}")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 70)
    
    total_gates = len(gates)
    passed_gates = sum(1 for r in results.values() if r['passed'])
    critical_gates = sum(1 for g in gates if g.critical)
    passed_critical = sum(1 for r in results.values() if r['passed'] and r['critical'])
    
    overall_score = sum(r['score'] for r in results.values()) / len(results)
    
    print(f"📈 Overall Results: {passed_gates}/{total_gates} gates passed")
    print(f"⚡ Critical Gates: {passed_critical}/{critical_gates} passed")
    print(f"🎯 Overall Score: {overall_score:.2f}/1.00")
    
    # Deployment decision
    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES: {', '.join(critical_failures)}")
        print("🚫 DEPLOYMENT BLOCKED until critical issues are resolved.")
        deployment_ready = False
    else:
        print(f"\n✅ ALL CRITICAL GATES PASSED")
        if overall_score >= 0.9:
            print("🚀 DEPLOYMENT APPROVED - Excellent quality!")
            deployment_ready = True
        elif overall_score >= 0.8:
            print("✅ DEPLOYMENT APPROVED - Good quality with minor issues")
            deployment_ready = True
        elif overall_score >= 0.7:
            print("⚠️  DEPLOYMENT CONDITIONAL - Quality acceptable")
            deployment_ready = True
        else:
            print("❌ DEPLOYMENT NOT RECOMMENDED - Quality score too low")
            deployment_ready = False
    
    # Quality level assessment
    if overall_score >= 0.95:
        quality_level = "EXCELLENT"
        emoji = "🌟"
    elif overall_score >= 0.85:
        quality_level = "GOOD"
        emoji = "✅"
    elif overall_score >= 0.75:
        quality_level = "FAIR"
        emoji = "⚠️"
    elif overall_score >= 0.6:
        quality_level = "POOR"
        emoji = "❌"
    else:
        quality_level = "CRITICAL"
        emoji = "🚫"
    
    print(f"\n{emoji} QUALITY LEVEL: {quality_level}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for gate_name, result in results.items():
        status = "PASS ✅" if result['passed'] else "FAIL ❌"
        critical_flag = " (CRITICAL)" if result['critical'] else ""
        print(f"  • {gate_name}: {status}{critical_flag}")
        print(f"    {result['message']}")
    
    return deployment_ready, overall_score, results


def main():
    """Main entry point."""
    try:
        deployment_ready, overall_score, results = run_quality_gates()
        
        # Save results
        report = {
            'timestamp': time.time(),
            'deployment_ready': deployment_ready,
            'overall_score': overall_score,
            'results': results
        }
        
        with open('quality_report_simplified.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: quality_report_simplified.json")
        
        # Final verdict
        if deployment_ready and overall_score >= 0.85:
            print("\n🎉 QUALITY GATES: EXCELLENT - System ready for deployment!")
            return 0
        elif deployment_ready:
            print("\n✅ QUALITY GATES: APPROVED - Conditional deployment ready")
            return 0
        else:
            print("\n❌ QUALITY GATES: FAILED - System not ready for deployment")
            return 1
    
    except Exception as e:
        print(f"\n💥 QUALITY GATES ERROR: {e}")
        return 2


if __name__ == '__main__':
    sys.exit(main())