#!/usr/bin/env python3
"""Core functionality test without gymnasium dependency.

Tests the implemented device models and physics directly.
"""

import sys
import os

# Add the repo to Python path
sys.path.insert(0, '/root/repo')

def test_device_models():
    """Test device model implementations."""
    print("🔧 Testing Device Models...")
    
    try:
        # Test base device
        from spin_torque_gym.devices.base_device import BaseSpintronicDevice
        print("✅ Base device imported successfully")
        
        # Test STT-MRAM
        from spin_torque_gym.devices.stt_mram import STTMRAMDevice
        print("✅ STT-MRAM device imported successfully")
        
        # Test device instantiation
        device_params = {
            'volume': 50e-9 * 100e-9 * 2e-9,
            'saturation_magnetization': 800e3,
            'damping': 0.01,
            'uniaxial_anisotropy': 1e6,
            'polarization': 0.7,
            'reference_magnetization': [0, 0, 1],
            'easy_axis': [0, 0, 1]
        }
        
        stt_device = STTMRAMDevice(device_params)
        print(f"✅ STT-MRAM device created: {stt_device}")
        
        # Test device operations
        magnetization = [0, 0, 1]
        applied_field = [0, 0, 1000]
        effective_field = stt_device.compute_effective_field(magnetization, applied_field)
        print(f"✅ Effective field computed: {effective_field}")
        
        resistance = stt_device.compute_resistance(magnetization)
        print(f"✅ Resistance computed: {resistance}")
        
        # Test SOT-MRAM
        from spin_torque_gym.devices.sot_mram import SOTMRAMDevice
        sot_params = device_params.copy()
        sot_params.update({
            'spin_hall_angle': 0.2,
            'damping_like_efficiency': 0.2,
            'field_like_efficiency': 0.1
        })
        
        sot_device = SOTMRAMDevice(sot_params)
        print(f"✅ SOT-MRAM device created: {sot_device}")
        
        # Test VCMA-MRAM
        from spin_torque_gym.devices.vcma_mram import VCMAMRAMDevice
        vcma_params = device_params.copy()
        vcma_params.update({
            'vcma_coefficient': 100e-6,
            'dielectric_thickness': 1e-9
        })
        
        vcma_device = VCMAMRAMDevice(vcma_params)
        print(f"✅ VCMA-MRAM device created: {vcma_device}")
        
        # Test Skyrmion device
        from spin_torque_gym.devices.skyrmion_device import SkyrmionDevice
        skyrmion_params = device_params.copy()
        skyrmion_params.update({
            'dmi_constant': 3e-3,
            'skyrmion_radius': 20e-9,
            'exchange_constant': 15e-12
        })
        
        skyrmion_device = SkyrmionDevice(skyrmion_params)
        print(f"✅ Skyrmion device created: {skyrmion_device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device model test failed: {e}")
        return False


def test_physics_models():
    """Test physics simulation components."""
    print("\n🔬 Testing Physics Models...")
    
    try:
        # Test materials database
        from spin_torque_gym.physics.materials import MaterialDatabase
        materials = MaterialDatabase()
        available_materials = materials.get_available_materials()
        print(f"✅ Materials database: {len(available_materials)} materials available")
        
        # Test specific material
        cofeb_props = materials.get_material_properties('CoFeB')
        print(f"✅ CoFeB properties loaded: Ms = {cofeb_props['saturation_magnetization']}")
        
        # Test simple solver
        from spin_torque_gym.physics.simple_solver import SimplePhysicsSolver
        solver = SimplePhysicsSolver()
        print(f"✅ Simple physics solver created: {solver}")
        
        # Test thermal model
        from spin_torque_gym.physics.thermal_model import ThermalFluctuations
        thermal = ThermalFluctuations()
        print(f"✅ Thermal model created: {thermal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics model test failed: {e}")
        return False


def test_utility_modules():
    """Test utility modules."""
    print("\n🔧 Testing Utility Modules...")
    
    try:
        # Test error handling
        from spin_torque_gym.utils.error_handling import setup_error_handling, safe_division
        setup_error_handling()
        result = safe_division(10, 2)
        print(f"✅ Error handling utilities work: {result}")
        
        # Test security validation
        from spin_torque_gym.utils.security_validation import initialize_security
        initialize_security()
        print("✅ Security validation system initialized")
        
        # Test monitoring
        from spin_torque_gym.utils.advanced_monitoring import initialize_monitoring
        monitor = initialize_monitoring(start_background=False)
        status = monitor.get_status_report()
        print(f"✅ Monitoring system initialized: {status['overall_health']}")
        
        # Test performance optimization
        from spin_torque_gym.utils.performance_optimization import initialize_performance_optimization
        optimizer = initialize_performance_optimization()
        report = optimizer.get_performance_report()
        print(f"✅ Performance optimization initialized: {report['optimization_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility module test failed: {e}")
        return False


def test_configurations():
    """Test configuration system."""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        from spin_torque_gym.config import get_config, validate_config
        
        config = get_config()
        print(f"✅ Configuration loaded: {len(config)} sections")
        
        # Test validation
        test_config = {
            'environment': {
                'max_steps': 1000,
                'render_mode': None
            }
        }
        
        validated = validate_config(test_config)
        print(f"✅ Configuration validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all core functionality tests."""
    print("🧪 CORE FUNCTIONALITY TESTS - SPIN TORQUE RL-GYM")
    print("Testing without gymnasium dependency")
    print("=" * 60)
    
    tests = [
        ("Device Models", test_device_models),
        ("Physics Models", test_physics_models),
        ("Utility Modules", test_utility_modules),
        ("Configuration System", test_configurations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
        print("✅ System core is working without external dependencies")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())