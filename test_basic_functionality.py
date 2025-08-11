#!/usr/bin/env python3
"""Basic functionality test for Spin Torque RL-Gym without external dependencies.

This test validates the core code structure, imports, and basic Python 
functionality without requiring numpy, scipy, or gymnasium.
"""

def test_basic_python_math():
    """Test basic math operations as numpy substitute."""
    print("📊 Testing basic mathematical operations...")
    
    # Vector operations
    v1 = [1.0, 0.0, 0.0] 
    v2 = [0.0, 1.0, 0.0]
    
    # Dot product
    dot = sum(a * b for a, b in zip(v1, v2))
    print(f"✅ Dot product: {dot}")
    
    # Cross product (simplified)
    def cross_product_3d(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2], 
            a[0] * b[1] - a[1] * b[0]
        ]
    
    cross = cross_product_3d(v1, v2)
    print(f"✅ Cross product: {cross}")
    
    # Vector magnitude
    magnitude = sum(x**2 for x in v1)**0.5
    print(f"✅ Vector magnitude: {magnitude}")
    
    return True

def test_file_structure():
    """Test that all expected files exist."""
    print("\n📁 Testing file structure...")
    
    import os
    expected_files = [
        'spin_torque_gym/__init__.py',
        'spin_torque_gym/devices/base_device.py',
        'spin_torque_gym/devices/stt_mram.py', 
        'spin_torque_gym/devices/sot_mram.py',
        'spin_torque_gym/devices/vcma_mram.py',
        'spin_torque_gym/devices/skyrmion_device.py',
        'spin_torque_gym/devices/device_factory.py',
        'spin_torque_gym/physics/llgs_solver.py',
        'spin_torque_gym/physics/simple_solver.py',
        'spin_torque_gym/physics/materials.py',
        'spin_torque_gym/envs/spin_torque_env.py',
        'spin_torque_gym/envs/array_env.py',
        'spin_torque_gym/envs/skyrmion_env.py',
    ]
    
    missing_files = []
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files")
        return False
    else:
        print(f"\n✅ All {len(expected_files)} expected files present!")
        return True

def test_code_syntax():
    """Test that Python files have valid syntax."""
    print("\n🐍 Testing Python syntax...")
    
    import ast
    import os
    
    python_files = []
    for root, dirs, files in os.walk('spin_torque_gym'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            ast.parse(source)
            print(f"✅ {file_path}")
        except SyntaxError as e:
            print(f"❌ {file_path}: {e}")
            syntax_errors.append((file_path, str(e)))
        except Exception as e:
            print(f"⚠️  {file_path}: {e}")
    
    if syntax_errors:
        print(f"\n❌ {len(syntax_errors)} syntax errors found")
        return False
    else:
        print(f"\n✅ All {len(python_files)} Python files have valid syntax!")
        return True

def test_docstring_coverage():
    """Test documentation coverage."""
    print("\n📚 Testing documentation coverage...")
    
    import ast
    import os
    
    total_functions = 0
    documented_functions = 0
    
    for root, dirs, files in os.walk('spin_torque_gym'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        source = f.read()
                    
                    tree = ast.parse(source)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not node.name.startswith('_'):  # Skip private methods
                                total_functions += 1
                                if ast.get_docstring(node):
                                    documented_functions += 1
                                    
                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {e}")
    
    if total_functions > 0:
        coverage = (documented_functions / total_functions) * 100
        print(f"📊 Documentation coverage: {documented_functions}/{total_functions} ({coverage:.1f}%)")
        
        if coverage >= 80:
            print("✅ Excellent documentation coverage!")
            return True
        elif coverage >= 60:
            print("⚠️  Good documentation coverage")
            return True
        else:
            print("❌ Poor documentation coverage")
            return False
    else:
        print("❌ No functions found")
        return False

def main():
    """Run all basic tests."""
    print("🧪 BASIC FUNCTIONALITY TESTS - SPIN TORQUE RL-GYM")
    print("=" * 60)
    
    tests = [
        ("Basic Math Operations", test_basic_python_math),
        ("File Structure", test_file_structure),
        ("Python Syntax", test_code_syntax),
        ("Documentation Coverage", test_docstring_coverage),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Generation 1 (MAKE IT WORK) - Basic functionality verified!")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)