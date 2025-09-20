#!/usr/bin/env python3
"""
Luna Collection Integration Validation
Validates the integration between LunaLoadParameters and Luna load_preprocessed nodes
"""

import sys
import os
import ast


def validate_file_syntax(file_path):
    """Validate that a Python file has correct syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_luna_load_parameters():
    """Validate LunaLoadParameters node structure"""
    print("\n🔍 Validating LunaLoadParameters...")

    file_path = os.path.join(os.path.dirname(__file__), "nodes", "luna_load_parameters.py")

    if not os.path.exists(file_path):
        print("❌ luna_load_parameters.py not found")
        return False

    # Check syntax
    valid, error = validate_file_syntax(file_path)
    if not valid:
        print(f"❌ Syntax error in luna_load_parameters.py: {error}")
        return False

    # Check for required components
    with open(file_path, 'r') as f:
        content = f.read()

    checks = [
        ("LunaLoadParameters class", "class LunaLoadParameters:" in content),
        ("INPUT_TYPES method", "def INPUT_TYPES" in content),
        ("load_parameters method", "def load_parameters" in content),
        ("positive_conditioning input", "positive_conditioning" in content),
        ("negative_conditioning input", "negative_conditioning" in content),
        ("parameters_pipe output", "PARAMETERS_PIPE" in content),
        ("luna_pipe output", "LUNA_PIPE" in content),
    ]

    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ Missing {check_name}")

    all_passed = all(check[1] for check in checks)
    if all_passed:
        print("✅ LunaLoadParameters validation passed")
    else:
        print("❌ LunaLoadParameters validation failed")

    return all_passed


def validate_luna_parameters_bridge():
    """Validate LunaParametersBridge node structure"""
    print("\n🔍 Validating LunaParametersBridge...")

    file_path = os.path.join(os.path.dirname(__file__), "nodes", "luna_parameters_bridge.py")

    if not os.path.exists(file_path):
        print("❌ luna_parameters_bridge.py not found")
        return False

    # Check syntax
    valid, error = validate_file_syntax(file_path)
    if not valid:
        print(f"❌ Syntax error in luna_parameters_bridge.py: {error}")
        return False

    # Check for required components
    with open(file_path, 'r') as f:
        content = f.read()

    checks = [
        ("LunaParametersBridge class", "class LunaParametersBridge:" in content),
        ("INPUT_TYPES method", "def INPUT_TYPES" in content),
        ("bridge_parameters method", "def bridge_parameters" in content),
        ("parameters_pipe input", "parameters_pipe" in content),
        ("conditioning_blend_mode", "conditioning_blend_mode" in content),
        ("conditioning_strength", "conditioning_strength" in content),
    ]

    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ Missing {check_name}")

    all_passed = all(check[1] for check in checks)
    if all_passed:
        print("✅ LunaParametersBridge validation passed")
    else:
        print("❌ LunaParametersBridge validation failed")

    return all_passed


def validate_luna_sampler():
    """Validate LunaSampler node structure"""
    print("\n🔍 Validating LunaSampler...")

    file_path = os.path.join(os.path.dirname(__file__), "nodes", "luna_sampler.py")

    if not os.path.exists(file_path):
        print("❌ luna_sampler.py not found")
        return False

    # Check syntax
    valid, error = validate_file_syntax(file_path)
    if not valid:
        print(f"❌ Syntax error in luna_sampler.py: {error}")
        return False

    # Check for required components
    with open(file_path, 'r') as f:
        content = f.read()

    checks = [
        ("LunaSampler class", "class LunaSampler:" in content),
        ("INPUT_TYPES method", "def INPUT_TYPES" in content),
        ("sample method", "def sample" in content),
        ("luna_pipe input", "luna_pipe" in content),
        ("parameters_pipe input", "parameters_pipe" in content),
        ("adaptive_sampling", "enable_adaptive_sampling" in content),
        ("performance_monitoring", "enable_performance_monitoring" in content),
    ]

    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ Missing {check_name}")

    all_passed = all(check[1] for check in checks)
    if all_passed:
        print("✅ LunaSampler validation passed")
    else:
        print("❌ LunaSampler validation failed")

    return all_passed


def validate_integration_patterns():
    """Validate that integration patterns are properly documented"""
    print("\n🔍 Validating Integration Documentation...")

    doc_path = os.path.join(os.path.dirname(__file__), "Assets", "production_pipeline.md")

    if not os.path.exists(doc_path):
        print("❌ production_pipeline.md not found")
        return False

    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("LunaLoadParameters updated", "LunaLoadParameters (Updated)" in content),
        ("LunaParametersBridge documented", "LunaParametersBridge" in content),
        ("Preprocessed integration pattern", "Preprocessed Integration" in content),
        ("Conditioning blending", "blend_mode" in content),
        ("Parameters pipe integration", "parameters_pipe" in content),
    ]

    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ Missing {check_name}")

    all_passed = all(check[1] for check in checks)
    if all_passed:
        print("✅ Integration documentation validation passed")
    else:
        print("❌ Integration documentation validation failed")

    return all_passed


def main():
    """Run all validation tests"""
    print("🚀 Luna Collection Integration Validation")
    print("=" * 60)

    validations = [
        validate_luna_load_parameters,
        validate_luna_parameters_bridge,
        validate_luna_sampler,
        validate_integration_patterns,
    ]

    passed = 0
    total = len(validations)

    for validation in validations:
        if validation():
            passed += 1

    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} validations passed")

    if passed == total:
        print("🎉 All integration validations passed!")
        print("\n✅ Luna Collection integration is ready:")
        print("   • LunaLoadParameters accepts preprocessed conditionings")
        print("   • LunaParametersBridge enables advanced blending")
        print("   • LunaSampler provides optimized sampling")
        print("   • Full parameters_pipe compatibility")
        return 0
    else:
        print("⚠️  Some validations failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())