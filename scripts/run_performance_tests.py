#!/usr/bin/env python3
"""
Performance testing runner for Luna Collection nodes.
Generates detailed performance reports and regression analysis.
"""

import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path


def run_performance_tests():
    """Run performance tests and generate reports."""
    print("🚀 Running Luna Collection Performance Tests")
    print("=" * 50)

    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Install performance dependencies if needed
    print("📦 Installing performance dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-performance.txt"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

    # Run performance tests with simplified timing
    print("⚡ Running performance benchmarks...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/performance/test_performance.py",
            "-v", "-s"
        ], capture_output=True, text=True, check=True)

        print("✅ Performance tests completed successfully!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Performance tests failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    # Generate performance report
    generate_performance_report()

    return True


def generate_performance_report():
    """Generate a simplified performance report."""
    print("📊 Generating performance report...")

    # Since we're not using pytest-benchmark, create a simple summary
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "python_version": sys.version,
            "platform": sys.platform,
            "note": "Performance tests completed. Detailed metrics printed to console."
        }
    }

    # Save simple report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📈 Performance Test Summary")
    print("=" * 40)
    print("✅ All performance tests completed successfully!")
    print("� Detailed metrics were printed during test execution.")
    print("💾 Report saved to: performance_report.json")


def compare_with_baseline():
    """Compare current results with baseline for regression detection."""
    print("\n🔍 Performance regression detection...")
    print("ℹ️  Baseline comparison not available with simplified performance tests.")
    print("💡 Consider using pytest-benchmark for detailed regression tracking.")


if __name__ == "__main__":
    success = run_performance_tests()
    if success:
        compare_with_baseline()
        print("\n🎉 Performance testing completed successfully!")
    else:
        print("\n💥 Performance testing failed!")
        sys.exit(1)