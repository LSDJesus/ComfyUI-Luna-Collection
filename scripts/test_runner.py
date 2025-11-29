#!/usr/bin/env python3
"""
Luna Collection Test Suite Runner

Comprehensive testing framework for validating all Luna Collection nodes
before public release. This suite ensures reliability, performance, and
compatibility across different scenarios.

Usage:
    python test_runner.py                    # Run all tests
    python test_runner.py --unit-only        # Unit tests only
    python test_runner.py --integration-only # Integration tests only
    python test_runner.py --performance-only # Performance tests only
    python test_runner.py --quick            # Fast validation run
    python test_runner.py --verbose          # Detailed output
"""

import argparse
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"

@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: TestStatus
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuiteResult:
    """Complete test suite result."""
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def passed(self) -> int:
        return len([r for r in self.results if r.status == TestStatus.PASSED])

    @property
    def failed(self) -> int:
        return len([r for r in self.results if r.status == TestStatus.FAILED])

    @property
    def errors(self) -> int:
        return len([r for r in self.results if r.status == TestStatus.ERROR])

    @property
    def skipped(self) -> int:
        return len([r for r in self.results if r.status == TestStatus.SKIPPED])

    @property
    def total(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        """Generate test summary."""
        return f"""
Test Suite: {self.suite_name}
Duration: {self.duration:.2f}s
Results: {self.passed}/{self.total} passed
Details:
  - Passed: {self.passed}
  - Failed: {self.failed}
  - Errors: {self.errors}
  - Skipped: {self.skipped}
"""

class LunaTestRunner:
    """Main test runner for Luna Collection."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results: List[TestSuiteResult] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run_unit_tests(self) -> TestSuiteResult:
        """Run unit tests for individual nodes."""
        self.log("Starting unit tests...")
        suite = TestSuiteResult("Unit Tests")
        suite.start_time = time.time()

        # Import test modules
        try:
            from tests.unit import test_nodes

            # Run node tests
            suite.results.extend(self._run_test_module(test_nodes, "Node Tests"))

        except ImportError as e:
            suite.results.append(TestResult(
                "Import Test Modules",
                TestStatus.ERROR,
                0.0,
                f"Failed to import test modules: {e}"
            ))

        suite.end_time = time.time()
        return suite

    def run_integration_tests(self) -> TestSuiteResult:
        """Run integration tests for complete workflows."""
        self.log("Starting integration tests...")
        suite = TestSuiteResult("Integration Tests")
        suite.start_time = time.time()

        try:
            # Integration tests not implemented yet
            suite.results.append(TestResult(
                "Integration Tests",
                TestStatus.SKIPPED,
                0.0,
                "Integration tests not yet implemented"
            ))

        except Exception as e:
            suite.results.append(TestResult(
                "Integration Tests",
                TestStatus.ERROR,
                0.0,
                f"Integration tests failed: {e}"
            ))

        suite.end_time = time.time()
        return suite

    def run_performance_tests(self) -> TestSuiteResult:
        """Run performance benchmark tests."""
        self.log("Starting performance tests...")
        suite = TestSuiteResult("Performance Tests")
        suite.start_time = time.time()

        try:
            # Performance tests not implemented yet
            suite.results.append(TestResult(
                "Performance Tests",
                TestStatus.SKIPPED,
                0.0,
                "Performance tests not yet implemented"
            ))

        except Exception as e:
            suite.results.append(TestResult(
                "Performance Tests",
                TestStatus.ERROR,
                0.0,
                f"Performance tests failed: {e}"
            ))

        suite.end_time = time.time()
        return suite

    def run_error_handling_tests(self) -> TestSuiteResult:
        """Run error handling and edge case tests."""
        self.log("Starting error handling tests...")
        suite = TestSuiteResult("Error Handling Tests")
        suite.start_time = time.time()

        try:
            # Error handling tests not implemented yet
            suite.results.append(TestResult(
                "Error Handling Tests",
                TestStatus.SKIPPED,
                0.0,
                "Error handling tests not yet implemented"
            ))

        except Exception as e:
            suite.results.append(TestResult(
                "Error Handling Tests",
                TestStatus.ERROR,
                0.0,
                f"Error handling tests failed: {e}"
            ))

        suite.end_time = time.time()
        return suite

    def _run_test_module(self, module, module_name: str) -> List[TestResult]:
        """Run all tests in a module."""
        results = []

        if not hasattr(module, 'TESTS'):
            results.append(TestResult(
                f"{module_name} - No Tests",
                TestStatus.SKIPPED,
                0.0,
                "Module has no TESTS attribute"
            ))
            return results

        for test_name, test_func in module.TESTS.items():
            start_time = time.time()

            try:
                self.log(f"Running {module_name}: {test_name}")
                result = test_func()

                if result.get('success', False):
                    status = TestStatus.PASSED
                    message = result.get('message', 'Test passed')
                else:
                    status = TestStatus.FAILED
                    message = result.get('message', 'Test failed')

                duration = time.time() - start_time
                results.append(TestResult(
                    f"{module_name}: {test_name}",
                    status,
                    duration,
                    message,
                    result.get('details', {})
                ))

            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    f"{module_name}: {test_name}",
                    TestStatus.ERROR,
                    duration,
                    f"Test error: {str(e)}",
                    {"exception": str(e), "traceback": sys.exc_info()}
                ))

        return results

    def run_validation_checklist(self) -> TestSuiteResult:
        """Run comprehensive validation checklist."""
        self.log("Running validation checklist...")
        suite = TestSuiteResult("Validation Checklist")
        suite.start_time = time.time()

        checklist_items = [
            self._check_imports,
            self._check_node_registration,
            self._check_dependencies,
            self._check_file_structure,
            self._check_documentation,
            self._check_examples
        ]

        for check_func in checklist_items:
            start_time = time.time()

            try:
                result = check_func()
                duration = time.time() - start_time

                status = TestStatus.PASSED if result['success'] else TestStatus.FAILED
                suite.results.append(TestResult(
                    result['name'],
                    status,
                    duration,
                    result.get('message', ''),
                    result.get('details', {})
                ))

                # Log result
                self.log(f"{result['name']}: {'PASS' if result['success'] else 'FAIL'} - {result.get('message', '')}")

            except Exception as e:
                duration = time.time() - start_time
                suite.results.append(TestResult(
                    check_func.__name__,
                    TestStatus.ERROR,
                    duration,
                    f"Checklist error: {str(e)}"
                ))
                self.log(f"{check_func.__name__}: ERROR - {str(e)}")

        suite.end_time = time.time()
        return suite

    def _check_imports(self) -> Dict[str, Any]:
        """Check that all modules can be imported."""
        try:
            # Test importing root-level packages
            import utils
            import validation
            import nodes

            return {
                'name': 'Module Imports',
                'success': True,
                'message': 'All available modules imported successfully'
            }
        except ImportError as e:
            return {
                'name': 'Module Imports',
                'success': False,
                'message': f'Import failed: {e}'
            }

    def _check_node_registration(self) -> Dict[str, Any]:
        """Check that nodes are properly registered."""
        try:
            # Import the main __init__ which aggregates all node registrations
            from nodes import NODE_CLASS_MAPPINGS

            # Count registered nodes
            node_count = len(NODE_CLASS_MAPPINGS)

            if node_count == 0:
                return {
                    'name': 'Node Registration',
                    'success': False,
                    'message': 'No nodes registered in NODE_CLASS_MAPPINGS'
                }

            return {
                'name': 'Node Registration',
                'success': True,
                'message': f'{node_count} nodes registered',
                'details': {'node_names': list(NODE_CLASS_MAPPINGS.keys())}
            }

        except Exception as e:
            return {
                'name': 'Node Registration',
                'success': False,
                'message': f'Registration check failed: {e}'
            }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check that all dependencies are available."""
        required_deps = [
            'torch', 'torchvision', 'numpy', 'PIL',
            'pydantic'
        ]

        missing_deps = []
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            return {
                'name': 'Dependencies',
                'success': False,
                'message': f'Missing dependencies: {missing_deps}'
            }

        return {
            'name': 'Dependencies',
            'success': True,
            'message': 'All dependencies available'
        }

    def _check_file_structure(self) -> Dict[str, Any]:
        """Check that file structure is correct."""
        required_files = [
            '__init__.py',
            'nodes/__init__.py',
            'validation/__init__.py',
            'utils/__init__.py',
            'README.md',
            'requirements.txt'
        ]

        missing_files = []
        for file_path in required_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            return {
                'name': 'File Structure',
                'success': False,
                'message': f'Missing files: {missing_files}'
            }

        return {
            'name': 'File Structure',
            'success': True,
            'message': 'File structure is correct'
        }

    def _check_documentation(self) -> Dict[str, Any]:
        """Check that documentation is complete."""
        required_docs = [
            'README.md',
            'assets/guides/node-reference.md',
            'assets/guides/performance-guide.md',
            'assets/guides/development-guide.md'
        ]

        missing_docs = []
        for doc_path in required_docs:
            if not (project_root / doc_path).exists():
                missing_docs.append(doc_path)

        if missing_docs:
            return {
                'name': 'Documentation',
                'success': False,
                'message': f'Missing documentation: {missing_docs}'
            }

        return {
            'name': 'Documentation',
            'success': True,
            'message': 'Documentation is complete'
        }

    def _check_examples(self) -> Dict[str, Any]:
        """Check that examples are available."""
        example_files = [
            'assets/samples/basic-generation.json',
            'assets/samples/advanced-processing.json',
            'assets/prompts/artistic.json',
            'assets/prompts/photorealistic.json'
        ]

        missing_examples = []
        for example_path in example_files:
            if not (project_root / example_path).exists():
                missing_examples.append(example_path)

        if missing_examples:
            return {
                'name': 'Examples',
                'success': False,
                'message': f'Missing examples: {missing_examples}'
            }

        return {
            'name': 'Examples',
            'success': True,
            'message': 'Examples are available'
        }

    def run_all_tests(self, test_types: Optional[List[str]] = None) -> List[TestSuiteResult]:
        """Run all test suites."""
        if test_types is None:
            test_types = ['unit', 'integration', 'performance', 'error_handling', 'validation']

        results = []

        if 'unit' in test_types:
            results.append(self.run_unit_tests())

        if 'integration' in test_types:
            results.append(self.run_integration_tests())

        if 'performance' in test_types:
            results.append(self.run_performance_tests())

        if 'error_handling' in test_types:
            results.append(self.run_error_handling_tests())

        if 'validation' in test_types:
            results.append(self.run_validation_checklist())

        return results

    def generate_report(self, results: List[TestSuiteResult]) -> str:
        """Generate comprehensive test report."""
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)
        total_errors = sum(r.errors for r in results)
        total_tests = sum(r.total for r in results)
        total_duration = sum(r.duration for r in results)

        report = f"""
{'='*60}
LUNA COLLECTION TEST REPORT
{'='*60}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Test Suites: {len(results)}
Total Tests: {total_tests}
Total Duration: {total_duration:.2f}s
Overall Result: {'PASSED' if total_failed == 0 and total_errors == 0 else 'FAILED'}

Results:
  ✓ Passed: {total_passed}
  ✗ Failed: {total_failed}
  ⚠ Errors: {total_errors}

SUITE DETAILS
-------------
"""

        for suite_result in results:
            report += f"\n{suite_result.summary()}"

        # Add recommendations
        if total_failed > 0 or total_errors > 0:
            report += "\n\nRECOMMENDATIONS\n---------------\n"
            if total_failed > 0:
                report += "• Review failed tests and fix issues\n"
            if total_errors > 0:
                report += "• Investigate error conditions and improve error handling\n"
            report += "• Run tests again after fixes\n"
            report += "• Consider adding more comprehensive error handling\n"

        report += f"\n{'='*60}\n"

        return report

    def save_report(self, report: str, filename: Optional[str] = None):
        """Save test report to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"test_report_{timestamp}.txt"

        report_path = project_root / "test_reports" / filename
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.log(f"Report saved to: {report_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Luna Collection Test Runner")
    parser.add_argument('--unit-only', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration-only', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance-only', action='store_true', help='Run performance tests only')
    parser.add_argument('--error-handling-only', action='store_true', help='Run error handling tests only')
    parser.add_argument('--validation-only', action='store_true', help='Run validation checklist only')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (subset of tests)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save-report', action='store_true', help='Save test report to file')

    args = parser.parse_args()

    # Determine which tests to run
    if args.unit_only:
        test_types = ['unit']
    elif args.integration_only:
        test_types = ['integration']
    elif args.performance_only:
        test_types = ['performance']
    elif args.error_handling_only:
        test_types = ['error_handling']
    elif args.validation_only:
        test_types = ['validation']
    elif args.quick:
        test_types = ['validation']  # Quick validation only
    else:
        test_types = ['unit', 'integration', 'performance', 'error_handling', 'validation']

    # Run tests
    runner = LunaTestRunner(verbose=args.verbose)
    results = runner.run_all_tests(test_types)

    # Generate and display report
    report = runner.generate_report(results)
    print(report)

    # Save report if requested
    if args.save_report:
        runner.save_report(report)

    # Exit with appropriate code
    total_failed = sum(r.failed for r in results)
    total_errors = sum(r.errors for r in results)

    if total_failed > 0 or total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()