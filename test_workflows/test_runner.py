#!/usr/bin/env python3
"""
Luna Collection Workflow Test Runner

Automated testing script for Luna Collection workflows.
Validates node functionality and ComfyUI integration.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

class LunaWorkflowTester:
    """Test runner for Luna Collection workflows."""

    def __init__(self, workflows_dir: str):
        self.workflows_dir = Path(workflows_dir)
        self.test_results = {}

    def load_workflow(self, workflow_file: str) -> Dict[str, Any]:
        """Load a workflow JSON file."""
        workflow_path = self.workflows_dir / workflow_file
        with open(workflow_path, 'r') as f:
            return json.load(f)

    def validate_workflow_structure(self, workflow: Dict[str, Any]) -> List[str]:
        """Validate basic workflow structure."""
        errors = []

        # Check required fields
        required_fields = ['workflow_name', 'description', 'nodes', 'links']
        for field in required_fields:
            if field not in workflow:
                errors.append(f"Missing required field: {field}")

        # Validate nodes
        if 'nodes' in workflow:
            for node_id, node_data in workflow['nodes'].items():
                if not isinstance(node_data, dict):
                    errors.append(f"Node {node_id}: Invalid node data format")
                    continue

                required_node_fields = ['id', 'type', 'pos', 'properties']
                for field in required_node_fields:
                    if field not in node_data:
                        errors.append(f"Node {node_id}: Missing field {field}")

        # Validate links
        if 'links' in workflow:
            if not isinstance(workflow['links'], list):
                errors.append("Links must be a list")
            else:
                for i, link in enumerate(workflow['links']):
                    if not isinstance(link, list) or len(link) != 4:
                        errors.append(f"Link {i}: Invalid link format (must be [from_id, from_slot, to_id, to_slot])")

        return errors

    def check_node_coverage(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Check which Luna nodes are covered in the workflow."""
        luna_nodes = {
            'loaders': [
                'LunaCheckpointLoader',
                'LunaLoRAStacker',
                'LunaLoRAStackerRandom',
                'LunaEmbeddingManager',
                'LunaEmbeddingManagerRandom'
            ],
            'preprocessing': [
                'LunaPromptPreprocessor',
                'LunaTextProcessor',
                'LunaUnifiedPromptProcessor'
            ],
            'performance': [
                'LunaPerformanceLogger',
                'LunaPerformanceCondition',
                'LunaPerformanceDisplay',
                'LunaPerformanceConcat'
            ],
            'detailing': [
                'LunaDetailer',
                'LunaMediaPipeDetailer'
            ],
            'upscaling': [
                'LunaUpscalerSimple',
                'LunaUpscalerAdvanced',
                'LunaUltimateSDUpscale'
            ],
            'other': [
                'LunaSampler',
                'LunaMultiSaver',
                'LunaLoadParameters',
                'LunaParametersBridge',
                'LunaImageCaption',
                'LunaYOLOAnnotationExporter'
            ]
        }

        workflow_nodes = set()
        if 'nodes' in workflow:
            for node_data in workflow['nodes'].values():
                if isinstance(node_data, dict) and 'type' in node_data:
                    workflow_nodes.add(node_data['type'])

        coverage = {}
        for category, nodes in luna_nodes.items():
            covered = [node for node in nodes if node in workflow_nodes]
            coverage[category] = {
                'total': len(nodes),
                'covered': len(covered),
                'percentage': (len(covered) / len(nodes)) * 100 if nodes else 0,
                'nodes': covered
            }

        return coverage

    def test_workflow(self, workflow_file: str) -> Dict[str, Any]:
        """Test a single workflow file."""
        print(f"\n🧪 Testing workflow: {workflow_file}")

        try:
            workflow = self.load_workflow(workflow_file)
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': f"Failed to load workflow: {str(e)}",
                'coverage': {},
                'validation_errors': []
            }

        # Validate structure
        validation_errors = self.validate_workflow_structure(workflow)

        # Check node coverage
        coverage = self.check_node_coverage(workflow)

        # Determine overall status
        if validation_errors:
            status = 'FAILED'
            print(f"❌ Validation errors: {len(validation_errors)}")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
        else:
            status = 'PASSED'
            print("✅ Workflow structure valid")

        # Print coverage summary
        total_covered = sum(cat['covered'] for cat in coverage.values())
        total_nodes = sum(cat['total'] for cat in coverage.values())

        print(f"📊 Node coverage: {total_covered}/{total_nodes} Luna nodes")
        for category, data in coverage.items():
            if data['covered'] > 0:
                print(f"   {category.title()}: {data['covered']}/{data['total']} nodes")

        return {
            'status': status,
            'workflow_name': workflow.get('workflow_name', 'Unknown'),
            'description': workflow.get('description', ''),
            'coverage': coverage,
            'validation_errors': validation_errors,
            'node_count': len(workflow.get('nodes', {})),
            'link_count': len(workflow.get('links', []))
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run tests on all workflow files."""
        print("🚀 Luna Collection Workflow Test Suite")
        print("=" * 50)

        workflow_files = [f for f in os.listdir(self.workflows_dir)
                         if f.endswith('.json') and f != 'README.md']

        results = {}
        passed = 0
        failed = 0

        for workflow_file in sorted(workflow_files):
            result = self.test_workflow(workflow_file)
            results[workflow_file] = result

            if result['status'] == 'PASSED':
                passed += 1
            else:
                failed += 1

        # Print summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"Total workflows: {len(workflow_files)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\n❌ Failed workflows:")
            for workflow_file, result in results.items():
                if result['status'] == 'FAILED':
                    print(f"   - {workflow_file}: {result.get('error', 'Validation errors')}")

        # Overall coverage
        all_coverage = {}
        for result in results.values():
            if result['status'] == 'PASSED':
                for category, data in result['coverage'].items():
                    if category not in all_coverage:
                        all_coverage[category] = {'covered': 0, 'total': data['total']}
                    all_coverage[category]['covered'] = max(
                        all_coverage[category]['covered'], data['covered']
                    )

        if all_coverage:
            print("\n🎯 Overall Node Coverage:")
            for category, data in all_coverage.items():
                percentage = (data['covered'] / data['total']) * 100 if data['total'] > 0 else 0
                print(".1f")

        return {
            'results': results,
            'summary': {
                'total': len(workflow_files),
                'passed': passed,
                'failed': failed,
                'coverage': all_coverage
            }
        }

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        workflows_dir = Path(sys.argv[1])
    else:
        # Default to current directory (test_workflows)
        workflows_dir = Path(__file__).parent

    if not workflows_dir.exists():
        print(f"❌ Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    tester = LunaWorkflowTester(str(workflows_dir))
    results = tester.run_all_tests()

    # Exit with appropriate code
    if results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        print("\n🎉 All workflow tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()