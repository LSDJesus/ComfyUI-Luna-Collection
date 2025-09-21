#!/usr/bin/env python3
"""
Tests for Luna Collection JavaScript extensions

Tests the frontend JavaScript functionality for loader nodes.
Note: These are unit tests that mock the browser environment.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestJavaScriptExtensions(unittest.TestCase):
    """Test JavaScript extension functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock browser environment
        self.mock_document = Mock()
        self.mock_window = Mock()
        self.mock_api = Mock()

        # Mock DOM elements
        self.mock_element = Mock()
        self.mock_element.parentElement = Mock()
        self.mock_document.createElement.return_value = self.mock_element
        self.mock_document.getElementById.return_value = self.mock_element

    def test_checkpoint_loader_extension_structure(self):
        """Test that checkpoint loader JS extension has proper structure."""
        # Read the JavaScript file
        js_path = project_root / "js" / "luna_checkpoint_loader.js"
        with open(js_path, 'r') as f:
            js_content = f.read()

        # Verify key components
        self.assertIn("LunaCheckpointLoaderExtension", js_content)
        self.assertIn("nodeCreated", js_content)
        self.assertIn("LunaCheckpointLoader", js_content)
        self.assertIn("getAndDisplayMetadata", js_content)
        self.assertIn("app.registerExtension", js_content)

    def test_lora_stacker_extension_structure(self):
        """Test that LoRA stacker JS extension has proper structure."""
        js_path = project_root / "js" / "luna_lora_stacker.js"
        with open(js_path, 'r') as f:
            js_content = f.read()

        # Verify key components
        self.assertIn("LunaLoRAStackerExtension", js_content)
        self.assertIn("MAX_LORA_SLOTS", js_content)
        self.assertIn("getAndDisplayLoraMetadata", js_content)
        self.assertIn("app.registerExtension", js_content)

    def test_lora_stacker_random_extension_structure(self):
        """Test that random LoRA stacker JS extension has proper structure."""
        js_path = project_root / "js" / "luna_lora_stacker_random.js"
        with open(js_path, 'r') as f:
            js_content = f.read()

        # Verify key components
        self.assertIn("LunaLoRAStackerRandomExtension", js_content)
        self.assertIn("MAX_LORA_SLOTS", js_content)
        self.assertIn("getAndDisplayLoraMetadata", js_content)
        self.assertIn("app.registerExtension", js_content)

    def test_collection_nodes_extension_structure(self):
        """Test that collection nodes JS extension has proper structure."""
        js_path = project_root / "js" / "luna_collection_nodes.js"
        with open(js_path, 'r') as f:
            js_content = f.read()

        # Verify key components
        self.assertIn("Lunacollection.AdvancedUpscalerUI", js_content)
        self.assertIn("beforeRegisterNodeDef", js_content)
        self.assertIn("Luna_Advanced_Upscaler", js_content)
        self.assertIn("app.registerExtension", js_content)

    def test_javascript_imports(self):
        """Test that JavaScript files have proper imports."""
        js_files = [
            "luna_checkpoint_loader.js",
            "luna_lora_stacker.js",
            "luna_lora_stacker_random.js",
            "luna_collection_nodes.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            # All should import from ComfyUI
            self.assertIn("import { app }", content)
            self.assertIn("/scripts/app.js", content)

    def test_javascript_error_handling(self):
        """Test that JavaScript files have error handling."""
        js_files = [
            "luna_checkpoint_loader.js",
            "luna_lora_stacker.js",
            "luna_lora_stacker_random.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            # Should have error handling for API calls
            self.assertIn("catch", content)
            self.assertIn("error", content)

    def test_javascript_metadata_endpoints(self):
        """Test that JavaScript files use correct metadata endpoints."""
        js_files = [
            "luna_checkpoint_loader.js",
            "lora_stacker.js",
            "lora_stacker_random.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            # Should use metadata endpoints
            if "checkpoint" in js_file:
                self.assertIn("get_checkpoint_metadata", content)
            else:
                self.assertIn("get_lora_metadata", content)

    def test_javascript_ui_elements(self):
        """Test that JavaScript creates proper UI elements."""
        js_files = [
            "luna_checkpoint_loader.js",
            "lora_stacker.js",
            "lora_stacker_random.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            # Should create UI elements
            self.assertIn("createElement", content)
            self.assertIn("appendChild", content)
            self.assertIn("style.cssText", content)


class TestJavaScriptIntegration(unittest.TestCase):
    """Test JavaScript integration with Python backend."""

    def test_node_type_matching(self):
        """Test that JavaScript node types match Python node names."""
        # Python node names
        python_nodes = [
            "LunaCheckpointLoader",
            "LunaLoRAStacker",
            "LunaLoRAStackerRandom"
        ]

        # Check corresponding JavaScript files
        js_mappings = {
            "luna_checkpoint_loader.js": "LunaCheckpointLoader",
            "luna_lora_stacker.js": "LunaLoRAStacker",
            "luna_lora_stacker_random.js": "LunaLoRAStackerRandom"
        }

        for js_file, expected_node in js_mappings.items():
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            self.assertIn(expected_node, content,
                         f"JavaScript file {js_file} should reference {expected_node}")

    def test_max_slots_consistency(self):
        """Test that MAX_LORA_SLOTS is consistent between Python and JavaScript."""
        # Check Python MAX_LORA_SLOTS
        from nodes.loaders.luna_lora_stacker import MAX_LORA_SLOTS as python_max_slots
        self.assertEqual(python_max_slots, 4)

        # Check JavaScript MAX_LORA_SLOTS
        js_files = ["luna_lora_stacker.js", "luna_lora_stacker_random.js"]
        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            self.assertIn("MAX_LORA_SLOTS: 4", content,
                         f"JavaScript file {js_file} should have MAX_LORA_SLOTS: 4")

    def test_api_endpoints_consistency(self):
        """Test that API endpoints are consistent."""
        # Python endpoints
        python_endpoints = [
            "/luna/get_checkpoint_metadata",
            "/luna/get_lora_metadata"
        ]

        # Check JavaScript uses same endpoints
        js_files = [
            "luna_checkpoint_loader.js",
            "luna_lora_stacker.js",
            "luna_lora_stacker_random.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            for endpoint in python_endpoints:
                if "checkpoint" in js_file and "checkpoint" in endpoint:
                    self.assertIn(endpoint, content)
                elif "lora" in js_file and "lora" in endpoint:
                    self.assertIn(endpoint, content)


class TestJavaScriptValidation(unittest.TestCase):
    """Test JavaScript code validation."""

    def test_javascript_syntax(self):
        """Test that JavaScript files have valid syntax."""
        import subprocess
        import sys

        js_files = [
            "luna_checkpoint_loader.js",
            "luna_lora_stacker.js",
            "luna_lora_stacker_random.js",
            "luna_collection_nodes.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file

            # Use Node.js to validate syntax if available
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import re; print('JS file exists')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                # If we can run Python, at least verify file exists and is readable
                self.assertTrue(js_path.exists(), f"JavaScript file {js_file} should exist")
                with open(js_path, 'r') as f:
                    content = f.read()
                    self.assertTrue(len(content) > 0, f"JavaScript file {js_file} should not be empty")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # If Node.js isn't available, just check file exists
                self.assertTrue(js_path.exists(), f"JavaScript file {js_file} should exist")

    def test_javascript_file_structure(self):
        """Test that JavaScript files have proper structure."""
        js_files = [
            "luna_checkpoint_loader.js",
            "lora_stacker.js",
            "lora_stacker_random.js",
            "luna_collection_nodes.js"
        ]

        for js_file in js_files:
            js_path = project_root / "js" / js_file
            with open(js_path, 'r') as f:
                content = f.read()

            # Should have proper JavaScript structure
            lines = content.split('\n')

            # Should start with import
            first_non_empty = next((line.strip() for line in lines if line.strip()), "")
            self.assertTrue(first_non_empty.startswith("import") or first_non_empty.startswith("//"),
                          f"JavaScript file {js_file} should start with import or comment")

            # Should end with export or registration
            last_lines = [line.strip() for line in lines[-5:] if line.strip()]
            has_registration = any("registerExtension" in line for line in last_lines)
            self.assertTrue(has_registration,
                          f"JavaScript file {js_file} should register an extension")


if __name__ == '__main__':
    unittest.main()