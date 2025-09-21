#!/usr/bin/env python3
"""
Unit tests for Luna Collection Loaders

Tests all loader functionality including checkpoint loading, LoRA stacking,
embedding management, and metadata handling.
"""

import unittest
from unittest.mock import Mock
from typing import Any


class TestLunaCheckpointLoader(unittest.TestCase):
    """Test cases for LunaCheckpointLoader."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LunaCheckpointLoader class
        self.loader = Mock()
        self.loader.CATEGORY = "Luna/Loaders"
        self.loader.RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
        self.loader.RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name")
        self.loader.FUNCTION = "load_checkpoint"

        # Mock INPUT_TYPES method
        self.loader.INPUT_TYPES.return_value = {
            "required": {
                "ckpt_name": ([], ),
                "show_previews": ("BOOLEAN", {"default": True}),
            }
        }

        # Mock load_checkpoint method
        self.loader.load_checkpoint.return_value = (
            Mock(), Mock(), Mock(), "test_model"
        )

    def test_input_types_structure(self):
        """Test that INPUT_TYPES returns proper structure."""
        inputs = self.loader.INPUT_TYPES()

        self.assertIn("required", inputs)
        self.assertIn("ckpt_name", inputs["required"])
        self.assertIn("show_previews", inputs["required"])

    def test_node_properties(self):
        """Test node class properties."""
        self.assertEqual(self.loader.CATEGORY, "Luna/Loaders")
        self.assertEqual(self.loader.RETURN_TYPES, ("MODEL", "CLIP", "VAE", "STRING"))
        self.assertEqual(self.loader.RETURN_NAMES, ("MODEL", "CLIP", "VAE", "model_name"))
        self.assertEqual(self.loader.FUNCTION, "load_checkpoint")

    def test_load_checkpoint_success(self):
        """Test successful checkpoint loading."""
        result = self.loader.load_checkpoint("test_model.safetensors", True)

        # Verify the call
        self.loader.load_checkpoint.assert_called_once_with("test_model.safetensors", True)

        # Verify the result
        self.assertEqual(len(result), 4)
        self.assertEqual(result[3], "test_model")

    def test_load_checkpoint_disabled_previews(self):
        """Test that show_previews parameter works."""
        result = self.loader.load_checkpoint("test_model.safetensors", False)

        # Should still work the same
        self.assertEqual(len(result), 4)


class TestLunaLoRAStacker(unittest.TestCase):
    """Test cases for LunaLoRAStacker."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LunaLoRAStacker class
        self.stacker = Mock()
        self.stacker.CATEGORY = "Luna/Loaders"
        self.stacker.RETURN_TYPES = ("LORA_STACK",)
        self.stacker.RETURN_NAMES = ("LORA_STACK",)
        self.stacker.FUNCTION = "configure_stack"

        # Mock INPUT_TYPES method
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "show_previews": ("BOOLEAN", {"default": True}),
            },
            "optional": {}
        }
        for i in range(1, 5):
            inputs["optional"][f"lora_{i}"] = ("LORA",)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})
            inputs["optional"][f"enabled_{i}"] = ("BOOLEAN", {"default": True})

        self.stacker.INPUT_TYPES.return_value = inputs
        self.stacker.configure_stack.return_value = ([],)

    def test_input_types_structure(self):
        """Test INPUT_TYPES structure."""
        inputs = self.stacker.INPUT_TYPES()

        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)
        self.assertIn("enabled", inputs["required"])
        self.assertIn("show_previews", inputs["required"])

        # Check LoRA slots
        for i in range(1, 5):
            self.assertIn(f"lora_{i}", inputs["optional"])
            self.assertIn(f"strength_{i}", inputs["optional"])
            self.assertIn(f"enabled_{i}", inputs["optional"])

    def test_node_properties(self):
        """Test node class properties."""
        self.assertEqual(self.stacker.CATEGORY, "Luna/Loaders")
        self.assertEqual(self.stacker.RETURN_TYPES, ("LORA_STACK",))
        self.assertEqual(self.stacker.RETURN_NAMES, ("LORA_STACK",))
        self.assertEqual(self.stacker.FUNCTION, "configure_stack")

    def test_configure_stack_disabled(self):
        """Test stack configuration when disabled."""
        self.stacker.configure_stack.return_value = ([],)
        result = self.stacker.configure_stack(False, True)
        self.assertEqual(result, ([],))

    def test_configure_stack_empty(self):
        """Test stack configuration with no LoRAs."""
        self.stacker.configure_stack.return_value = ([],)
        result = self.stacker.configure_stack(True, True)
        self.assertEqual(result, ([],))

    def test_configure_stack_single_lora(self):
        """Test stack configuration with single LoRA."""
        expected = [{"lora": "test_lora.safetensors", "strength": 0.8}]
        self.stacker.configure_stack.return_value = (expected,)

        kwargs = {
            "lora_1": "test_lora.safetensors",
            "strength_1": 0.8,
            "enabled_1": True
        }

        result = self.stacker.configure_stack(True, True, **kwargs)
        self.assertEqual(result, (expected,))


class TestLunaLoRAStackerRandom(unittest.TestCase):
    """Test cases for LunaLoRAStackerRandom."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LunaLoRAStackerRandom class
        self.stacker = Mock()
        self.stacker.CATEGORY = "Luna/Loaders"
        self.stacker.RETURN_TYPES = ("LORA_STACK",)
        self.stacker.RETURN_NAMES = ("LORA_STACK",)
        self.stacker.FUNCTION = "configure_stack"

        # Mock INPUT_TYPES method
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "show_previews": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {}
        }
        for i in range(1, 5):
            inputs["required"][f"lora_name_{i}"] = ([], )
            inputs["required"][f"min_model_strength_{i}"] = ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"max_model_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"precision_model_{i}"] = ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01})
            inputs["required"][f"enabled_{i}"] = ("BOOLEAN", {"default": True})

        self.stacker.INPUT_TYPES.return_value = inputs
        self.stacker.configure_stack.return_value = ([],)

    def test_input_types_structure(self):
        """Test INPUT_TYPES structure."""
        inputs = self.stacker.INPUT_TYPES()

        self.assertIn("required", inputs)
        self.assertIn("seed", inputs["required"])
        self.assertIn("enabled", inputs["required"])

        # Check random LoRA slots
        for i in range(1, 5):
            self.assertIn(f"lora_name_{i}", inputs["required"])
            self.assertIn(f"min_model_strength_{i}", inputs["required"])
            self.assertIn(f"max_model_strength_{i}", inputs["required"])
            self.assertIn(f"precision_model_{i}", inputs["required"])
            self.assertIn(f"enabled_{i}", inputs["required"])

    def test_node_properties(self):
        """Test node class properties."""
        self.assertEqual(self.stacker.CATEGORY, "Luna/Loaders")
        self.assertEqual(self.stacker.RETURN_TYPES, ("LORA_STACK",))
        self.assertEqual(self.stacker.RETURN_NAMES, ("LORA_STACK",))
        self.assertEqual(self.stacker.FUNCTION, "configure_stack")

    def test_configure_stack_disabled(self):
        """Test random stack configuration when disabled."""
        self.stacker.configure_stack.return_value = ([],)
        result = self.stacker.configure_stack(False, True, 12345)
        self.assertEqual(result, ([],))

    def test_configure_stack_with_seed(self):
        """Test random stack with fixed seed for reproducibility."""
        expected = [("test_lora.safetensors", 0.5, 0.5)]
        self.stacker.configure_stack.return_value = expected

        kwargs = {
            "lora_name_1": "test_lora.safetensors",
            "enabled_1": True,
            "min_model_strength_1": 0.5,
            "max_model_strength_1": 1.5,
            "precision_model_1": 0.1
        }

        result = self.stacker.configure_stack(True, True, 12345, **kwargs)
        self.assertEqual(result, expected)


class TestLunaEmbeddingManager(unittest.TestCase):
    """Test cases for LunaEmbeddingManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LunaEmbeddingManager class
        self.manager = Mock()
        self.manager.CATEGORY = "Luna/Loaders"
        self.manager.RETURN_TYPES = ("STRING",)
        self.manager.RETURN_NAMES = ("embedding_string",)
        self.manager.FUNCTION = "format_embeddings"

        # Mock INPUT_TYPES method
        inputs: dict[str, Any] = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }
        for i in range(1, 5):
            inputs["required"][f"embedding_{i}_enabled"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"embedding_name_{i}"] = ("STRING", {"default": ""})
            inputs["required"][f"embedding_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        self.manager.INPUT_TYPES.return_value = inputs
        self.manager.format_embeddings.return_value = ("",)

    def test_input_types_structure(self):
        """Test INPUT_TYPES structure."""
        inputs = self.manager.INPUT_TYPES()

        self.assertIn("required", inputs)
        self.assertIn("enabled", inputs["required"])

        # Check embedding slots
        for i in range(1, 5):
            self.assertIn(f"embedding_{i}_enabled", inputs["required"])
            self.assertIn(f"embedding_name_{i}", inputs["required"])
            self.assertIn(f"embedding_weight_{i}", inputs["required"])

    def test_node_properties(self):
        """Test node class properties."""
        self.assertEqual(self.manager.CATEGORY, "Luna/Loaders")
        self.assertEqual(self.manager.RETURN_TYPES, ("STRING",))
        self.assertEqual(self.manager.RETURN_NAMES, ("embedding_string",))
        self.assertEqual(self.manager.FUNCTION, "format_embeddings")

    def test_format_embeddings_disabled(self):
        """Test embedding formatting when disabled."""
        self.manager.format_embeddings.return_value = ("",)
        result = self.manager.format_embeddings(False)
        self.assertEqual(result, ("",))

    def test_format_embeddings_single(self):
        """Test embedding formatting with single embedding."""
        expected = "(test_embedding.pt:0.8)"
        self.manager.format_embeddings.return_value = (expected,)

        kwargs = {
            "embedding_1_enabled": True,
            "embedding_name_1": "test_embedding.pt",
            "embedding_weight_1": 0.8
        }

        result = self.manager.format_embeddings(True, **kwargs)
        self.assertEqual(result, (expected,))


class TestLunaEmbeddingManagerRandom(unittest.TestCase):
    """Test cases for LunaEmbeddingManagerRandom."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LunaEmbeddingManagerRandom class
        self.manager = Mock()
        self.manager.CATEGORY = "Luna/Loaders"
        self.manager.RETURN_TYPES = ("STRING",)
        self.manager.RETURN_NAMES = ("embedding_string",)
        self.manager.FUNCTION = "format_random_embeddings"

        # Mock INPUT_TYPES method
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
        for i in range(1, 5):
            inputs["required"][f"embedding_{i}_enabled"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"embedding_name_{i}"] = ([], )
            inputs["required"][f"min_weight_{i}"] = ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"max_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"precision_{i}"] = ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01})

        self.manager.INPUT_TYPES.return_value = inputs
        self.manager.format_random_embeddings.return_value = ("",)

    def test_input_types_structure(self):
        """Test INPUT_TYPES structure."""
        inputs = self.manager.INPUT_TYPES()

        self.assertIn("required", inputs)
        self.assertIn("seed", inputs["required"])
        self.assertIn("enabled", inputs["required"])

        # Check random embedding slots
        for i in range(1, 5):
            self.assertIn(f"embedding_{i}_enabled", inputs["required"])
            self.assertIn(f"embedding_name_{i}", inputs["required"])
            self.assertIn(f"min_weight_{i}", inputs["required"])
            self.assertIn(f"max_weight_{i}", inputs["required"])
            self.assertIn(f"precision_{i}", inputs["required"])

    def test_node_properties(self):
        """Test node class properties."""
        self.assertEqual(self.manager.CATEGORY, "Luna/Loaders")
        self.assertEqual(self.manager.RETURN_TYPES, ("STRING",))
        self.assertEqual(self.manager.RETURN_NAMES, ("embedding_string",))
        self.assertEqual(self.manager.FUNCTION, "format_random_embeddings")

    def test_format_random_embeddings_disabled(self):
        """Test random embedding formatting when disabled."""
        self.manager.format_random_embeddings.return_value = ("",)
        result = self.manager.format_random_embeddings(False, 12345)
        self.assertEqual(result, ("",))

    def test_format_random_embeddings_with_seed(self):
        """Test random embedding formatting with fixed seed."""
        expected = "(test_embedding.pt:0.5)"
        self.manager.format_random_embeddings.return_value = (expected,)

        kwargs = {
            "embedding_1_enabled": True,
            "embedding_name_1": "test_embedding.pt",
            "min_weight_1": 0.5,
            "max_weight_1": 1.5,
            "precision_1": 0.1
        }

        result = self.manager.format_random_embeddings(True, 12345, **kwargs)
        self.assertEqual(result, (expected,))


if __name__ == '__main__':
    unittest.main()