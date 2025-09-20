"""
Unit tests for Luna Collection nodes.

These tests validate individual node functionality in isolation.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_node_imports():
    """Test that all nodes can be imported."""
    try:
        # Test basic imports
        import luna_collection
        from luna_collection import nodes

        # Test specific node imports
        from luna_collection.nodes import (
            LunaSampler, LunaSimpleUpscaler, LunaAdvancedUpscaler,
            LunaMediaPipeDetailer, LunaPerformanceLogger
        )

        return {
            'success': True,
            'message': 'All node imports successful'
        }

    except ImportError as e:
        return {
            'success': False,
            'message': f'Import failed: {e}'
        }

def test_validation_system():
    """Test validation system functionality."""
    try:
        from luna_collection.validation import LunaInputValidator

        validator = LunaInputValidator()

        # Test valid input
        valid_input = {
            'steps': 20,
            'cfg': 8.0,
            'denoise': 1.0
        }

        result = validator.validate_node_input(valid_input, 'LunaSampler')

        if not result.is_valid:
            return {
                'success': False,
                'message': f'Valid input rejected: {result.errors}'
            }

        # Test invalid input
        invalid_input = {
            'steps': -1,  # Invalid
            'cfg': 8.0,
            'denoise': 1.0
        }

        result = validator.validate_node_input(invalid_input, 'LunaSampler')

        if result.is_valid:
            return {
                'success': False,
                'message': 'Invalid input accepted'
            }

        return {
            'success': True,
            'message': 'Validation system working correctly'
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Validation test failed: {e}'
        }

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    try:
        from luna_collection.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Test basic monitoring
        with monitor.measure('test_operation'):
            import time
            time.sleep(0.1)  # Simulate work

        metrics = monitor.get_metrics('test_operation')

        if not metrics or metrics.get('count', 0) == 0:
            return {
                'success': False,
                'message': 'Performance monitoring not working'
            }

        return {
            'success': True,
            'message': 'Performance monitoring working correctly',
            'details': metrics
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Performance monitoring test failed: {e}'
        }

def test_memory_management():
    """Test memory management utilities."""
    try:
        from luna_collection.utils.memory import MemoryManager

        manager = MemoryManager()

        # Test memory tracking
        initial_memory = manager.get_current_memory()

        # Allocate some memory
        test_data = [0] * 1000000  # ~4MB

        current_memory = manager.get_current_memory()
        memory_increase = current_memory - initial_memory

        # Clean up
        del test_data

        if memory_increase <= 0:
            return {
                'success': False,
                'message': 'Memory tracking not working'
            }

        return {
            'success': True,
            'message': 'Memory management working correctly',
            'details': {
                'memory_increase': memory_increase,
                'initial_memory': initial_memory,
                'current_memory': current_memory
            }
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Memory management test failed: {e}'
        }

def test_configuration_loading():
    """Test configuration loading and validation."""
    try:
        from luna_collection.utils.config import ConfigManager

        config_manager = ConfigManager()

        # Test default configuration
        config = config_manager.get_config()

        if not config:
            return {
                'success': False,
                'message': 'Configuration loading failed'
            }

        # Test configuration validation
        is_valid = config_manager.validate_config(config)

        if not is_valid:
            return {
                'success': False,
                'message': 'Configuration validation failed'
            }

        return {
            'success': True,
            'message': 'Configuration system working correctly'
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Configuration test failed: {e}'
        }

# Test registry
TESTS = {
    'node_imports': test_node_imports,
    'validation_system': test_validation_system,
    'performance_monitoring': test_performance_monitoring,
    'memory_management': test_memory_management,
    'configuration_loading': test_configuration_loading
}