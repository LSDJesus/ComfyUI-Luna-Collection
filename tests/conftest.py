"""
Pytest configuration and shared fixtures for Luna Collection tests.
"""

import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image as PyTorch tensor."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_image_pil():
    """Create a sample image as PIL Image."""
    return Image.new('RGB', (64, 64), color='red')


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with sample content."""
    content = """Line 1: This is a test line
Line 2: Another test line
Line 3: Third line for testing
Line 4: Final test line"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def temp_image_file():
    """Create a temporary image file."""
    img = Image.new('RGB', (64, 64), color='blue')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def mock_validator():
    """Mock validator for testing."""
    from validation import LunaInputValidator
    return LunaInputValidator()


@pytest.fixture(scope="session")
def luna_test_config():
    """Test configuration for Luna Collection."""
    return {
        "test_mode": True,
        "mock_external_deps": True,
        "temp_dir": tempfile.gettempdir(),
        "test_timeout": 30,
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Reset any global state
    from validation import validation_cache
    validation_cache.clear()

    # Set test environment variables
    os.environ['LUNA_TEST_MODE'] = '1'

    yield

    # Cleanup
    if 'LUNA_TEST_MODE' in os.environ:
        del os.environ['LUNA_TEST_MODE']


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "utils: Utility function tests")
    config.addinivalue_line("markers", "validation: Input validation tests")