# 🔄 CI/CD Pipeline Guide

This guide covers the comprehensive CI/CD pipeline for the Luna Collection, including automated testing, deployment, quality assurance, and continuous integration best practices.

## Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [GitHub Actions Workflows](#-github-actions-workflows)
- [Testing Strategy](#-testing-strategy)
- [Code Quality](#-code-quality)
- [Deployment Process](#-deployment-process)
- [Monitoring & Alerting](#-monitoring--alerting)
- [Troubleshooting](#-troubleshooting)

---

## 📊 Pipeline Overview

The Luna Collection uses a comprehensive CI/CD pipeline built on GitHub Actions with automated testing, quality checks, and deployment processes.

### Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Push/PR   │───▶│   Quality   │───▶│   Testing   │───▶│ Deployment  │
│             │    │   Checks    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Linting    │    │  Security   │    │   Unit      │    │   Release   │
│             │    │  Scanning   │    │   Tests     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Key Features

- **Automated Testing**: Comprehensive test suite with coverage reporting
- **Code Quality**: Linting, type checking, and security scanning
- **Multi-Platform**: Windows, Linux, and macOS support
- **Performance Testing**: Automated performance regression detection
- **Security Scanning**: Vulnerability detection and dependency checking
- **Release Automation**: Automated versioning and package publishing

---

## ⚙️ GitHub Actions Workflows

### Main CI Workflow

#### `.github/workflows/ci.yml`
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Type checking
        run: |
          mypy . --ignore-missing-imports --no-strict-optional

      - name: Security scanning
        run: |
          bandit -r . -f json -o security-report.json

  test-suite:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest --cov=luna_collection --cov-report=xml --cov-report=html

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  performance-test:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest-benchmark

      - name: Run performance tests
        run: |
          pytest tests/performance/ --benchmark-json=perf-results.json

      - name: Compare performance
        run: |
          python scripts/compare_performance.py perf-results.json
```

### Release Workflow

#### `.github/workflows/release.yml`
```yaml
name: Release

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*
```

### Security Workflow

#### `.github/workflows/security.yml`
```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
  push:
    branches: [ main ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results
        uses: github/codecov/codecov-action@v3
        if: always()
        with:
          file: trivy-results.sarif
```

---

## 🧪 Testing Strategy

### Test Categories

#### Unit Tests
```python
# tests/unit/test_validation.py
import pytest
from luna_collection.validation import LunaInputValidator

class TestLunaInputValidator:
    def setup_method(self):
        self.validator = LunaInputValidator()

    def test_valid_node_input(self):
        """Test validation of valid node input."""
        input_data = {
            "steps": 20,
            "cfg": 8.0,
            "denoise": 1.0
        }

        result = self.validator.validate_node_input(input_data, "LunaSampler")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_node_input(self):
        """Test validation of invalid node input."""
        input_data = {
            "steps": "invalid",
            "cfg": 8.0,
            "denoise": 1.0
        }

        result = self.validator.validate_node_input(input_data, "LunaSampler")
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "steps" in str(result.errors[0])
```

#### Integration Tests
```python
# tests/integration/test_workflow_execution.py
import pytest
from luna_collection.core import WorkflowExecutor

class TestWorkflowExecution:
    def test_complete_workflow(self):
        """Test execution of complete workflow."""
        workflow = {
            "nodes": [
                {
                    "id": "sampler",
                    "type": "LunaSampler",
                    "inputs": {
                        "steps": 20,
                        "cfg": 8.0
                    }
                }
            ]
        }

        executor = WorkflowExecutor()
        result = executor.execute_workflow(workflow)

        assert result.success
        assert "output" in result.data
        assert result.execution_time < 30.0  # seconds
```

#### Performance Tests
```python
# tests/performance/test_inference_speed.py
import pytest
import time
from luna_collection.nodes import LunaSampler

class TestInferencePerformance:
    def setup_method(self):
        self.sampler = LunaSampler()

    def test_inference_speed(self, benchmark):
        """Benchmark inference speed."""
        def run_inference():
            return self.sampler.sample(
                model=self.model,
                steps=20,
                cfg=8.0
            )

        result = benchmark(run_inference)

        # Assert performance requirements
        assert result.stats.mean < 2.0  # seconds
        assert result.stats.stddev < 0.1  # consistency
```

### Test Configuration

#### `pytest.ini`
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --disable-warnings
    --tb=short
    --cov=luna_collection
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=85
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

#### Coverage Configuration

#### `.coveragerc`
```ini
[run]
source = luna_collection
omit =
    */tests/*
    */migrations/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov
```

---

## 🔍 Code Quality

### Linting Configuration

#### `.flake8`
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    .pytest_cache,
    .tox,
    build,
    dist,
    *.egg-info,
    .venv,
    venv,
    .env
per-file-ignores =
    __init__.py:F401
    tests/*:S101
max-complexity = 10
```

#### `mypy.ini`
```ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = True
```

### Pre-commit Hooks

#### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 🚀 Deployment Process

### Version Management

#### Semantic Versioning
```python
# version.py
__version__ = "1.2.3"

# Version bump script
def bump_version(version_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, __version__.split('.'))

    if version_type == 'major':
        return f"{major + 1}.0.0"
    elif version_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif version_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid version type: {version_type}")
```

### Release Process

#### Automated Release
```bash
# Create release branch
git checkout -b release/v1.2.3

# Update version
echo "1.2.3" > VERSION

# Update changelog
vim CHANGELOG.md

# Commit changes
git add VERSION CHANGELOG.md
git commit -m "Release v1.2.3"

# Push and create PR
git push origin release/v1.2.3
```

#### Release Checklist
- [ ] Update version number
- [ ] Update changelog
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release
- [ ] Deploy to production

### Package Publishing

#### `setup.py`
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="luna-collection",
    version="1.2.3",
    author="Luna Collection Team",
    description="Advanced ComfyUI node collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LSDJesus/ComfyUI-Luna-Collection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
        ],
    },
)
```

---

## 📊 Monitoring & Alerting

### Pipeline Metrics

#### GitHub Actions Metrics
```yaml
# Monitor workflow success rates
- name: Report metrics
  if: always()
  run: |
    echo "Workflow: ${{ github.workflow }}"
    echo "Status: ${{ job.status }}"
    echo "Duration: ${{ github.event.head_commit.timestamp }}"
    echo "Commit: ${{ github.sha }}"
```

#### Test Metrics
```python
# tests/conftest.py
import pytest
import time

@pytest.fixture(autouse=True)
def measure_test_duration(request):
    start_time = time.time()
    yield
    duration = time.time() - start_time

    # Log test metrics
    print(f"Test {request.node.name} took {duration:.3f}s")

    # Store for aggregation
    if hasattr(request.config, '_test_durations'):
        request.config._test_durations.append(duration)
```

### Alerting Configuration

#### Failure Notifications
```yaml
# .github/workflows/alert.yml
name: Alert on Failure

on:
  workflow_run:
    workflows: ["CI Pipeline"]
    types: [completed]

jobs:
  alert:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - name: Send Discord notification
        run: |
          curl -H "Content-Type: application/json" \
               -d '{"content": "🚨 CI Pipeline failed: ${{ github.event.workflow_run.html_url }}"}' \
               ${{ secrets.DISCORD_WEBHOOK_URL }}
```

#### Performance Regression Alerts
```python
# scripts/check_performance_regression.py
def check_performance_regression(current_results: dict, baseline_results: dict) -> bool:
    """Check for performance regressions."""
    regression_threshold = 0.1  # 10% degradation

    for test_name, current_time in current_results.items():
        baseline_time = baseline_results.get(test_name)
        if baseline_time:
            degradation = (current_time - baseline_time) / baseline_time
            if degradation > regression_threshold:
                print(f"Performance regression in {test_name}: {degradation:.1%}")
                return True

    return False
```

---

## 🔧 Troubleshooting

### Common CI/CD Issues

#### Workflow Failures

**Test Failures**
```
Symptoms: Tests failing in CI but passing locally
Solutions:
- Check Python version compatibility
- Verify dependency versions
- Review test environment setup
- Check for race conditions
```

**Linting Errors**
```
Symptoms: Code style violations blocking merge
Solutions:
- Run pre-commit hooks locally
- Fix formatting with black
- Address type hints for mypy
- Review flake8 violations
```

**Dependency Issues**
```
Symptoms: Package installation failures
Solutions:
- Check dependency version conflicts
- Update requirements files
- Use compatible Python versions
- Review platform-specific dependencies
```

#### Performance Issues

**Slow CI Runs**
```
Symptoms: CI taking too long to complete
Solutions:
- Optimize test parallelization
- Use test caching
- Reduce test scope for PRs
- Implement incremental testing
```

**Memory Issues**
```
Symptoms: Out of memory errors in CI
Solutions:
- Reduce test parallelism
- Use smaller test datasets
- Implement memory profiling
- Optimize resource usage
```

### Debug Tools

#### Local CI Simulation
```bash
# Simulate CI environment locally
docker run -it \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.10 \
  bash -c "
    pip install -r requirements-dev.txt
    flake8 .
    mypy .
    pytest --cov=luna_collection
  "
```

#### CI Debug Logs
```yaml
# Enable debug logging in workflows
- name: Debug information
  run: |
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version)"
    echo "Available memory: $(free -h)"
    echo "Disk usage: $(df -h)"
```

---

## 📈 Best Practices

### CI/CD Best Practices

#### Workflow Organization
- Keep workflows modular and reusable
- Use matrix builds for multi-platform testing
- Implement proper caching strategies
- Separate concerns (lint, test, deploy)

#### Testing Best Practices
- Write tests first (TDD)
- Maintain high test coverage (>85%)
- Use descriptive test names
- Test edge cases and error conditions

#### Quality Assurance
- Enforce code style consistency
- Use type hints throughout
- Implement security scanning
- Regular dependency updates

#### Deployment Best Practices
- Use semantic versioning
- Automate release process
- Implement rollback procedures
- Monitor deployment success

### Performance Optimization

#### CI Performance Tips
- Cache dependencies between runs
- Use incremental testing
- Parallelize independent jobs
- Optimize resource allocation

#### Test Performance
- Use fixtures efficiently
- Mock external dependencies
- Profile slow tests
- Implement test timeouts

---

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [Development Guide](development.md)
- [Performance Guide](performance-guide.md)
- [API Reference](api-reference.md)
- [Troubleshooting](troubleshooting.md)

---

## 📞 Support

For CI/CD pipeline support:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- Join the [Discord Community](https://discord.gg/luna-collection)

---

*CI/CD pipeline configuration may need adjustment based on project size, team preferences, and infrastructure constraints. Monitor and optimize based on your specific requirements.*