# Standalone Test Environment

This directory contains a standalone test environment using `uv` for dependency management.

## Quick Start

### 1. Build Test Environment

```bash
cd tests
./docker-shell.sh
```

This will:
- Build a test container with all dependencies
- Enter the container shell
- Mount source code for testing

### 2. Run Tests Inside Container

Once inside the container:

```bash
# Run all tests
uv run pytest /app/tests -v

# Run specific test types
uv run pytest /app/tests/unit -v -m unit
uv run pytest /app/tests/integration -v -m integration

# Generate coverage
uv run pytest /app/tests/unit /app/tests/integration \
    --cov=/app/src/backend-tmp/api \
    --cov=/app/src/rule_agent \
    --cov=/app/src/orchestrator \
    --cov-report=html
```

### 3. Run Tests from Host

Without entering the container:

```bash
cd tests

# Run specific test types
./run-tests.sh unit
./run-tests.sh integration
./run-tests.sh coverage
./run-tests.sh all  # default
```

## Dependency Management

### Using UV

This test environment uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

**Key Files:**
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions (auto-generated)

### Adding Dependencies

1. Edit `pyproject.toml`:
```toml
dependencies = [
    "pytest==7.4.4",
    "your-new-package==1.0.0",
]
```

2. Rebuild the container:
```bash
./docker-shell.sh
```

UV will automatically:
- Resolve dependencies
- Update `uv.lock`
- Install packages

### Updating Dependencies

```bash
# Inside container
cd /app/tests
uv lock --upgrade
```

## Architecture

```
tests/
├── Dockerfile              # Test container definition (using uv)
├── pyproject.toml          # Python dependencies
├── uv.lock                 # Locked dependency versions
├── docker-shell.sh         # Enter test container
├── run-tests.sh            # Run tests from host
├── unit/                   # Unit tests
├── integration/            # Integration tests
└── system/                 # System tests
```

## Why UV?

- **Fast**: 10-100x faster than pip
- **Reliable**: Deterministic dependency resolution
- **Compatible**: Works with standard `pyproject.toml`
- **Consistent**: Same as backend-tmp and rule_agent services

## Troubleshooting

### "uv: command not found"

The container should have uv pre-installed. If not:
```bash
pip install uv
```

### Import errors

Check PYTHONPATH is set correctly:
```bash
echo $PYTHONPATH
# Should include: /app/src/backend-tmp:/app/src/rule_agent:/app/src/orchestrator
```

### Dependency conflicts

Regenerate lock file:
```bash
cd /app/tests
rm uv.lock
uv sync
```

## Comparison with Main Project

| Aspect | Main Project | Test Environment |
|--------|-------------|------------------|
| Dependency Tool | uv | uv (same) |
| Python Version | 3.12 | 3.12 (same) |
| Base Image | python:3.12-slim-bookworm | python:3.12-slim-bookworm (same) |
| Config File | pyproject.toml | tests/pyproject.toml |
| Purpose | Production services | Testing only |

## Integration with CI

GitHub Actions uses the same test container:

```yaml
- name: Build test container
  run: docker build -f tests/Dockerfile -t dnd/test-runner:ci .

- name: Run tests
  run: |
    docker run --rm dnd/test-runner:ci \
      uv run pytest /app/tests -v
```

This ensures local and CI environments are identical.
