# Testing Documentation

## Overview

This directory contains comprehensive tests for the DnD Master game system, covering all microservices:
- **Combat Agent** (port 9000) - Combat mechanics and battle management
- **Rule Agent** (port 9002) - D&D rule validation using RAG
- **Orchestrator/API Gateway** (port 8000) - Main game controller
- **ChromaDB** (port 8000) - Vector database for rule retrieval

Tests are organized in three layers following the Testing Pyramid:

```
        /\
       /  \  System Tests (5-10 tests)
      /────\
     / Integ \ Integration Tests (20-30 tests)
    /────────\
   /   Unit   \ Unit Tests (50-100 tests)
  /────────────\
```

## Test Structure

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   └── test_combat_engine.py  # Combat mechanics unit tests
├── integration/               # API integration tests (TestClient)
│   ├── test_combat_api.py     # Combat Agent API tests
│   ├── test_rule_agent.py     # Rule Agent API tests
│   └── test_orchestrator.py   # Orchestrator/Gateway API tests
└── system/                    # End-to-end tests (requires Docker)
    ├── test_combat_system.py  # Combat-specific system tests
    └── test_full_game_flow.py # Complete multi-service flows
```

## Running Tests

### Method 1: Using Test Script (Recommended)

The easiest way to run tests is using the provided test script:

```bash
# Run all tests
./run-tests.sh all

# Run specific test types
./run-tests.sh unit
./run-tests.sh integration
./run-tests.sh system

# Generate coverage report
./run-tests.sh coverage
```

### Method 2: Using Docker Compose Directly

```bash
# Build test container
docker-compose -f docker-compose.test.yml build test-runner

# Run unit tests
docker-compose -f docker-compose.test.yml run --rm test-runner \
    pytest tests/unit -v -m unit

# Run integration tests
docker-compose -f docker-compose.test.yml run --rm test-runner \
    pytest tests/integration -v -m integration

# Run system tests (with backend services)
docker-compose -f docker-compose.test.yml up -d combat-agent-test rule-agent-test chromadb-test
docker-compose -f docker-compose.test.yml run --rm test-runner \
    pytest tests/system -v -m system
docker-compose -f docker-compose.test.yml down -v
```

### Method 3: Local Python Environment (Advanced)

If you prefer to run tests without Docker:

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend-tmp:$(pwd)/src/rule_agent:$(pwd)/src/orchestrator"

# Run tests
pytest tests/ -v
```

## Test Markers

Tests use pytest markers for easy filtering:

- `@pytest.mark.unit` - Fast unit tests (isolated, no external dependencies)
- `@pytest.mark.integration` - API integration tests (using TestClient)
- `@pytest.mark.system` - System tests (requires running Docker services)
- `@pytest.mark.slow` - Slow running tests

Filter tests by marker:

```bash
# Run only fast tests (skip slow)
./run-tests.sh all -m "not slow"

# Or with Docker directly
docker-compose -f docker-compose.test.yml run --rm test-runner \
    pytest -m "unit or integration"
```

## Test Coverage

### Generate Coverage Report

```bash
# Using test script (easiest)
./run-tests.sh coverage

# Or manually with Docker
docker-compose -f docker-compose.test.yml run --rm \
    -v $(pwd)/htmlcov:/app/htmlcov \
    test-runner \
    pytest tests/unit tests/integration \
        --cov=src/backend-tmp/api \
        --cov=src/rule_agent \
        --cov=src/orchestrator \
        --cov-report=html

# Open report in browser
open htmlcov/index.html
```

## Continuous Integration

### GitHub Actions

The CI pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**CI Jobs:**

1. **Unit & Integration Tests (Containerized)**
   - Builds dedicated test container (`tests/Dockerfile`)
   - Runs unit tests in isolated container
   - Runs integration tests using FastAPI TestClient
   - Generates coverage report (html + xml)
   - All dependencies managed in container

2. **System Tests (Docker Compose)**
   - Uses `docker-compose.test.yml` for orchestration
   - Starts backend services (Combat Agent, Rule Agent, ChromaDB)
   - Waits for all services to be healthy
   - Runs E2E tests in test container against real services
   - Tests complete game flows (narrative → combat)
   - Only runs if unit/integration tests pass

3. **Code Quality Checks**
   - Black (code formatting)
   - isort (import sorting)
   - flake8 (linting)

**Key Features:**
- ✅ Test container ensures consistent environment
- ✅ No need to install dependencies on host
- ✅ Same test setup locally and in CI
- ✅ Services use healthchecks for reliability

### View CI Results

Check the **Actions** tab in GitHub to see test results and logs.

## Writing New Tests

### Unit Test Example

```python
import pytest
from api.utils.combat_engine import Character

@pytest.mark.unit
class TestCharacter:
    def test_character_creation(self):
        char = Character("Knight", 0, 20, 18, {"STR": 4}, 6, 10)
        assert char.name == "Knight"
        assert char.hp == 20
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
class TestCombatAPI:
    def test_start_combat(self, api_test_client):
        response = api_test_client.post("/combat/start", json={})
        assert response.status_code == 200
```

### System Test Example

```python
import pytest
import requests

@pytest.mark.system
class TestCombatSystem:
    def test_full_flow(self):
        response = requests.post("http://localhost:9000/combat/start", json={})
        assert response.status_code == 200
```

## Fixtures

Shared test fixtures are defined in `conftest.py`:

- `sample_player` - Test player character
- `sample_enemy` - Test enemy character
- `sample_characters` - Set of test characters
- `combat_engine` - Initialized combat engine
- `api_test_client` - FastAPI TestClient
- `sample_combat_request` - Sample API request data

### Using Fixtures

```python
def test_with_fixture(sample_player):
    """Fixture is automatically injected."""
    assert sample_player.hp == 20
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Add src/backend-tmp to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend-tmp"
pytest
```

### System Tests Fail

System tests require running Docker services:

```bash
# Start services
docker-compose up -d combat-agent

# Check service health
curl http://localhost:9000/health

# View logs
docker-compose logs combat-agent

# Run tests
pytest tests/system -v

# Stop services
docker-compose down
```

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow"
```

## Best Practices

1. **Test Naming**
   - Use descriptive names: `test_character_takes_damage_correctly`
   - Classes: `TestCharacter`, `TestCombatEngine`

2. **Test Organization**
   - One test file per module
   - Group related tests in classes
   - Use fixtures for common setup

3. **Assertions**
   - One logical assertion per test
   - Use descriptive assertion messages
   - Test both success and failure cases

4. **Test Independence**
   - Tests should not depend on each other
   - Clean up resources (sessions, files)
   - Use fixtures for setup/teardown

5. **Coverage Goals**
   - Unit tests: 80%+ coverage
   - Integration tests: Cover all API endpoints
   - System tests: Cover critical user flows

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
