# Test Environment Quick Start

## 🚀 Fastest Way to Run Tests

```bash
cd tests
./run-tests.sh
```

That's it! This will:
1. Build test container with all dependencies (using `uv`)
2. Run all unit & integration tests
3. Show results

## 📝 Available Commands

```bash
# In tests/ directory

./run-tests.sh unit         # Unit tests only
./run-tests.sh integration  # Integration tests only
./run-tests.sh coverage     # With coverage report
./run-tests.sh all          # All tests (default)

./docker-shell.sh           # Enter test container shell
```

## 🔧 What's Different Here

This test environment is **completely independent**:

✅ Own `Dockerfile`
✅ Own `pyproject.toml` (with all test dependencies)
✅ Own dependency management (using `uv`, same as your services)
✅ No conflict with main project dependencies

## 📦 How It Works

```
tests/Dockerfile  →  Build container with uv
                 →  Install dependencies from pyproject.toml
                 →  Copy source code & tests
                 →  Ready to run pytest!
```

## 🐛 Troubleshooting

**Import errors?**
- The container handles PYTHONPATH automatically
- All dependencies are in the container

**Need to add a dependency?**
1. Edit `tests/pyproject.toml`
2. Run `./docker-shell.sh` (rebuilds automatically)

**Want to debug?**
```bash
./docker-shell.sh
# Now you're in the container
uv run pytest /app/tests/unit -v --pdb
```

## 📊 CI Integration

When you push to GitHub, the same test container runs automatically!

```
GitHub Actions  →  Build tests/Dockerfile
                →  Run uv pytest
                →  Report results
```

## ✨ Benefits

- 🔒 **Isolated**: Test dependencies don't affect main code
- 🏃 **Fast**: UV makes dependency installation super fast
- 🔁 **Reproducible**: Same environment locally and in CI
- 🎯 **Simple**: One script to rule them all

Ready to test! 🎉
