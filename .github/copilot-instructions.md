# GitHub Copilot Instructions for pyVertexModel

## Project Overview

**pyVertexModel** is a versatile 3D vertex model for modeling tissue and cell mechanics, particularly for tissue repair simulations. This is a scientific computing project written primarily in Python with performance-critical components in Cython.

- **Project Type**: Scientific computing / computational biology
- **Primary Language**: Python (3.9+)
- **Size**: Medium-sized research codebase
- **Key Technologies**: Python, Cython, NumPy, SciPy, VTK, PyVista
- **Target Versions**: Python 3.9 and 3.10 on Linux, macOS, and Windows

**Important**: The `paper` branch contains the polished version used in publications. The `main` branch is under active development.

## Environment Setup

### Initial Setup (REQUIRED)

**ALWAYS** follow this exact sequence for first-time setup:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build Cython extensions (REQUIRED before running)
python setup.py build_ext --inplace

# 3. Install package in editable mode (REQUIRED for tests and imports)
pip install -e .
```

**Critical Notes**:
- Step 2 (Cython build) MUST be run before running the application or tests
- Step 3 (editable install) MUST be run before running tests
- If you modify `.pyx` files, re-run step 2 to rebuild Cython extensions
- The Cython build creates `.c` and `.so` files in `src/pyVertexModel/Kg/` (these are in `.gitignore`)

### After Code Changes

```bash
# If you modified Python files only:
# No rebuild needed, changes take effect immediately

# If you modified .pyx files:
python setup.py build_ext --inplace

# If you modified setup.py or dependencies:
pip install -e .
```

## Build and Validation Commands

### Linting

The project uses **Ruff** for linting and formatting:

```bash
# Check and auto-fix all issues (recommended)
ruff check src/ Tests/ --fix

# Check specific file types
ruff check src/ --select I  # Import sorting
ruff check src/ --select E,F  # Errors and warnings
```

**Configuration**: See `[tool.ruff]` in `pyproject.toml`
- Target version: Python 3.9+
- Auto-fix is enabled by default
- Security checks are enabled (S rules)

**Note**: Current ruff config uses deprecated top-level settings. If you see warnings about this, the tool will still work correctly.

### Testing

Tests are in the `Tests/` directory and use pytest:

```bash
# Run all tests (most tests will fail due to missing test data files)
pytest Tests/ -v

# Run specific test file
pytest Tests/test_utils.py -v

# Run with coverage (as CI does)
pytest Tests/ --cov --cov-report=lcov

# Run tests excluding those that need data files
pytest Tests/ -k "not filename" -v
```

**Important Test Information**:
- Many tests require `.mat` data files in `Tests/data/` or `data/` directories
- These data files are gitignored and NOT included in the repository
- Tests that load data will fail with `FileNotFoundError` - this is expected
- Some tests pass without data files (e.g., `test_utils.py` tests like `test_assert_matrix`)
- CI runs tests with `-k "not filename"` to exclude data-dependent tests

**Using tox** (multi-environment testing):
```bash
# Note: tox has configuration issues with legacy setup
# Prefer using pytest directly as shown above
tox  # Run tests across Python versions (if working)
```

### Running the Application

```bash
# Main application (requires build first)
python src/pyVertexModel/main.py

# Note: main.py is configured for specific simulations
# Check the file to understand what it's set to run
```

**Output**: Results are saved to `Result/` directory with timestamp-based folder names (this directory is in `.gitignore`).

## Project Structure

### Root Files
- `setup.py` - Build configuration (Cython extensions)
- `pyproject.toml` - Project metadata, dependencies, tool configs (ruff, pytest, tox, coverage)
- `requirements.txt` - Python dependencies
- `README.md` - Basic usage instructions
- `.python-version` - Python version (3.10)

### Key Directories

```
src/pyVertexModel/          # Main source code
├── algorithm/              # Core vertex model algorithms
│   ├── VertexModelVoronoi3D.py
│   ├── newtonRaphson.py    # Newton-Raphson solver
│   └── vertexModel.py
├── Kg/                     # Energy and force calculations
│   ├── kg_functions.pyx    # Cython performance-critical code
│   ├── kgVolume.py         # Volume energy
│   ├── kgContractility.py  # Contractility forces
│   ├── kgViscosity.py      # Viscosity
│   └── kgSubstrate.py      # Substrate interactions
├── geometry/               # Geometric data structures
│   ├── geo.py              # Main geometry class
│   ├── cell.py             # Cell objects
│   └── face.py             # Face objects
├── mesh_remodelling/       # Mesh operations
│   ├── flip.py             # Flip operations
│   └── remodelling.py      # Mesh remodelling
├── parameters/             # Configuration
│   └── set.py              # Parameter sets (modify wing_disc() function)
├── util/                   # Utility functions
│   └── utils.py
└── main.py                 # Main entry point

Tests/                      # Test suite (pytest)
├── tests.py                # Test utilities and base classes
├── test_*.py               # Individual test modules
└── (data/)                 # Missing: test data files (.mat format)

Input/                      # Input data files
└── wing_disc_150.mat       # Wing disc model data

resources/                  # Additional resources
└── LblImg_imageSequence.mat
```

### Build Artifacts (Gitignored)

These files are generated during build and should NOT be committed:
- `src/pyVertexModel/Kg/kg_functions.c` - Generated from .pyx
- `src/pyVertexModel/Kg/*.so` - Compiled Cython extension
- `src/pyVertexModel/_version.py` - Auto-generated version file
- `build/` - Build directory
- `Result/` - Simulation results

## CI/CD Information

### GitHub Actions Workflows

**`.github/workflows/tests.yaml`** (runs on push to main):
1. Python 3.9 setup
2. Install dependencies: `pip install -r requirements.txt`
3. Build Cython: `python setup.py build_ext --inplace`
4. Install test deps: `pip install pytest pytest-cov`
5. Run tests: `pytest Tests/ -k "not filename" --doctest-modules --cov . --cov-report=xml`
6. Upload coverage to Codecov and Coveralls

**Key CI Facts**:
- Uses Python 3.9 on Ubuntu
- Skips tests requiring data files with `-k "not filename"`
- Coverage reports to Codecov (token required) and Coveralls
- Tests can continue on error (`continue-on-error: true`)

**`.github/workflows/python-publish.yml`** (runs on release):
- Builds and publishes package to PyPI using trusted publishing

## Common Pitfalls and Solutions

### 1. ModuleNotFoundError: No module named 'pyVertexModel'
**Solution**: Run `pip install -e .` after building Cython extensions

### 2. ImportError with kg_functions
**Solution**: Run `python setup.py build_ext --inplace` to build Cython extensions

### 3. Tests fail with FileNotFoundError for .mat files
**Expected**: Test data files are not in the repository. Tests looking for files in `data/` or `Tests/data/` will fail. Use `-k "not filename"` to skip these tests or focus on tests that don't require data.

### 4. Tox configuration error
**Known Issue**: Tox shows "SetupCfg failed loading" error. Use pytest directly instead.

### 5. Ruff deprecated settings warning
**Not a problem**: Configuration uses deprecated top-level settings but still works. Auto-fix and checking function normally.

## Code Conventions

### Adding New Features

**New Energy Term**:
1. Create module in `src/pyVertexModel/Kg/` following `kg*.py` pattern
2. Implement energy calculation and gradient
3. Add Cython implementation in `kg_functions.pyx` if performance-critical
4. Rebuild: `python setup.py build_ext --inplace`
5. Add tests in `Tests/test_kg.py`

**Mesh Operations**:
1. Edit code in `src/pyVertexModel/mesh_remodelling/`
2. Ensure geometric consistency
3. Add tests in `Tests/test_flip.py` or `Tests/test_remodelling.py`

### Testing Conventions

- Use pytest assertions
- Use `np.testing.assert_almost_equal()` for floating-point comparisons
- Test files follow pattern `test_*.py`
- Fixtures and utilities in `Tests/tests.py`
- Test data loaded with `load_data()` helper (expects `.mat` files)

### Code Style

- PEP 8 naming conventions
- Numpy docstring format for scientific functions
- Use type hints where appropriate
- Performance-critical code goes in `.pyx` files
- Numerical computations use numpy arrays extensively

## Quick Command Reference

```bash
# Complete setup from scratch
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .

# Lint code
ruff check src/ Tests/ --fix

# Run tests (skip data-dependent ones)
pytest Tests/ -k "not filename" -v

# Run specific test
pytest Tests/test_utils.py::TestUtils::test_assert_matrix -v

# Clean build artifacts
rm -rf build/ src/pyVertexModel/Kg/*.so src/pyVertexModel/Kg/*.c

# Rebuild after .pyx changes
python setup.py build_ext --inplace
```

## Key Dependencies

- **numpy, scipy**: Numerical computing
- **cython**: Performance optimization (C extensions)
- **vtk, pyvista**: 3D visualization
- **matplotlib, seaborn**: Plotting
- **networkx**: Graph operations for mesh topology
- **pandas**: Data manipulation
- **scikit-image, scikit-learn**: Image processing and ML
- **pytest, pytest-cov**: Testing and coverage

## Trust These Instructions

These instructions are based on thorough testing of the build and test process. If you encounter issues not covered here:
1. Check if you followed the exact command sequence
2. Verify all dependencies are installed
3. Ensure Cython extensions are built
4. Only then search for additional information

For repository-specific questions, create an issue or contact the maintainers (see README).
