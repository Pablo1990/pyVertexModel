# GitHub Copilot Instructions for pyVertexModel

## Project Overview

pyVertexModel is a versatile 3D vertex model for modeling tissue and cell mechanics, particularly for tissue repair simulations. This is a scientific computing project written primarily in Python with performance-critical components in Cython.

## Project Structure

- `src/pyVertexModel/`: Main source code directory
  - `algorithm/`: Core vertex model algorithms (VertexModelVoronoi3D, Newton-Raphson solver)
  - `Kg/`: Energy and force calculations (contractility, volume, viscosity, substrate interactions)
  - `geometry/`: Geometric data structures (cells, faces, vertices)
  - `mesh_remodelling/`: Mesh operations (flips, remodelling)
  - `parameters/`: Configuration and parameter sets
  - `util/`: Utility functions
- `Tests/`: Test suite using pytest
- `Input/`: Input data files
- `resources/`: Additional resources

## Development Setup

### Environment Setup
```bash
conda create -n pyVertexModel python=3.9
conda activate pyVertexModel
pip install -r requirements.txt
```

### First-Time Build
The project uses Cython for performance-critical code. Before running, compile Cython extensions:
```bash
python setup.py build_ext --inplace
```

### Running the Application
```bash
python main.py
```

## Testing

- **Test Framework**: pytest with tox for multi-environment testing
- **Test Location**: `Tests/` directory
- **Running Tests**: 
  ```bash
  tox  # Run full test suite across Python versions and platforms
  pytest  # Run tests directly
  pytest --cov --cov-report=lcov  # Run with coverage
  ```
- **Test Environments**: Python 3.9 and 3.10 on Linux, macOS, and Windows
- **Coverage**: Project uses coverage.py with reporting to coveralls.io and codecov.io

## Code Standards

### Linting and Formatting
- **Tool**: Ruff (modern Python linter and formatter)
- **Target Version**: Python 3.9+
- **Configuration**: See `[tool.ruff]` in `pyproject.toml`
- **Auto-fix**: Enabled by default

### Important Rules
- Follow PEP 8 naming conventions
- Use type hints where appropriate
- Security checks are enabled (S rules in Ruff)
- Import sorting is enforced (I rules)
- Complexity checks are enabled (C, PLR rules)

### Code Style Notes
- Use numpy docstring format for scientific functions
- Performance-critical code should be in Cython (`.pyx` files)
- Numerical computations use numpy arrays extensively
- Test data is loaded from `.mat` files (MATLAB format)

## Dependencies

### Core Dependencies
- **numpy**: Array operations and numerical computing
- **scipy**: Scientific computing algorithms
- **cython**: Performance optimization
- **vtk, pyvista**: 3D visualization
- **networkx**: Graph operations for mesh topology
- **pandas**: Data manipulation
- **scikit-image, scikit-learn**: Image processing and ML utilities
- **matplotlib, seaborn**: Plotting

### Build Dependencies
- setuptools, setuptools-scm for versioning
- wheel for distribution
- cython and numpy for compilation

## Key Conventions

1. **Geometry Objects**: Use custom classes (Cell, Face, Vertex) defined in `geometry/`
2. **Energy Calculations**: Each energy term has its own `kg*` module in `Kg/` directory
3. **Testing**: Tests follow pattern `test_*.py` and use pytest assertions
4. **Data Loading**: Scientific data is in MATLAB `.mat` format, use scipy.io.loadmat
5. **Cython Code**: Performance-critical loops in `kg_functions.pyx` are compiled to C

## Common Tasks

### Adding a New Energy Term
1. Create new module in `src/pyVertexModel/Kg/` following `kg*.py` pattern
2. Implement energy calculation and gradient computation
3. Add corresponding Cython implementation if performance-critical
4. Add tests in `Tests/test_kg.py`

### Modifying Mesh Operations
1. Edit code in `src/pyVertexModel/mesh_remodelling/`
2. Ensure geometric consistency is maintained
3. Add tests in `Tests/test_flip.py` or `Tests/test_remodelling.py`

### Adding Tests
1. Create or modify `Tests/test_*.py` files
2. Use pytest fixtures from `Tests/tests.py`
3. Load test data with `load_data()` helper
4. Use `np.testing.assert_almost_equal()` for floating-point comparisons

## Important Notes

- **Version Control**: Check the README for information about stable branches (historically the `paper` branch has contained polished versions for publications)
- **Results**: Simulations are saved to `Result/` directory (gitignored) with timestamp-based names
- **Parameters**: Model parameters are configured in `parameters/set.py`
- **Performance**: Profile code using Cython's profiling directive before optimizing
- **CI/CD**: GitHub Actions workflows in `.github/workflows/` run tests on push

## Debugging Tips

- Enable Cython profiling in `setup.py` (already configured)
- Use `annotate=True` in Cython compilation to generate HTML reports
- Check numerical stability with smaller time steps when debugging convergence
- Visualization with PyVista can help debug geometric issues

## Contact

For issues with the code, create an issue in the repository. Check the README or repository settings for current contact information.
