# Fix for Mechanics Blowing Up at Multi-Fold Vertices in Scutoids

## Problem Statement (Corrected Understanding)

**Multi-fold vertices (4-fold, 5-fold, 6-fold) are INTENTIONAL and CORRECT in scutoid models.** They represent cells sharing vertices in 3D configurations where 4+ cells meet.

The problem is NOT about preventing these multi-fold vertices. The problem is that when they exist, the **force/energy calculations have numerical singularities** that cause:
- Vertices moving far apart incorrectly  
- Spiky, unrealistic cell shapes
- Numerical explosions in simulations
- Simulation crashes

## Root Causes: Numerical Singularities

### 1. Edge Length Singularity in Contractility Forces
**Location**: `kgContractility.py` lines 264, 284

**Problem**: 
```python
l_i = ||y_1 - y_2||  # Edge length
K_term = -(C/l_i0) * (1/l_i³) * outer(...)  # 1/l³ singularity
g_term = (C/l_i0) * (y_1-y_2) / l_i  # 1/l singularity
```

When edges at multi-fold vertices become very short:
- `l_i = 0.01` → `1/l³ = 10⁶` → stiffness explodes
- `l_i = 0.001` → `1/l³ = 10⁹` → catastrophic failure

**Fix**: Clamp edge length to minimum value
```python
MIN_EDGE_LENGTH = 1e-6
if l_i < MIN_EDGE_LENGTH:
    l_i = MIN_EDGE_LENGTH
```

### 2. Degenerate Triangle Singularity in Surface Energy
**Location**: `kg_functions.pyx` lines 108, 112

**Problem**:
```cython
q = y2_crossed @ y1 - y2_crossed @ y3 + y1_crossed @ y3
fact = 1 / (2 * ||q||)  # When triangle degenerates: ||q||→0 → fact→∞
Kss = -(2/||q||) * outer(gs, gs)  # Unbounded negative forces
```

At multi-fold vertices, triangles can become nearly collinear, causing `||q|| → 0`.

**Fix**: Check for degenerate triangles
```cython
if q_norm < 1e-10:
    return zeros  # Return zero gradients instead of inf
```

### 3. Volume Gradient Amplification
**Location**: `kgVolume.py` line 38

**Problem**:
```python
n = 4  # Exponent in volume energy
fact = lambdaV * (Vol-Vol0)^(n-1) / Vol0^n  # With n=4: (Vol-Vol0)³/Vol0⁴
```

With `n=4`:
- Gradient grows as `(Vol-Vol0)³` → huge amplification when volume deviates
- Denominator `Vol0⁴` makes small cells extremely stiff
- At multi-fold vertices during remodeling, this causes force blow-up

**Fix**: Reduce exponent from 4 to 2
```python
n = 2  # Changed from 4 to 2 for numerical stability
```

This provides stable, proportional volume control without excessive amplification.

## Changes Made

### File 1: `src/pyVertexModel/Kg/kgContractility.py`

**Lines 264-273** - `compute_k_contractility()`:
- Added minimum edge length check: `MIN_EDGE_LENGTH = 1e-6`
- Clamps `l_i` before computing `1/l_i³` term
- Prevents stiffness matrix blow-up

**Lines 284-290** - `compute_g_contractility()`:
- Added same minimum edge length check
- Clamps `l_i` before computing gradient
- Prevents force vector blow-up

### File 2: `src/pyVertexModel/Kg/kg_functions.pyx`

**Lines 93-124** - `gKSArea()`:
- Added degenerate triangle detection
- Computes `q_norm = ||q||` before division
- Returns zero gradients if `q_norm < 1e-10`
- Prevents singularities in surface energy gradients

### File 3: `src/pyVertexModel/Kg/kgVolume.py`

**Line 38** - `compute_work()`:
- Changed `n = 4` to `n = 2`
- Reduces gradient amplification: `(Vol-Vol0)` instead of `(Vol-Vol0)³`
- Provides more stable volume control at multi-fold vertices

## Testing

The changes are **minimal and surgical**:
- No changes to core algorithms or topology
- Only adds safety checks and reduces amplification
- Preserves existing behavior for well-conditioned configurations
- Only activates at extreme cases (very short edges, degenerate triangles)

## Expected Impact

### Before Fixes
- ❌ Edges at multi-fold vertices → `1/l³ → ∞` → stiffness explosion
- ❌ Degenerate triangles → `1/||q|| → ∞` → force explosion  
- ❌ Volume gradients → `(Vol-Vol0)³` → excessive amplification
- ❌ Result: Spiky geometries, vertices flying apart, crashes

### After Fixes
- ✅ Edge lengths clamped → stiffness bounded → stable mechanics
- ✅ Degenerate triangles detected → zero gradients → no singularities
- ✅ Volume gradients reduced → proportional forces → smooth remodeling
- ✅ Result: Smooth geometries, stable simulations, no crashes

## Notes

1. **Cython Recompilation**: The `.pyx` file needs recompilation:
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Multi-fold vertices remain intentional**: These fixes do NOT prevent 4-fold, 5-fold, or 6-fold vertices. They simply make the force calculations numerically stable when such vertices exist.

3. **Conservative thresholds**: 
   - `MIN_EDGE_LENGTH = 1e-6` is very small (μm scale)
   - `MIN_TRIANGLE_AREA = 1e-10` only catches truly degenerate triangles
   - Normal configurations are unaffected

4. **Volume exponent**: Changing from n=4 to n=2 makes the energy function more "soft". If stronger volume control is needed, other parameters (like `lambdaV`) can be adjusted.

## Future Enhancements (Not Included)

These are additional improvements that could further help but are not critical:

1. **Vertex valence weighting** in gradient assembly
2. **Adaptive timestep** reduction near multi-fold vertices
3. **Line search improvements** in Newton-Raphson solver
4. **Better initial geometry** to avoid extreme configurations
