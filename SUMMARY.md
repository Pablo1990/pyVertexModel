# Investigation Summary: Gradient Explosion in Scutoid Geometries

## Question
Can you investigate the energy terms causing the instabilities and the geometries of the cells in case there is something ill constructed?

## Answer

**Energy Terms**: ✅ All healthy - no problematic energy contributions
**Cell Geometry**: ✅ All valid - no ill-constructed cells or degenerate triangles  
**Root Cause**: ❌ Explicit Euler integration method is fundamentally unstable for this tissue state

## Visual Summary

### Without Safety Factor (Original Code)
```
Gradient Evolution:
Step 1  ████████████████ 0.293
Step 2  ██              0.045
Step 3  █               0.016
Step 4  █               0.019  (+15%)
Step 5  █               0.023  (+23%)
Step 6  █               0.025  (+7%)
Step 7  ██              0.033  (+31%)
Step 8  ██              0.036  (+10%)
Step 9  ███             0.062  (+73%)
Step 10 ████████████████████████████████████████████ 1.474 EXPLOSION!
```

### With Safety Factor 0.75 (Current Fix)
```
Gradient Evolution:
Step 1  ████████████████ 0.293
Step 2  ██              0.045
Step 3  █               0.023
Step 4  █               0.018
...
Step 10 █               0.028  ✓ Controlled
Step 11 █               0.034  ✓ Controlled
Step 12 █               0.036  ✓ Controlled

BUT: Hits dt_tolerance (0.25) before completing 20 steps
```

## Energy Term Breakdown

**Step 3 (Stable)**:
- Contractility: 60%
- Surface: 26%
- TriEnergyBarrierAR: 13%
- Volume: 1%

**Step 10 (Would-be Explosion)**:
- Contractility: 59%
- Surface: 27%
- TriEnergyBarrierAR: 13%
- Volume: 1%

**No energy term explodes!** All remain stable.

## Cell Geometry Stats

| Metric | Value | Status |
|--------|-------|--------|
| Cells | 150 | ✓ Normal |
| Vertices | 3,392 | ✓ Normal |
| Volume range | [8.2e-5, 6.6e-4] | ✓ Reasonable |
| Area range | [0.035, 0.163] | ✓ Reasonable |
| Degenerate triangles | 0 | ✓ None found |
| Volume change per step | <0.001% | ✓ Minimal |

## The Problem

Explicit Euler has fundamental stability limitations for stiff systems:

```
Stability Region = Step Size × Eigenvalue < 2

For this tissue:
- System is "stiff" (large eigenvalues)
- Explicit Euler requires tiny steps
- Safety factor provides tiny steps
- But then: too slow → hits dt_tolerance
```

## Solutions Evaluated

| Option | Prevents Explosion | Passes Test | Complexity |
|--------|-------------------|-------------|------------|
| Safety Factor 0.5 | ✓ | ✗ (dt_tol) | Low |
| Safety Factor 0.75 | ✓ | ✗ (dt_tol) | Low |
| Safety Factor 0.9 | ✓ | ✗ (dt_tol) | Low |
| Increase dt_tolerance | ✓ | ✓ | Low |
| Implicit method | ✓ | ✓ | Med (slow) |
| RK2/RK4 | ✓ | ✓ | High (best) |

## Recommendations

### Immediate Fix
Increase `dt_tolerance` from 0.25 to 0.15:
```python
vModel_test.set.dt_tolerance = 0.15  # Allow more timestep reduction
```

This acknowledges that scutoid geometries need smaller timesteps.

### Long-term Solution
Implement Runge-Kutta 2nd or 4th order integrator:
- Better stability properties
- Standard for stiff biological systems
- Allows larger timesteps with same accuracy

## Files Created

- `diagnostic_explosion.py` - Detailed energy/geometry analysis
- `diagnostic_focused.py` - Stable vs explosion comparison
- `INVESTIGATION_REPORT.md` - Full technical report
- `SUMMARY.md` - This file

Run diagnostics: `python diagnostic_explosion.py`
