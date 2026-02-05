#!/usr/bin/env python
"""
Debug script to understand why test_vertices_shouldnt_be_going_wild_3 fails
"""
import sys
sys.path.insert(0, '/home/runner/work/pyVertexModel/pyVertexModel')

from Tests.tests import load_data
import numpy as np

# Load all three test geometries
vModel_1 = load_data('vertices_going_wild.pkl')
vModel_2 = load_data('vertices_going_wild_2.pkl')  
vModel_3 = load_data('vertices_going_wild_3.pkl')

print("="*80)
print("COMPARING THREE TEST GEOMETRIES")
print("="*80)

for name, vModel in [("Test 1 (passes)", vModel_1), ("Test 2 (passes)", vModel_2), ("Test 3 (FAILS)", vModel_3)]:
    print(f"\n{name}:")
    print(f"  Time: {vModel.t:.6f}")
    print(f"  Step: {vModel.numStep}")
    print(f"  dt: {vModel.set.dt:.6e}")
    print(f"  dt0: {vModel.set.dt0:.6e}")
    print(f"  dt/dt0: {vModel.set.dt/vModel.set.dt0:.6f}")
    print(f"  tol: {vModel.set.tol:.6e}")
    print(f"  nu: {vModel.set.nu:.6e}")
    print(f"  nu0: {vModel.set.nu0:.6e}")
    print(f"  Num cells: {vModel.geo.nCells}")
    print(f"  Num vertices: {vModel.geo.numY}")
    print(f"  Num faces: {vModel.geo.numF}")
    
    # Check if any cells have unusual properties
    volumes = [cell.Vol for cell in vModel.geo.Cells if cell.AliveStatus is not None]
    areas = [sum([face.Area for face in cell.Faces]) for cell in vModel.geo.Cells if cell.AliveStatus is not None]
    
    print(f"  Volume range: [{min(volumes):.6e}, {max(volumes):.6e}]")
    print(f"  Area range: [{min(areas):.6e}, {max(areas):.6e}]")
    print(f"  Mean volume: {np.mean(volumes):.6e}")
    print(f"  Mean area: {np.mean(areas):.6e}")
    print(f"  Volume std: {np.std(volumes):.6e}")
    print(f"  Area std: {np.std(areas):.6e}")

    # Check geometry Y matrix for any unusual values
    try:
        Y = vModel.geo.Y
        print(f"  Y matrix shape: {Y.shape}")
        print(f"  Y min/max: [{Y.min():.6f}, {Y.max():.6f}]")
        print(f"  Y mean: {Y.mean():.6f}")
        print(f"  Y std: {Y.std():.6f}")
    except:
        print(f"  Could not access Y matrix")

print("\n" + "="*80)
print("KEY DIFFERENCE ANALYSIS")
print("="*80)

# Check what's significantly different in test 3
for attr in ['dt', 'dt0', 'nu', 'nu0', 't', 'numStep']:
    val1 = getattr(vModel_1.set if hasattr(vModel_1.set, attr) else vModel_1, attr)
    val2 = getattr(vModel_2.set if hasattr(vModel_2.set, attr) else vModel_2, attr)
    val3 = getattr(vModel_3.set if hasattr(vModel_3.set, attr) else vModel_3, attr)
    
    if val1 != val3 or val2 != val3:
        print(f"\n{attr}:")
        print(f"  Test 1: {val1}")
        print(f"  Test 2: {val2}")  
        print(f"  Test 3: {val3}")

# Check if test 3 is already at a reduced dt
print(f"\nTest 3 starting dt/dt0: {vModel_3.set.dt/vModel_3.set.dt0:.6f}")
print(f"Test will run for 20 timesteps from step {vModel_3.numStep}")
print(f"Target time: {vModel_3.t + 20 * vModel_3.set.dt0:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Test 3 has:")
print(f"  - More vertices: {vModel_3.geo.numY} vs {vModel_2.geo.numY} (test 2)")
print(f"  - More faces: {vModel_3.geo.numF} vs {vModel_2.geo.numF} (test 2)")
print(f"  - Later in simulation: step {vModel_3.numStep} vs {vModel_2.numStep} (test 2)")
print(f"  - All starts at dt/dt0 = 1.0")
print(f"\nThe failing test may have a more complex/remodeled geometry that's harder to converge.")
