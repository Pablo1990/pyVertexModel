#!/usr/bin/env python
"""
Quick check of initial gradient for test 3
"""
import sys
sys.path.insert(0, '/home/runner/work/pyVertexModel/pyVertexModel')

from Tests.tests import load_data
from pyVertexModel.algorithm.newtonRaphson import KgGlobal
import numpy as np

print("Loading test 3 data...")
vModel = load_data('vertices_going_wild_3.pkl')

print(f"Computing initial gradient...")
g, K, energy, grb = KgGlobal(vModel.geo_0, vModel.geo_n, vModel.geo, vModel.set)

# Get free dofs
vModel.Dofs.get_dofs(vModel.geo, vModel.set)
gr = np.linalg.norm(g[vModel.Dofs.Free])

print(f"\nInitial state:")
print(f"  Gradient norm (free DOFs): {gr:.6e}")
print(f"  Tolerance: {vModel.set.tol:.6e}")
print(f"  Ratio gr/tol: {gr/vModel.set.tol:.3f}")
print(f"  dt: {vModel.set.dt:.6e}")
print(f"  dt0: {vModel.set.dt0:.6e}")
print(f"  nu: {vModel.set.nu:.6e}")

print(f"\nCurrent adaptive scaling logic:")
if gr > vModel.set.tol:
    SAFETY_FACTOR = 0.5
    scale_factor = max(0.1, min(1.0, vModel.set.tol / gr * SAFETY_FACTOR))
    print(f"  gr > tol: using adaptive scaling")
    print(f"  scale_factor = max(0.1, min(1.0, {vModel.set.tol:.3f} / {gr:.3f} * 0.5))")
    print(f"  scale_factor = {scale_factor:.4f}")
else:
    scale_factor = 0.75
    print(f"  gr <= tol: using fixed conservative scaling")
    print(f"  scale_factor = {scale_factor:.4f}")

print(f"\nEnergy breakdown:")
for key, val in energy.items():
    print(f"  {key}: {val:.6e}")
