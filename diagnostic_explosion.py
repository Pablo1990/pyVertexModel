#!/usr/bin/env python
"""
Diagnostic script to investigate energy terms and geometry at explosion point
"""

from Tests.tests import load_data
from src.pyVertexModel.algorithm.newtonRaphson import gGlobal
import numpy as np
import sys

def analyze_explosion():
    vModel = load_data('vertices_going_wild.pkl')
    vModel.set.dt_tolerance = 0.25
    
    print('=== GRADIENT EXPLOSION DIAGNOSTIC ===')
    print()
    
    # Run steps and capture the explosion
    for step in range(12):
        # Get energy before step
        g, energies = gGlobal(vModel.geo_0, vModel.geo_n, vModel.geo, vModel.set)
        gr = np.linalg.norm(g[vModel.Dofs.Free])
        
        print(f'Step {step+1:2d}: gr={gr:.6e}, dt/dt0={vModel.set.dt/vModel.set.dt0:.4f}')
        
        # Check if this is approaching explosion (gr > 0.05)
        if gr > 0.05 or step == 9:
            print(f'  ** DETAILED ANALYSIS **')
            total_energy = sum(energies.values())
            print(f'  Total energy: {total_energy:.6e}')
            print(f'  Energy breakdown:')
            for key, val in sorted(energies.items(), key=lambda x: -x[1]):
                pct = (val / total_energy * 100) if total_energy > 0 else 0
                print(f'    {key:20s}: {val:12.6e} ({pct:6.2f}%)')
            
            # Check for problematic cells
            print(f'  Cell geometry check:')
            for cell in vModel.geo.Cells:
                if cell.AliveStatus:
                    if cell.Vol > 1e-3 or cell.Vol < 1e-10:
                        print(f'    Cell {cell.ID}: Vol={cell.Vol:.6e} (ABNORMAL)')
                    if cell.Area > 1e-2 or cell.Area < 1e-10:
                        print(f'    Cell {cell.ID}: Area={cell.Area:.6e} (ABNORMAL)')
            print()
        
        # Take step
        vModel.single_iteration()
        
        if vModel.didNotConverge:
            print('  STOPPED: didNotConverge')
            break
        
        # Check if gradient exploded
        if step > 0:
            g_after, _ = gGlobal(vModel.geo_0, vModel.geo_n, vModel.geo, vModel.set)
            gr_after = np.linalg.norm(g_after[vModel.Dofs.Free])
            if gr_after > gr * 10:
                print(f'  *** EXPLOSION DETECTED: {gr:.6e} -> {gr_after:.6e} ***')
                break

if __name__ == '__main__':
    analyze_explosion()
