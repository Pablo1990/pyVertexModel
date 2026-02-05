#!/usr/bin/env python
"""
Focused diagnostic: Compare stable step (3) vs explosion step (10)
"""

from Tests.tests import load_data
from src.pyVertexModel.algorithm.newtonRaphson import gGlobal
import numpy as np

def analyze_step(vModel, step_num):
    """Analyze a single step in detail"""
    g, energies = gGlobal(vModel.geo_0, vModel.geo_n, vModel.geo, vModel.set)
    gr = np.linalg.norm(g[vModel.Dofs.Free])
    
    # Cell metrics
    volumes = [c.Vol for c in vModel.geo.Cells if c.AliveStatus]
    areas = [c.Area for c in vModel.geo.Cells if c.AliveStatus]
    
    return {
        'step': step_num,
        'gr': gr,
        'energies': energies,
        'total_energy': sum(energies.values()),
        'vol_min': min(volumes),
        'vol_max': max(volumes),
        'vol_mean': np.mean(volumes),
        'area_min': min(areas),
        'area_max': max(areas),
        'area_mean': np.mean(areas),
        'dt_ratio': vModel.set.dt / vModel.set.dt0
    }

def main():
    vModel = load_data('vertices_going_wild.pkl')
    vModel.set.dt_tolerance = 0.25
    
    print('=== COMPARING STABLE VS EXPLOSION STEPS ===')
    print()
    
    snapshots = []
    
    # Run to step 10 and capture key snapshots
    for step in range(11):
        # Capture state
        if step in [2, 8, 9]:  # Stable, pre-explosion, explosion
            snapshot = analyze_step(vModel, step + 1)
            snapshots.append(snapshot)
        
        # Take step
        vModel.single_iteration()
        
        if vModel.didNotConverge:
            break
    
    # Compare snapshots
    for snap in snapshots:
        print(f"Step {snap['step']}:")
        print(f"  Gradient norm: {snap['gr']:.6e}")
        print(f"  dt/dt0: {snap['dt_ratio']:.4f}")
        print(f"  Total energy: {snap['total_energy']:.6e}")
        print(f"  Volume range: [{snap['vol_min']:.6e}, {snap['vol_max']:.6e}], mean={snap['vol_mean']:.6e}")
        print(f"  Area range: [{snap['area_min']:.6e}, {snap['area_max']:.6e}], mean={snap['area_mean']:.6e}")
        print(f"  Energy composition:")
        for key, val in sorted(snap['energies'].items(), key=lambda x: -x[1]):
            pct = (val / snap['total_energy'] * 100) if snap['total_energy'] > 0 else 0
            print(f"    {key:20s}: {val:12.6e} ({pct:5.2f}%)")
        print()
    
    # Check for volume explosions
    print("=== KEY FINDINGS ===")
    if len(snapshots) >= 3:
        vol_ratio = snapshots[-1]['vol_max'] / snapshots[0]['vol_max']
        print(f"Volume explosion ratio: {vol_ratio:.2f}x")
        
        energy_ratios = {}
        for key in snapshots[0]['energies']:
            ratio = snapshots[-1]['energies'][key] / snapshots[0]['energies'][key] if snapshots[0]['energies'][key] > 0 else 0
            energy_ratios[key] = ratio
        
        print("Energy term growth (explosion/stable):")
        for key, ratio in sorted(energy_ratios.items(), key=lambda x: -x[1]):
            print(f"  {key:20s}: {ratio:8.2f}x")

if __name__ == '__main__':
    main()
