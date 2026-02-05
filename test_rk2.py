#!/usr/bin/env python
"""
Test RK2 implementation against the problematic test case
"""

from Tests.tests import load_data
from src.pyVertexModel.algorithm.newtonRaphson import gGlobal
import numpy as np

def test_rk2_vs_euler():
    """Compare RK2 and Euler integrators on the problematic geometry"""
    
    print('='*80)
    print('TESTING RK2 vs EULER INTEGRATORS')
    print('='*80)
    print()
    
    # Test Euler first
    print('1. Testing EULER integrator:')
    print('-' * 40)
    vModel_euler = load_data('vertices_going_wild.pkl')
    vModel_euler.set.dt_tolerance = 0.25
    vModel_euler.set.integrator = 'euler'
    
    gradients_euler = []
    for step in range(12):
        gr = vModel_euler.single_iteration()
        gradients_euler.append(gr)
        dt_ratio = vModel_euler.set.dt / vModel_euler.set.dt0
        print(f'  Step {step+1:2d}: gr={gr:.6e}, dt/dt0={dt_ratio:.4f}')
        
        if vModel_euler.didNotConverge:
            print('  FAILED: Hit dt_tolerance')
            break
    
    euler_max_gr = max(gradients_euler) if gradients_euler else float('inf')
    euler_steps = len(gradients_euler)
    
    print()
    print('2. Testing RK2 integrator:')
    print('-' * 40)
    vModel_rk2 = load_data('vertices_going_wild.pkl')
    vModel_rk2.set.dt_tolerance = 0.25
    vModel_rk2.set.integrator = 'rk2'
    
    gradients_rk2 = []
    for step in range(12):
        gr = vModel_rk2.single_iteration()
        gradients_rk2.append(gr)
        dt_ratio = vModel_rk2.set.dt / vModel_rk2.set.dt0
        print(f'  Step {step+1:2d}: gr={gr:.6e}, dt/dt0={dt_ratio:.4f}')
        
        if vModel_rk2.didNotConverge:
            print('  FAILED: Hit dt_tolerance')
            break
    
    rk2_max_gr = max(gradients_rk2) if gradients_rk2 else float('inf')
    rk2_steps = len(gradients_rk2)
    
    print()
    print('='*80)
    print('COMPARISON RESULTS:')
    print('='*80)
    print(f'Euler: {euler_steps} steps, max gradient = {euler_max_gr:.6e}')
    print(f'RK2:   {rk2_steps} steps, max gradient = {rk2_max_gr:.6e}')
    print()
    
    if rk2_max_gr < euler_max_gr:
        print('✓ RK2 has better gradient control')
    else:
        print('✗ RK2 gradient control not improved')
    
    if rk2_steps >= euler_steps:
        print('✓ RK2 completed more steps')
    else:
        print('✗ RK2 completed fewer steps')
    
    print()
    return vModel_euler, vModel_rk2

if __name__ == '__main__':
    test_rk2_vs_euler()
