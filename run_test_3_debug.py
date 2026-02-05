#!/usr/bin/env python
"""
Run test 3 with diagnostic logging to see where it fails
"""
import sys
sys.path.insert(0, '/home/runner/work/pyVertexModel/pyVertexModel')

from Tests.tests import load_data
import numpy as np

print("Loading test 3 data...")
vModel_test = load_data('vertices_going_wild_3.pkl')

# Set up test conditions
vModel_test.set.tend = vModel_test.t + 20 * vModel_test.set.dt0
vModel_test.set.dt_tolerance = 0.25

print(f"Initial state:")
print(f"  Step: {vModel_test.numStep}")
print(f"  Time: {vModel_test.t:.6f}")
print(f"  Target time: {vModel_test.set.tend:.6f}")
print(f"  dt: {vModel_test.set.dt:.6e}")
print(f"  dt0: {vModel_test.set.dt0:.6e}")
print(f"  dt/dt0: {vModel_test.set.dt/vModel_test.set.dt0:.6f}")
print(f"  dt_tolerance: {vModel_test.set.dt_tolerance:.6f}")
print(f"  tol: {vModel_test.set.tol:.6e}")
print(f"  integrator: {getattr(vModel_test.set, 'integrator', 'euler')}")

print(f"\nRunning simulation for 20 timesteps...")
print(f"Will monitor dt/dt0 and stop if it drops below {vModel_test.set.dt_tolerance}")

# Save original iterate function to wrap it
original_iterate = vModel_test.iterate_over_time

# Track steps
step_count = 0
initial_step = vModel_test.numStep

def monitored_iterate():
    global step_count
    
    # Do one step
    try:
        vModel_test.iterate_over_time_single_step()
        step_count += 1
        
        dt_ratio = vModel_test.set.dt / vModel_test.set.dt0
        print(f"Step {vModel_test.numStep} (#{step_count}/20): t={vModel_test.t:.6f}, dt/dt0={dt_ratio:.6f}", end="")
        
        if dt_ratio < vModel_test.set.dt_tolerance:
            print(f" ❌ FAILED - hit dt_tolerance!")
            return False
        elif vModel_test.didNotConverge:
            print(f" ❌ FAILED - didNotConverge!")
            return False
        else:
            print(f" ✓")
            return True
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run step by step
try:
    while vModel_test.t < vModel_test.set.tend and step_count < 20:
        if not monitored_iterate():
            break
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT:")
    print(f"{'='*80}")
    print(f"Steps completed: {step_count}/20")
    print(f"Final step: {vModel_test.numStep}")
    print(f"Final time: {vModel_test.t:.6f}")
    print(f"Final dt/dt0: {vModel_test.set.dt/vModel_test.set.dt0:.6f}")
    print(f"Did not converge: {vModel_test.didNotConverge}")
    
    if not vModel_test.didNotConverge and step_count >= 20:
        print(f"\n✅ TEST PASSED!")
    else:
        print(f"\n❌ TEST FAILED!")
        if vModel_test.didNotConverge:
            print(f"   Reason: didNotConverge = True")
        if step_count < 20:
            print(f"   Reason: Only {step_count}/20 steps completed")
            
except KeyboardInterrupt:
    print(f"\n\nInterrupted at step {step_count}")
except Exception as e:
    print(f"\n\nError: {e}")
    import traceback
    traceback.print_exc()
