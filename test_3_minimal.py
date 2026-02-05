#!/usr/bin/env python
"""
Run a minimal version of test 3 to see where the time is spent
"""
import sys
sys.path.insert(0, '/home/runner/work/pyVertexModel/pyVertexModel')

from Tests.tests import load_data
import time

print("Loading test 3 data...")
vModel = load_data('vertices_going_wild_3.pkl')

# Set up test conditions
vModel.set.tend = vModel.t + 5 * vModel.set.dt0  # Only 5 steps instead of 20
vModel.set.dt_tolerance = 0.25

print(f"Initial state:")
print(f"  Step: {vModel.numStep}")
print(f"  Time: {vModel.t:.6f}")
print(f"  Target time: {vModel.set.tend:.6f}")
print(f"  Will run 5 timesteps")

print(f"\nStarting simulation...")
start_time = time.time()

try:
    vModel.iterate_over_time()
    elapsed = time.time() - start_time
    
    print(f"\nCompleted!")
    print(f"  Final step: {vModel.numStep}")
    print(f"  Final time: {vModel.t:.6f}")
    print(f"  Final dt/dt0: {vModel.set.dt/vModel.set.dt0:.6f}")
    print(f"  Did not converge: {vModel.didNotConverge}")
    print(f"  Elapsed time: {elapsed:.1f} seconds")
    print(f"  Time per step: {elapsed/5:.1f} seconds")
    
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n\nInterrupted after {elapsed:.1f} seconds")
    print(f"  Current step: {vModel.numStep}")
    print(f"  Current time: {vModel.t:.6f}")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n\nError after {elapsed:.1f} seconds: {e}")
    import traceback
    traceback.print_exc()
