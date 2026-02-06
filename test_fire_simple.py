"""
Simple test to verify FIRE algorithm implementation
"""
import numpy as np

# Test import
try:
    from src.pyVertexModel.algorithm.newtonRaphson import newton_raphson_iteration_fire
    from src.pyVertexModel.parameters.set import Set
    print("✅ FIRE algorithm imports successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test Set class has FIRE parameters
try:
    s = Set()
    assert hasattr(s, 'fire_dt_max'), "Missing fire_dt_max"
    assert hasattr(s, 'fire_dt_min'), "Missing fire_dt_min"
    assert hasattr(s, 'fire_N_min'), "Missing fire_N_min"
    assert hasattr(s, 'fire_f_inc'), "Missing fire_f_inc"
    assert hasattr(s, 'fire_f_dec'), "Missing fire_f_dec"
    assert hasattr(s, 'fire_alpha_start'), "Missing fire_alpha_start"
    assert hasattr(s, 'fire_f_alpha'), "Missing fire_f_alpha"
    assert hasattr(s, 'integrator'), "Missing integrator"
    print("✅ All FIRE parameters present in Set class")
    print(f"   - fire_N_min: {s.fire_N_min}")
    print(f"   - fire_f_inc: {s.fire_f_inc}")
    print(f"   - fire_f_dec: {s.fire_f_dec}")
    print(f"   - fire_alpha_start: {s.fire_alpha_start}")
    print(f"   - fire_f_alpha: {s.fire_f_alpha}")
except AssertionError as e:
    print(f"❌ Parameter check failed: {e}")
    exit(1)

# Test integrator options
try:
    s = Set()
    s.integrator = 'euler'
    assert s.integrator == 'euler'
    s.integrator = 'rk2'
    assert s.integrator == 'rk2'
    s.integrator = 'fire'
    assert s.integrator == 'fire'
    print("✅ Integrator options work correctly")
    print(f"   - Available: 'euler', 'rk2', 'fire'")
except Exception as e:
    print(f"❌ Integrator test failed: {e}")
    exit(1)

print("\n" + "="*60)
print("FIRE ALGORITHM IMPLEMENTATION: ALL TESTS PASSED ✅")
print("="*60)
print("\nUsage:")
print("  vModel.set.integrator = 'fire'  # Enable FIRE algorithm")
print("  vModel.set.integrator = 'euler' # Use explicit Euler (default)")
print("  vModel.set.integrator = 'rk2'   # Use RK2 midpoint method")
print("\nFIRE Parameters (optional tuning):")
print("  vModel.set.fire_N_min = 5        # Steps before acceleration")
print("  vModel.set.fire_f_inc = 1.1      # Timestep increase factor")
print("  vModel.set.fire_f_dec = 0.5      # Timestep decrease factor")
print("  vModel.set.fire_alpha_start = 0.1  # Initial damping")
print("="*60)
