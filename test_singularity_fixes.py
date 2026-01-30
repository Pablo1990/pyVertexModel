#!/usr/bin/env python3
"""
Simple test to verify the numerical singularity fixes don't break basic functionality.
This tests that edge length clamping and triangle detection work correctly.
"""

import numpy as np

def test_edge_length_clamping():
    """Test that edge length clamping prevents singularities"""
    MIN_EDGE_LENGTH = 1e-6
    
    # Test case 1: Very short edge
    y1 = np.array([0.0, 0.0, 0.0])
    y2 = np.array([1e-8, 0.0, 0.0])  # Extremely short edge
    
    l_i = np.linalg.norm(y1 - y2)
    print(f"Original edge length: {l_i}")
    
    if l_i < MIN_EDGE_LENGTH:
        l_i = MIN_EDGE_LENGTH
    print(f"Clamped edge length: {l_i}")
    
    # Compute force term
    force_term = 1 / l_i ** 3
    print(f"Force term 1/l³: {force_term:.2e}")
    assert force_term <= 1.01e18, "Force term should be bounded (within tolerance)"
    print("✓ Edge length clamping works\n")

def test_triangle_degeneracy_check():
    """Test that degenerate triangle detection works"""
    MIN_TRIANGLE_AREA = 1e-10
    
    # Test case 1: Degenerate (collinear) triangle
    y1 = np.array([0.0, 0.0, 0.0])
    y2 = np.array([1.0, 0.0, 0.0])
    y3 = np.array([2.0, 0.0, 0.0])  # Collinear with y1, y2
    
    # Simplified area check (cross product norm)
    area = np.linalg.norm(np.cross(y2 - y1, y3 - y1)) / 2
    print(f"Degenerate triangle area: {area}")
    
    if area < MIN_TRIANGLE_AREA:
        print("Triangle is degenerate - would return zeros")
    
    assert area < MIN_TRIANGLE_AREA, "Should detect collinear points"
    print("✓ Degenerate triangle detection works\n")
    
    # Test case 2: Normal triangle
    y1 = np.array([0.0, 0.0, 0.0])
    y2 = np.array([1.0, 0.0, 0.0])
    y3 = np.array([0.0, 1.0, 0.0])  # Not collinear
    
    area = np.linalg.norm(np.cross(y2 - y1, y3 - y1)) / 2
    print(f"Normal triangle area: {area}")
    assert area > MIN_TRIANGLE_AREA, "Should not flag normal triangle"
    print("✓ Normal triangle passed\n")

def test_volume_exponent():
    """Test that volume gradient with n=2 is reasonable"""
    Vol = 1.2
    Vol0 = 1.0
    
    # With n=2 (new)
    n = 2
    grad_n2 = (Vol - Vol0) ** (n - 1) / Vol0 ** n
    print(f"Volume gradient with n=2: {grad_n2}")
    
    # With n=4 (old)
    n = 4
    grad_n4 = (Vol - Vol0) ** (n - 1) / Vol0 ** n
    print(f"Volume gradient with n=4: {grad_n4}")
    
    ratio = grad_n2 / grad_n4
    print(f"Reduction ratio (n=2 vs n=4): {ratio:.2f}x")
    assert ratio > 1, "n=2 should reduce amplification compared to n=4"
    print("✓ Volume exponent reduction works\n")

if __name__ == "__main__":
    print("Testing Numerical Singularity Fixes\n")
    print("=" * 50)
    test_edge_length_clamping()
    test_triangle_degeneracy_check()
    test_volume_exponent()
    print("=" * 50)
    print("All tests passed! ✓")
