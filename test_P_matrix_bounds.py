#!/usr/bin/env python
"""
Test script to check P matrix bounds in the Risk_LTL_Planning system.
This script demonstrates how to use the bounds checking functions.
"""

import numpy as np
from abstraction.abstraction import Abstraction

def test_P_matrix_bounds():
    """Test the P matrix bounds checking functions"""
    
    # Set up the same configuration as in the main test
    region_size = (20, 20)
    region_res = (5, 5)
    max_speed = 80
    speed_res = 10
    
    # Define label function
    SPEED_LOW_BOUND = 10
    speed_limit = 30
    label_func = {
        (15, 20, 5, 10, 0, max_speed): "t",
        (5, 20, 5, 10, speed_limit, max_speed): "o",   # "o" for "overspeed"
        (0, 20, 0, 20, 0, SPEED_LOW_BOUND): "s"    # "s" for "slow"
    }
    
    # Initial ego state
    ego_state = np.array([2.5, 12.5, np.pi / 2, 0])
    
    # Create abstraction model
    print("Creating abstraction model...")
    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func)
    
    print(f"State space: {len(abs_model.state_set)} states")
    print(f"Action space: {len(abs_model.action_set)} actions")
    print(f"State bounds: x=[0,{region_size[0]//region_res[0]-1}], y=[0,{region_size[1]//region_res[1]-1}], v=[0,{max_speed//speed_res-1}]")
    print()
    
    # Test 1: Check P matrix for invalid transitions
    print("=" * 60)
    print("TEST 1: Checking P Matrix for Invalid Transitions")
    print("=" * 60)
    
    result1 = abs_model.check_P_matrix_bounds()
    print(f"Result: {result1['summary']}")
    
    if not result1['valid']:
        print(f"Found {len(result1['invalid_transitions'])} invalid transitions")
        print("This indicates a bug in the transition function!")
    
    print()
    
    # Test 2: Check state+action simple addition
    print("=" * 60)
    print("TEST 2: Checking State+Action Simple Addition")
    print("=" * 60)
    
    result2 = abs_model.check_state_action_simple_addition()
    print(f"Result: {result2['summary']}")
    
    if not result2['valid']:
        print(f"Found {len(result2['problematic_combinations'])} problematic combinations")
        print("This explains why the LP solver gets out-of-bounds target states!")
    
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if result1['valid'] and result2['valid']:
        print("✅ All checks passed - no bounds issues found")
    elif result1['valid'] and not result2['valid']:
        print("⚠️  P matrix is valid, but LP solver target calculation has issues")
        print("   This is the likely cause of the '[0, 2, -1] is not in list' error")
    elif not result1['valid'] and result2['valid']:
        print("❌ P matrix has invalid transitions - transition function bug!")
    else:
        print("❌ Both P matrix and target calculation have issues")
    
    return result1, result2

if __name__ == '__main__':
    test_P_matrix_bounds() 