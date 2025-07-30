#!/usr/bin/env python
"""
Test script to check product automaton P_matrix bounds in the Risk_LTL_Planning system.
This script demonstrates how to use the product automaton bounds checking functions.
"""

import numpy as np
from abstraction.abstraction import Abstraction
from abstraction.MDP import MDP
from abstraction.prod_MDP import Prod_MDP
from specification.prod_auto import Product
from specification.specification import LTL_Spec

def test_product_automaton_bounds():
    """Test the product automaton bounds checking functions"""
    
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
    
    # Create trivial environment MDP
    state_set = [0]
    action_set = [0]
    transitions = np.array([[[1.0]]])  # Stay in state 0
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, ["none"], initial_state)
    
    # Create product MDP
    mdp_sys = abs_model.MDP
    mdp_prod = Prod_MDP(mdp_sys, mdp_env)
    
    # Create LTL specifications
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])
    
    # Create product automaton
    print("Creating product automaton...")
    prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
    
    print(f"MDP states: {len(mdp_sys.states)}")
    print(f"Product MDP states: {len(mdp_prod.MDP.states)}")
    print(f"Product automaton states: {len(prod_auto.prod_state_set)}")
    print(f"Product automaton actions: {len(prod_auto.prod_action_set)}")
    print()
    
    # Test 1: Check product state consistency
    print("=" * 60)
    print("TEST 1: Checking Product State Consistency")
    print("=" * 60)
    
    consistency_result = prod_auto.check_product_state_consistency()
    print(f"Result: {consistency_result['summary']}")
    
    if not consistency_result['consistent']:
        print(f"Found {len(consistency_result['inconsistent_states'])} inconsistent product states")
        print("This indicates a problem with product state construction!")
    
    print()
    
    # Test 2: Check product automaton P_matrix bounds
    print("=" * 60)
    print("TEST 2: Checking Product Automaton P_Matrix Bounds")
    print("=" * 60)
    
    bounds_result = prod_auto.check_product_P_matrix_bounds(abs_model, mdp_prod)
    print(f"Result: {bounds_result['summary']}")
    
    if not bounds_result['valid']:
        print(f"Found {len(bounds_result['invalid_transitions'])} invalid transitions")
        print("This indicates a problem with product automaton transition generation!")
    
    print()
    
    # Test 2b: Check product automaton mdp_sys bounds
    print("=" * 60)
    print("TEST 2b: Checking Product Automaton MDP_sys Bounds")
    print("=" * 60)
    
    mdp_sys_bounds_result = prod_auto.check_product_mdp_sys_bounds(abs_model, mdp_prod)
    print(f"Result: {mdp_sys_bounds_result['summary']}")
    
    if not mdp_sys_bounds_result['valid']:
        print(f"Found {len(mdp_sys_bounds_result['invalid_sys_transitions'])} invalid mdp_sys transitions")
        print("This indicates out-of-bounds mdp_sys states in the product automaton!")
    
    print()
    
    # Test 3: Check underlying abstraction bounds (for comparison)
    print("=" * 60)
    print("TEST 3: Checking Underlying Abstraction Bounds (for comparison)")
    print("=" * 60)
    
    # Use the check functions from the abstraction (if they exist)
    try:
        abs_p_matrix_result = abs_model.check_P_matrix_bounds()
        abs_state_action_result = abs_model.check_state_action_simple_addition()
        
        print(f"Abstraction P_matrix: {abs_p_matrix_result['summary']}")
        print(f"Abstraction state+action: {abs_state_action_result['summary']}")
    except AttributeError:
        print("Abstraction bounds checking functions not available")
    
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if consistency_result['consistent'] and bounds_result['valid'] and mdp_sys_bounds_result['valid']:
        print("✅ All product automaton checks passed - no issues found")
    else:
        issues = []
        if not consistency_result['consistent']:
            issues.append("Product state consistency issues")
        if not bounds_result['valid']:
            issues.append("Product P_matrix bounds issues")
        if not mdp_sys_bounds_result['valid']:
            issues.append("MDP_sys bounds issues")
        
        print(f"⚠️  Product automaton issues found: {', '.join(issues)}")
        print("   These could indicate problems with the product automaton construction")
    
    return consistency_result, bounds_result, mdp_sys_bounds_result

if __name__ == '__main__':
    test_product_automaton_bounds() 