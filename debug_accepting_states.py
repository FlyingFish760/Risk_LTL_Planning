#!/usr/bin/env python3

import numpy as np
from abstraction.MDP import MDP
from specification.prod_auto import Product
from specification.specification import LTL_Spec
from abstraction.abstraction import Abstraction
from abstraction.prod_MDP import Prod_MDP

def main():
    # ---------- MDP Environment  ---------------------
    state_set = [0]
    action_set = [0]
    transitions = np.array([[[1.0]]])
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, ["none"], initial_state)

    # ---------- MPD System (Abstraction) --------------------
    region_size = (20, 20)
    region_res = (5, 5)
    max_speed = 80
    speed_res = 10

    speed_limit = 30
    label_func = {(15, 20, 5, 10, 0, max_speed): "t",
                   (5, 20, 5, 10, speed_limit, max_speed): "o"}
    
    ego_state = np.array([2.5, 12.5, np.pi / 2, 0])
    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # ---------- Create Product Automaton --------------------
    mdp_sys = abs_model.MDP
    mdp_prod = Prod_MDP(mdp_sys, mdp_env)
    prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)

    print("=== DEBUGGING: Accepting States Analysis ===")
    
    # Check accepting states
    print(f"Number of accepting states: {len(prod_auto.accepting_states)}")
    print(f"First 10 accepting states: {prod_auto.accepting_states[:10]}")
    
    # Show details of first few accepting states
    print(f"\nFirst 10 accepting states details:")
    for i, acc_state_idx in enumerate(prod_auto.accepting_states[:10]):
        prod_state = prod_auto.prod_state_set[acc_state_idx]
        prod_mdp_state_idx, dfa_cs_state, dfa_safe_state = prod_state
        
        # FIXED: Get the actual abstraction state from product MDP
        if prod_mdp_state_idx < len(mdp_prod.prod_state_set):
            x_sys, x_env = mdp_prod.prod_state_set[prod_mdp_state_idx]
            
            if x_sys < len(abs_model.state_set):
                abs_state_coords = abs_model.state_set[x_sys]
                abs_label = abs_model.MDP.labelling[x_sys]
                print(f"  {i+1}: Accept state {acc_state_idx} = {prod_state}")
                print(f"      prod_mdp_state_idx={prod_mdp_state_idx} -> x_sys={x_sys}, x_env={x_env}")
                print(f"      Abstraction state {x_sys}: coords {abs_state_coords}, label '{abs_label}'")
                print(f"      DFA states: cs={dfa_cs_state}, safe={dfa_safe_state}")
            else:
                print(f"  {i+1}: Accept state {acc_state_idx} = {prod_state} (INVALID x_sys={x_sys})")
        else:
            print(f"  {i+1}: Accept state {acc_state_idx} = {prod_state} (INVALID prod_mdp_state_idx={prod_mdp_state_idx})")
    
    # Check states labeled with 't'
    print(f"\n=== States labeled with 't' ===")
    t_states = []
    for state_idx, label in enumerate(abs_model.MDP.labelling):
        if 't' in label:
            state_coords = abs_model.state_set[state_idx]
            t_states.append((state_idx, state_coords, label))
            
    print(f"Number of states labeled with 't': {len(t_states)}")
    for i, (state_idx, coords, label) in enumerate(t_states):
        print(f"  {i+1}: State {state_idx}: coords {coords}, label '{label}'")

    # Check DFA states for F(t) formula
    print(f"\n=== F(t) DFA Analysis ===")
    print(f"DFA states: {scltl_frag.dfa.states}")
    print(f"DFA initial state: {scltl_frag.dfa.initial_state}")
    print(f"DFA sink states: {scltl_frag.dfa.sink_states}")
    
    # Check DFA transitions
    print(f"\nF(t) DFA transitions:")
    for transition, next_state in scltl_frag.dfa.transitions.items():
        print(f"  {transition} -> {next_state}")
    
    # Check what alphabet 't' maps to
    t_alphabet = scltl_frag.dfa.get_alphabet('t')
    empty_alphabet = scltl_frag.dfa.get_alphabet('_')
    print(f"\nAlphabet mapping:")
    print(f"  't' -> {t_alphabet}")
    print(f"  '_' -> {empty_alphabet}")
    
    # Check which states should be accepting for F(t)
    print(f"\n=== Accepting States Logic ===")
    print(f"According to gen_final_states(), accepting states are where DFA_cs.is_sink_state(str(s_cs)) is True")
    print(f"F(t) DFA sink states: {scltl_frag.dfa.sink_states}")
    
    # Check if states labeled with 't' are in product states that should be accepting
    print(f"\n=== Product States with 't' Labels ===")
    for abs_state_idx, label in enumerate(abs_model.MDP.labelling):
        if 't' in label:
            state_coords = abs_model.state_set[abs_state_idx]
            print(f"Abstraction state {abs_state_idx}: coords {state_coords}, label '{label}'")
            
            # Find all product states that include this abstraction state
            matching_prod_states = []
            for prod_idx, prod_state in enumerate(prod_auto.prod_state_set):
                prod_mdp_state_idx, dfa_cs_state, dfa_safe_state = prod_state
                
                # FIXED: Get the actual abstraction state index from product MDP
                if prod_mdp_state_idx < len(mdp_prod.prod_state_set):
                    x_sys, x_env = mdp_prod.prod_state_set[prod_mdp_state_idx]
                    
                    # Check if this product state corresponds to our abstraction state
                    if x_sys == abs_state_idx:
                        matching_prod_states.append((prod_idx, prod_state))
                        is_accepting = prod_idx in prod_auto.accepting_states
                        print(f"  Product state {prod_idx}: {prod_state} -> {'ACCEPTING' if is_accepting else 'NOT ACCEPTING'}")
                        print(f"    (prod_mdp_state_idx={prod_mdp_state_idx} -> x_sys={x_sys}, x_env={x_env})")
            
            if not matching_prod_states:
                print(f"  ❌ No product states found for abstraction state {abs_state_idx}!")
    
    # Check why sink state 2 is accepting when it should be for states that reached 't'
    print(f"\n=== DFA Transition Analysis ===")
    print(f"F(t) formula should make state 2 accepting only after seeing 't'")
    print(f"Let's trace what happens when we see 't' vs '_':")
    
    # Simulate DFA transitions
    initial_state = scltl_frag.dfa.initial_state
    print(f"Initial DFA state: {initial_state}")
    
    # What happens when we see 't'?
    t_transition_key = (str(initial_state), t_alphabet)
    if t_transition_key in scltl_frag.dfa.transitions:
        next_state_t = scltl_frag.dfa.transitions[t_transition_key]
        print(f"From state {initial_state}, seeing 't' -> state {next_state_t}")
        is_sink_t = scltl_frag.dfa.is_sink_state(str(next_state_t))
        print(f"  State {next_state_t} is sink: {is_sink_t}")
    else:
        print(f"No transition from state {initial_state} with 't' alphabet {t_alphabet}")
    
    # What happens when we see '_'?
    empty_transition_key = (str(initial_state), empty_alphabet)
    if empty_transition_key in scltl_frag.dfa.transitions:
        next_state_empty = scltl_frag.dfa.transitions[empty_transition_key]
        print(f"From state {initial_state}, seeing '_' -> state {next_state_empty}")
        is_sink_empty = scltl_frag.dfa.is_sink_state(str(next_state_empty))
        print(f"  State {next_state_empty} is sink: {is_sink_empty}")
    else:
        print(f"No transition from state {initial_state} with '_' alphabet {empty_alphabet}")

    print("\n=== CRITICAL INSIGHT ===")
    print("The issue might be that F(t) DFA is constructed incorrectly.")
    print("For F(t), we should have:")
    print("- State 1 (initial): not accepting, waiting for 't'")
    print("- State 2 (sink): accepting, reached after seeing 't'")
    print("But it seems like ALL states with dfa_cs_state=2 are accepting,")
    print("regardless of whether they actually saw 't' or not!")
    
    print("=== End Debug Analysis ===")
    
    # ========== ADDITIONAL DEBUG: Check initial states and transitions ==========
    print("\n=== ADDITIONAL DEBUG: Initial States and Transitions ===")
    
    # Check initial product state
    print(f"Product MDP initial state: {mdp_prod.MDP.initial_state}")
    initial_prod_state = mdp_prod.prod_state_set[mdp_prod.MDP.initial_state]
    print(f"Initial product MDP state: {initial_prod_state}")
    
    # Check initial product automaton state
    print(f"Current ego state: {ego_state}")
    current_abs_state = abs_model.get_abs_state(ego_state)
    current_abs_state_idx = abs_model.get_state_index(current_abs_state)
    print(f"Current abstraction state: {current_abs_state} (index {current_abs_state_idx})")
    
    # Check what product MDP state this corresponds to
    for prod_mdp_idx, (x_sys, x_env) in enumerate(mdp_prod.prod_state_set):
        if x_sys == current_abs_state_idx:
            print(f"Product MDP state {prod_mdp_idx}: (x_sys={x_sys}, x_env={x_env})")
            
            # Check all product automaton states for this product MDP state
            for prod_auto_idx, (prod_mdp_state_idx, dfa_cs_state, dfa_safe_state) in enumerate(prod_auto.prod_state_set):
                if prod_mdp_state_idx == prod_mdp_idx:
                    is_accepting = prod_auto_idx in prod_auto.accepting_states
                    print(f"  Product automaton state {prod_auto_idx}: ({prod_mdp_state_idx}, {dfa_cs_state}, {dfa_safe_state}) -> {'ACCEPTING' if is_accepting else 'NOT ACCEPTING'}")
            break
    
    # Check why states with label '_' are in dfa_cs_state=2
    print(f"\n=== Why are '_' states in dfa_cs_state=2? ===")
    print(f"This suggests the product automaton construction has a bug.")
    print(f"States should only be in dfa_cs_state=2 if they transitioned there by seeing 't'.")
    print(f"But we see states with label '_' in dfa_cs_state=2, which shouldn't happen.")
    
    # Check a specific transition
    print(f"\n=== Checking specific transition logic ===")
    # Take a state with label '_' and see how it could get to dfa_cs_state=2
    test_abs_state_idx = 0  # This has label '_'
    test_label = abs_model.MDP.labelling[test_abs_state_idx]
    test_coords = abs_model.state_set[test_abs_state_idx]
    print(f"Test state: abstraction state {test_abs_state_idx}, coords {test_coords}, label '{test_label}'")
    
    # Check what alphabet this maps to
    test_alphabet_cs = scltl_frag.dfa.get_alphabet(test_label)
    print(f"Label '{test_label}' maps to alphabet {test_alphabet_cs}")
    
    # Check transitions from DFA initial state with this alphabet
    initial_dfa_state = scltl_frag.dfa.initial_state
    transition_key = (str(initial_dfa_state), test_alphabet_cs)
    if transition_key in scltl_frag.dfa.transitions:
        next_dfa_state = scltl_frag.dfa.transitions[transition_key]
        print(f"DFA transition: state {initial_dfa_state} + alphabet {test_alphabet_cs} -> state {next_dfa_state}")
        print(f"This means a state with label '_' should go to dfa_cs_state={next_dfa_state}, not 2!")
    else:
        print(f"No DFA transition found for state {initial_dfa_state} + alphabet {test_alphabet_cs}")
    
    print("=== End Additional Debug ===")

if __name__ == '__main__':
    main() 