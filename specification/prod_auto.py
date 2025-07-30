import numpy as np

from abstraction.abstraction import Abstraction
from specification.specification import LTL_Spec

class Product:
    def __init__(self, MDP, DFA_cs, DFA_safe):
        self.MDP = MDP
        self.DFA_cs = DFA_cs
        self.DFA_safe = DFA_safe
        self.prod_state_set = [(x, s_cs, s_s)
                              for x in self.MDP.states
                              for s_cs in self.DFA_cs.states
                              for s_s in self.DFA_safe.states]
        self.prod_action_set = self.MDP.actions
        self.prod_transitions = self.gen_product_transition()
        self.accepting_states, self.trap_states = self.gen_final_states()

    def gen_product_transition(self):
        P_matrix = np.zeros((len(self.prod_state_set), len(self.prod_action_set), len(self.prod_state_set)))
        for n in range(len(self.prod_state_set)):
            x, s_cs, s_s = self.prod_state_set[n]
            for a in self.MDP.actions:
                next_x_prob = self.MDP.transitions[x, a, :]
                for next_x in range(len(next_x_prob)):
                    next_x_label = self.MDP.labelling[next_x]
                    alphabet_cs = self.DFA_cs.get_alphabet(next_x_label)
                    alphabet_s = self.DFA_safe.get_alphabet(next_x_label)

                    if self.DFA_cs.transitions.get((str(s_cs), alphabet_cs)) is not None:
                        next_s_cs = self.DFA_cs.transitions.get((str(s_cs), alphabet_cs))
                    else:
                        next_s_cs = s_cs

                    if self.DFA_safe.transitions.get((str(s_s), alphabet_s)) is not None:
                        next_s_s = self.DFA_safe.transitions.get((str(s_s), alphabet_s))
                    else:
                        next_s_s = s_s

                    next_state_index = self.prod_state_set.index((next_x, int(next_s_cs), int(next_s_s)))
                    P_matrix[n, a, next_state_index] = next_x_prob[next_x]
        return P_matrix


    def gen_final_states(self):
        accepting_states = []
        trap_states = []
        for n in range(len(self.prod_state_set)):
            x, s_cs, s_s = self.prod_state_set[n]
            if self.DFA_safe.is_sink_state(str(s_s)):
                trap_states.append(n)
            if self.DFA_cs.is_sink_state(str(s_cs)):
                accepting_states.append(n)
        return accepting_states, trap_states

    def gen_cost_map(self, cost_func):
        cost_map = np.zeros(len(self.prod_state_set))
        for n in self.trap_states:
            x, s_cs, s_s = self.prod_state_set[n]
            label = self.MDP.labelling[x]
            for ap, cost in cost_func.items():
                if ap in label:
                    cost_map[n] += cost
        return cost_map
    
    def check_cost_map(self, cost_func):
        """
        Check the cost map for potential issues that might affect the LP solver.
        
        Args:
            cost_func: Dictionary mapping atomic propositions to costs
        
        Returns:
            dict: Dictionary containing check results with keys:
                - 'valid': Boolean indicating if the cost map is valid
                - 'issues': List of issues found
                - 'summary': String summary of the check results
                - 'cost_map': The generated cost map for reference
        """
        print("=== Checking Cost Map ===")
        
        issues = []
        cost_map = self.gen_cost_map(cost_func)
        
        # Basic statistics
        total_states = len(self.prod_state_set)
        trap_states_count = len(self.trap_states)
        non_zero_costs = np.count_nonzero(cost_map)
        min_cost = np.min(cost_map)
        max_cost = np.max(cost_map)
        mean_cost = np.mean(cost_map)
        
        print(f"Product states: {total_states}")
        print(f"Trap states: {trap_states_count}")
        print(f"States with non-zero cost: {non_zero_costs}")
        print(f"Cost range: [{min_cost}, {max_cost}]")
        print(f"Mean cost: {mean_cost:.6f}")
        print()
        
        # Check 1: Cost function validation
        if not cost_func:
            issues.append({
                'type': 'empty_cost_func',
                'message': 'Cost function is empty - no costs will be assigned'
            })
        
        # Check 2: Negative costs
        if min_cost < 0:
            negative_states = np.where(cost_map < 0)[0]
            issues.append({
                'type': 'negative_costs',
                'message': f'Found {len(negative_states)} states with negative costs',
                'negative_states': negative_states.tolist(),
                'min_cost': min_cost
            })
        
        # Check 3: Extremely high costs that might cause numerical issues
        if max_cost > 1000:
            high_cost_states = np.where(cost_map > 1000)[0]
            issues.append({
                'type': 'high_costs',
                'message': f'Found {len(high_cost_states)} states with very high costs (>1000)',
                'high_cost_states': high_cost_states.tolist(),
                'max_cost': max_cost
            })
        
        # Check 4: Only trap states should have costs
        non_trap_states_with_cost = []
        for state_idx in range(total_states):
            if cost_map[state_idx] > 0 and state_idx not in self.trap_states:
                non_trap_states_with_cost.append(state_idx)
        
        if non_trap_states_with_cost:
            issues.append({
                'type': 'non_trap_states_with_cost',
                'message': f'Found {len(non_trap_states_with_cost)} non-trap states with positive costs',
                'non_trap_states': non_trap_states_with_cost,
                'note': 'Only trap states should have costs according to the current implementation'
            })
        
        # Check 5: Trap states without costs (might indicate missing atomic propositions)
        trap_states_without_cost = []
        for trap_state_idx in self.trap_states:
            if cost_map[trap_state_idx] == 0:
                trap_states_without_cost.append(trap_state_idx)
        
        if trap_states_without_cost:
            issues.append({
                'type': 'trap_states_without_cost',
                'message': f'Found {len(trap_states_without_cost)} trap states with zero cost',
                'trap_states_without_cost': trap_states_without_cost,
                'note': 'These trap states might not have labels matching the cost function'
            })
        
        # Check 6: Analyze label coverage
        print("=== Label Coverage Analysis ===")
        cost_func_aps = set(cost_func.keys())
        labels_in_trap_states = set()
        trap_state_details = []
        
        for trap_state_idx in self.trap_states:
            x, s_cs, s_s = self.prod_state_set[trap_state_idx]
            label = self.MDP.labelling[x]
            labels_in_trap_states.update(label)
            
            # Calculate expected cost for this trap state
            expected_cost = 0
            matching_aps = []
            for ap, cost in cost_func.items():
                if ap in label:
                    expected_cost += cost
                    matching_aps.append(ap)
            
            trap_state_details.append({
                'trap_state_idx': trap_state_idx,
                'prod_state': self.prod_state_set[trap_state_idx],
                'mdp_state_idx': x,
                'label': label,
                'expected_cost': expected_cost,
                'actual_cost': cost_map[trap_state_idx],
                'matching_aps': matching_aps
            })
        
        print(f"Atomic propositions in cost function: {cost_func_aps}")
        print(f"Labels found in trap states: {labels_in_trap_states}")
        
        # Check for unused cost function APs
        unused_aps = cost_func_aps - labels_in_trap_states
        if unused_aps:
            issues.append({
                'type': 'unused_cost_aps',
                'message': f'Cost function contains APs not found in any trap state: {unused_aps}',
                'unused_aps': list(unused_aps),
                'note': 'These APs will never contribute to the cost'
            })
        
        # Check for labels in trap states not covered by cost function
        uncovered_labels = labels_in_trap_states - cost_func_aps
        if uncovered_labels:
            issues.append({
                'type': 'uncovered_labels',
                'message': f'Trap states contain labels not in cost function: {uncovered_labels}',
                'uncovered_labels': list(uncovered_labels),
                'note': 'These labels will not contribute to the cost'
            })
        
        # Check 7: Verify cost calculations
        print("\n=== Cost Calculation Verification ===")
        cost_mismatches = []
        
        for detail in trap_state_details:
            if detail['expected_cost'] != detail['actual_cost']:
                cost_mismatches.append(detail)
        
        if cost_mismatches:
            issues.append({
                'type': 'cost_calculation_mismatch',
                'message': f'Found {len(cost_mismatches)} trap states with cost calculation mismatches',
                'mismatches': cost_mismatches
            })
            
            print(f"❌ Found {len(cost_mismatches)} cost calculation mismatches:")
            for mismatch in cost_mismatches[:5]:
                print(f"  Trap state {mismatch['trap_state_idx']}: expected {mismatch['expected_cost']}, got {mismatch['actual_cost']}")
                print(f"    Label: {mismatch['label']}, Matching APs: {mismatch['matching_aps']}")
        else:
            print("✅ All cost calculations match expectations")
        
        # Check 8: Distribution analysis
        print("\n=== Cost Distribution Analysis ===")
        unique_costs, cost_counts = np.unique(cost_map, return_counts=True)
        
        print("Cost distribution:")
        for cost_val, count in zip(unique_costs, cost_counts):
            percentage = (count / total_states) * 100
            print(f"  Cost {cost_val}: {count} states ({percentage:.1f}%)")
        
        # Check if all costs are zero (potential issue)
        if max_cost == 0:
            issues.append({
                'type': 'all_zero_costs',
                'message': 'All states have zero cost - this might indicate a problem with the cost function or trap state identification',
                'note': 'LP solver might not work correctly with all-zero costs'
            })
        
        # Summary
        if issues:
            print(f"\n❌ FOUND {len(issues)} ISSUES WITH COST MAP:")
            for i, issue in enumerate(issues):
                print(f"  {i+1}. {issue['type']}: {issue['message']}")
            
            summary = f"ISSUES FOUND: {len(issues)} problems detected in cost map"
            valid = False
        else:
            print("\n✅ Cost map appears to be valid")
            summary = "VALID: Cost map is correctly generated"
            valid = True
        
        print("=== End Cost Map Check ===\n")
        
        return {
            'valid': valid,
            'issues': issues,
            'summary': summary,
            'cost_map': cost_map,
            'statistics': {
                'total_states': total_states,
                'trap_states_count': trap_states_count,
                'non_zero_costs': non_zero_costs,
                'min_cost': min_cost,
                'max_cost': max_cost,
                'mean_cost': mean_cost,
                'cost_distribution': dict(zip(unique_costs, cost_counts))
            },
            'trap_state_details': trap_state_details
        }
    
    def update_prod_state(self, cur_abs_index, last_product_state):
        _, last_cs_state, last_safe_state  = last_product_state
        label = self.MDP.labelling[cur_abs_index]
        alphabet_cs = self.DFA_cs.get_alphabet(label)
        alphabet_s = self.DFA_safe.get_alphabet(label)

        if self.DFA_cs.transitions.get((str(last_cs_state), alphabet_cs)) is not None:
            next_cs_state = self.DFA_cs.transitions.get((str(last_cs_state), alphabet_cs))
        else:
            next_cs_state = last_cs_state
        if self.DFA_safe.transitions.get((str(last_safe_state), alphabet_s)) is not None:
            next_safe_state = self.DFA_safe.transitions.get((str(last_safe_state), alphabet_s))
        else:
            next_safe_state = last_safe_state

        cur_product_state = (cur_abs_index, int(next_cs_state), int(next_safe_state))
        product_state_index = self.prod_state_set.index(cur_product_state)
        return int(product_state_index), cur_product_state

    def check_product_P_matrix_bounds(self, abs_model, prod_mdp):
        """
        Check if the product automaton P_matrix contains any invalid transitions.
        This function validates that all transitions in the product P_matrix are valid,
        specifically checking if the mdp_sys part of product states would be out of bounds.
        
        Args:
            abs_model: The abstraction model to get the original state space bounds
            prod_mdp: The Prod_MDP object to access the underlying mdp_sys structure
        
        Returns:
            dict: Dictionary containing check results with keys:
                - 'valid': Boolean indicating if all transitions are valid
                - 'invalid_transitions': List of invalid transitions found
                - 'summary': String summary of the check results
        """
        print("=== Checking Product Automaton P_Matrix for Invalid Transitions ===")
        
        # Get mdp_sys state space bounds from the abstraction model
        mdp_sys_state_bounds = {
            'x_max': abs_model.map_range[0] // abs_model.map_res[0] - 1,
            'y_max': abs_model.map_range[1] // abs_model.map_res[1] - 1,
            'v_max': abs_model.speed_range // abs_model.speed_res - 1,
            'num_states': len(abs_model.state_set)
        }
        
        # Get DFA state bounds
        dfa_cs_states = set(self.DFA_cs.states)
        dfa_safe_states = set(self.DFA_safe.states)
        
        print(f"MDP_sys state bounds: x=[0,{mdp_sys_state_bounds['x_max']}], y=[0,{mdp_sys_state_bounds['y_max']}], v=[0,{mdp_sys_state_bounds['v_max']}]")
        print(f"MDP_sys states: {mdp_sys_state_bounds['num_states']} (indices: 0 to {mdp_sys_state_bounds['num_states']-1})")
        print(f"DFA_cs states: {dfa_cs_states}")
        print(f"DFA_safe states: {dfa_safe_states}")
        print(f"Product states: {len(self.prod_state_set)}")
        print(f"Product actions: {len(self.prod_action_set)}")
        print()
        
        invalid_transitions = []
        
        # Check each product state-action pair
        for prod_state_idx in range(len(self.prod_state_set)):
            current_prod_state = self.prod_state_set[prod_state_idx]
            prod_mdp_state_idx, dfa_cs_state, dfa_safe_state = current_prod_state
            
            # Get the (x_sys, x_env) tuple from the product MDP
            if prod_mdp_state_idx >= len(prod_mdp.prod_state_set):
                invalid_transitions.append({
                    'type': 'invalid_prod_mdp_state_idx',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'prod_mdp_state_idx': prod_mdp_state_idx,
                    'valid_range': f"[0, {len(prod_mdp.prod_state_set)-1}]"
                })
                continue
            
            x_sys, x_env = prod_mdp.prod_state_set[prod_mdp_state_idx]
            
            # Check if x_sys is within valid bounds
            if x_sys < 0 or x_sys >= mdp_sys_state_bounds['num_states']:
                invalid_transitions.append({
                    'type': 'invalid_x_sys_index',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'x_sys': x_sys,
                    'x_env': x_env,
                    'valid_x_sys_range': f"[0, {mdp_sys_state_bounds['num_states']-1}]"
                })
                continue
            
            # Get the actual mdp_sys state coordinates
            sys_state_coords = abs_model.state_set[x_sys]
            
            # Check if the coordinates are within bounds
            if (sys_state_coords[0] < 0 or sys_state_coords[0] > mdp_sys_state_bounds['x_max'] or
                sys_state_coords[1] < 0 or sys_state_coords[1] > mdp_sys_state_bounds['y_max'] or
                sys_state_coords[2] < 0 or sys_state_coords[2] > mdp_sys_state_bounds['v_max']):
                
                invalid_transitions.append({
                    'type': 'invalid_x_sys_coordinates',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'x_sys': x_sys,
                    'x_env': x_env,
                    'sys_state_coords': sys_state_coords.tolist(),
                    'bounds': f"x=[0,{mdp_sys_state_bounds['x_max']}], y=[0,{mdp_sys_state_bounds['y_max']}], v=[0,{mdp_sys_state_bounds['v_max']}]"
                })
            
            # Check if DFA states are valid
            if dfa_cs_state not in dfa_cs_states:
                invalid_transitions.append({
                    'type': 'invalid_dfa_cs_state',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'invalid_dfa_cs_state': dfa_cs_state,
                    'valid_dfa_cs_states': dfa_cs_states
                })
            
            if dfa_safe_state not in dfa_safe_states:
                invalid_transitions.append({
                    'type': 'invalid_dfa_safe_state',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'invalid_dfa_safe_state': dfa_safe_state,
                    'valid_dfa_safe_states': dfa_safe_states
                })
            
            # Check transitions from this product state
            for action_idx in range(len(self.prod_action_set)):
                # Get transition probabilities for this product state-action pair
                transition_probs = self.prod_transitions[prod_state_idx, action_idx, :]
                
                # Check each possible next product state
                for next_prod_state_idx in range(len(self.prod_state_set)):
                    if transition_probs[next_prod_state_idx] > 0:  # Non-zero transition probability
                        next_prod_state = self.prod_state_set[next_prod_state_idx]
                        next_prod_mdp_state_idx, next_dfa_cs_state, next_dfa_safe_state = next_prod_state
                        
                        # Check if next product MDP state is valid
                        if next_prod_mdp_state_idx >= len(prod_mdp.prod_state_set):
                            invalid_transitions.append({
                                'type': 'invalid_next_prod_mdp_state_idx',
                                'from_prod_state': current_prod_state,
                                'from_prod_state_idx': prod_state_idx,
                                'action_idx': action_idx,
                                'to_prod_state': next_prod_state,
                                'to_prod_state_idx': next_prod_state_idx,
                                'invalid_next_prod_mdp_state_idx': next_prod_mdp_state_idx,
                                'valid_range': f"[0, {len(prod_mdp.prod_state_set)-1}]",
                                'probability': transition_probs[next_prod_state_idx]
                            })
                            continue
                        
                        # Get the next (x_sys, x_env) tuple
                        next_x_sys, next_x_env = prod_mdp.prod_state_set[next_prod_mdp_state_idx]
                        
                        # Check if next x_sys is within valid bounds
                        if next_x_sys < 0 or next_x_sys >= mdp_sys_state_bounds['num_states']:
                            invalid_transitions.append({
                                'type': 'invalid_next_x_sys_index',
                                'from_prod_state': current_prod_state,
                                'from_prod_state_idx': prod_state_idx,
                                'action_idx': action_idx,
                                'to_prod_state': next_prod_state,
                                'to_prod_state_idx': next_prod_state_idx,
                                'next_x_sys': next_x_sys,
                                'next_x_env': next_x_env,
                                'valid_x_sys_range': f"[0, {mdp_sys_state_bounds['num_states']-1}]",
                                'probability': transition_probs[next_prod_state_idx]
                            })
                            continue
                        
                        # Get the actual next mdp_sys state coordinates
                        next_sys_state_coords = abs_model.state_set[next_x_sys]
                        
                        # Check if the next coordinates are within bounds
                        if (next_sys_state_coords[0] < 0 or next_sys_state_coords[0] > mdp_sys_state_bounds['x_max'] or
                            next_sys_state_coords[1] < 0 or next_sys_state_coords[1] > mdp_sys_state_bounds['y_max'] or
                            next_sys_state_coords[2] < 0 or next_sys_state_coords[2] > mdp_sys_state_bounds['v_max']):
                            
                            invalid_transitions.append({
                                'type': 'invalid_next_x_sys_coordinates',
                                'from_prod_state': current_prod_state,
                                'from_prod_state_idx': prod_state_idx,
                                'action_idx': action_idx,
                                'to_prod_state': next_prod_state,
                                'to_prod_state_idx': next_prod_state_idx,
                                'next_x_sys': next_x_sys,
                                'next_x_env': next_x_env,
                                'next_sys_state_coords': next_sys_state_coords.tolist(),
                                'bounds': f"x=[0,{mdp_sys_state_bounds['x_max']}], y=[0,{mdp_sys_state_bounds['y_max']}], v=[0,{mdp_sys_state_bounds['v_max']}]",
                                'probability': transition_probs[next_prod_state_idx]
                            })
                        
                        # Check if next DFA states are valid
                        if next_dfa_cs_state not in dfa_cs_states:
                            invalid_transitions.append({
                                'type': 'invalid_next_dfa_cs_state',
                                'from_prod_state': current_prod_state,
                                'from_prod_state_idx': prod_state_idx,
                                'action_idx': action_idx,
                                'to_prod_state': next_prod_state,
                                'to_prod_state_idx': next_prod_state_idx,
                                'invalid_next_dfa_cs_state': next_dfa_cs_state,
                                'valid_dfa_cs_states': dfa_cs_states,
                                'probability': transition_probs[next_prod_state_idx]
                            })
                        
                        if next_dfa_safe_state not in dfa_safe_states:
                            invalid_transitions.append({
                                'type': 'invalid_next_dfa_safe_state',
                                'from_prod_state': current_prod_state,
                                'from_prod_state_idx': prod_state_idx,
                                'action_idx': action_idx,
                                'to_prod_state': next_prod_state,
                                'to_prod_state_idx': next_prod_state_idx,
                                'invalid_next_dfa_safe_state': next_dfa_safe_state,
                                'valid_dfa_safe_states': dfa_safe_states,
                                'probability': transition_probs[next_prod_state_idx]
                            })
        
        # Report results
        if invalid_transitions:
            print(f"❌ FOUND {len(invalid_transitions)} INVALID TRANSITIONS IN PRODUCT AUTOMATON!")
            
            # Group by type for better reporting
            by_type = {}
            for trans in invalid_transitions:
                trans_type = trans['type']
                if trans_type not in by_type:
                    by_type[trans_type] = []
                by_type[trans_type].append(trans)
            
            for trans_type, trans_list in by_type.items():
                print(f"\n{len(trans_list)} {trans_type} issues:")
                for i, trans in enumerate(trans_list[:5]):
                    if 'x_sys' in trans:
                        if 'next_x_sys' in trans:
                            print(f"  {i+1}: Transition from prod_state {trans['from_prod_state']} to {trans['to_prod_state']}")
                            print(f"      Next x_sys={trans['next_x_sys']} coords={trans.get('next_sys_state_coords', 'N/A')} (prob={trans.get('probability', 'N/A'):.6f})")
                        else:
                            print(f"  {i+1}: Product state {trans['prod_state']} has x_sys={trans['x_sys']} coords={trans.get('sys_state_coords', 'N/A')}")
                    else:
                        print(f"  {i+1}: {trans}")
                
                if len(trans_list) > 5:
                    print(f"  ... and {len(trans_list) - 5} more {trans_type} issues")
            
            summary = f"INVALID: Found {len(invalid_transitions)} invalid transitions in product automaton"
        else:
            print("✅ All product automaton transitions are valid")
            summary = "VALID: All product automaton transitions are valid"
        
        print("=== End Product Automaton P_Matrix Bounds Check ===\n")
        
        return {
            'valid': len(invalid_transitions) == 0,
            'invalid_transitions': invalid_transitions,
            'summary': summary
        }

    def check_product_mdp_sys_bounds(self, abs_model, prod_mdp):
        """
        Check if the product automaton contains transitions that would lead to 
        out-of-bounds states in the underlying mdp_sys (abstraction).
        
        Args:
            abs_model: The abstraction model to get the original state space bounds
            prod_mdp: The Prod_MDP object to access mdp_sys structure
        
        Returns:
            dict: Dictionary containing check results
        """
        print("=== Checking Product Automaton for MDP_sys Out-of-Bounds Transitions ===")
        
        # Get mdp_sys state space bounds from the abstraction model
        mdp_sys_state_bounds = {
            'x_max': abs_model.map_range[0] // abs_model.map_res[0] - 1,
            'y_max': abs_model.map_range[1] // abs_model.map_res[1] - 1,
            'v_max': abs_model.speed_range // abs_model.speed_res - 1,
            'num_states': len(abs_model.state_set)
        }
        
        print(f"MDP_sys state bounds: x=[0,{mdp_sys_state_bounds['x_max']}], y=[0,{mdp_sys_state_bounds['y_max']}], v=[0,{mdp_sys_state_bounds['v_max']}]")
        print(f"MDP_sys states: {mdp_sys_state_bounds['num_states']} (indices: 0 to {mdp_sys_state_bounds['num_states']-1})")
        print(f"Product automaton states: {len(self.prod_state_set)}")
        print()
        
        invalid_sys_transitions = []
        
        # Check each product state
        for prod_state_idx in range(len(self.prod_state_set)):
            current_prod_state = self.prod_state_set[prod_state_idx]
            prod_mdp_state_idx, dfa_cs_state, dfa_safe_state = current_prod_state
            
            # Get the (x_sys, x_env) tuple from the product MDP
            if prod_mdp_state_idx < len(prod_mdp.prod_state_set):
                x_sys, x_env = prod_mdp.prod_state_set[prod_mdp_state_idx]
                
                # Check if x_sys is within valid bounds
                if x_sys < 0 or x_sys >= mdp_sys_state_bounds['num_states']:
                    invalid_sys_transitions.append({
                        'type': 'invalid_x_sys_index',
                        'prod_state_idx': prod_state_idx,
                        'prod_state': current_prod_state,
                        'x_sys': x_sys,
                        'x_env': x_env,
                        'valid_x_sys_range': f"[0, {mdp_sys_state_bounds['num_states']-1}]"
                    })
                else:
                    # Get the actual mdp_sys state coordinates
                    if x_sys < len(abs_model.state_set):
                        sys_state_coords = abs_model.state_set[x_sys]
                        
                        # Check if the coordinates are within bounds
                        if (sys_state_coords[0] < 0 or sys_state_coords[0] > mdp_sys_state_bounds['x_max'] or
                            sys_state_coords[1] < 0 or sys_state_coords[1] > mdp_sys_state_bounds['y_max'] or
                            sys_state_coords[2] < 0 or sys_state_coords[2] > mdp_sys_state_bounds['v_max']):
                            
                            invalid_sys_transitions.append({
                                'type': 'invalid_x_sys_coordinates',
                                'prod_state_idx': prod_state_idx,
                                'prod_state': current_prod_state,
                                'x_sys': x_sys,
                                'x_env': x_env,
                                'sys_state_coords': sys_state_coords.tolist(),
                                'bounds': f"x=[0,{mdp_sys_state_bounds['x_max']}], y=[0,{mdp_sys_state_bounds['y_max']}], v=[0,{mdp_sys_state_bounds['v_max']}]"
                            })
            else:
                invalid_sys_transitions.append({
                    'type': 'invalid_prod_mdp_state_idx',
                    'prod_state_idx': prod_state_idx,
                    'prod_state': current_prod_state,
                    'prod_mdp_state_idx': prod_mdp_state_idx,
                    'valid_range': f"[0, {len(prod_mdp.prod_state_set)-1}]"
                })
        
        # Report results
        if invalid_sys_transitions:
            print(f"❌ FOUND {len(invalid_sys_transitions)} INVALID MDP_SYS TRANSITIONS!")
            
            for i, invalid in enumerate(invalid_sys_transitions[:10]):
                if invalid['type'] == 'invalid_x_sys_index':
                    print(f"  {i+1}: Product state {invalid['prod_state']} has x_sys={invalid['x_sys']} out of range {invalid['valid_x_sys_range']}")
                elif invalid['type'] == 'invalid_x_sys_coordinates':
                    print(f"  {i+1}: Product state {invalid['prod_state']} has x_sys={invalid['x_sys']} with coords {invalid['sys_state_coords']} out of bounds {invalid['bounds']}")
                elif invalid['type'] == 'invalid_prod_mdp_state_idx':
                    print(f"  {i+1}: Product state {invalid['prod_state']} has invalid prod_mdp_state_idx={invalid['prod_mdp_state_idx']} (valid range: {invalid['valid_range']})")
            
            if len(invalid_sys_transitions) > 10:
                print(f"  ... and {len(invalid_sys_transitions) - 10} more invalid transitions")
            
            summary = f"INVALID: Found {len(invalid_sys_transitions)} invalid mdp_sys transitions"
        else:
            print("✅ All product automaton mdp_sys transitions are valid")
            summary = "VALID: All product automaton mdp_sys transitions are valid"
        
        print("=== End Product Automaton MDP_sys Bounds Check ===\n")
        
        return {
            'valid': len(invalid_sys_transitions) == 0,
            'invalid_sys_transitions': invalid_sys_transitions,
            'summary': summary
        }

    def check_product_state_consistency(self):
        """
        Check if all product states are consistent with their component states.
        This verifies that MDP states in product states actually exist in the MDP.
        
        Returns:
            dict: Dictionary containing consistency check results
        """
        print("=== Checking Product State Consistency ===")
        
        inconsistent_states = []
        mdp_state_indices = set(range(len(self.MDP.states)))
        dfa_cs_states = set(self.DFA_cs.states)
        dfa_safe_states = set(self.DFA_safe.states)
        
        for prod_state_idx, prod_state in enumerate(self.prod_state_set):
            mdp_state_idx, dfa_cs_state, dfa_safe_state = prod_state
            
            issues = []
            
            # Check MDP state index
            if mdp_state_idx not in mdp_state_indices:
                issues.append(f"MDP state {mdp_state_idx} not in valid range [0, {len(self.MDP.states)-1}]")
            
            # Check DFA states
            if dfa_cs_state not in dfa_cs_states:
                issues.append(f"DFA_cs state {dfa_cs_state} not in valid states {dfa_cs_states}")
            
            if dfa_safe_state not in dfa_safe_states:
                issues.append(f"DFA_safe state {dfa_safe_state} not in valid states {dfa_safe_states}")
            
            if issues:
                inconsistent_states.append({
                    'prod_state_idx': prod_state_idx,
                    'prod_state': prod_state,
                    'issues': issues
                })
        
        if inconsistent_states:
            print(f"❌ FOUND {len(inconsistent_states)} INCONSISTENT PRODUCT STATES!")
            for i, inconsistent in enumerate(inconsistent_states[:5]):
                print(f"  {i+1}: Product state {inconsistent['prod_state']} (index {inconsistent['prod_state_idx']})")
                for issue in inconsistent['issues']:
                    print(f"      - {issue}")
            if len(inconsistent_states) > 5:
                print(f"  ... and {len(inconsistent_states) - 5} more inconsistent states")
            
            summary = f"INCONSISTENT: Found {len(inconsistent_states)} inconsistent product states"
        else:
            print("✅ All product states are consistent")
            summary = "CONSISTENT: All product states are consistent"
        
        print("=== End Product State Consistency Check ===\n")
        
        return {
            'consistent': len(inconsistent_states) == 0,
            'inconsistent_states': inconsistent_states,
            'summary': summary
        }


if __name__ == '__main__':
    pcpt_range = (20, 20)
    pcpt_res = (5, 5)
    dt = 1
    initial_position = (2, 2)
    label_func = {(15, 20, 15, 20): "t",
                  (5, 15, 5, 10): "o",
                  (10, 20, 0, 20): "c"}

    abs_model = Abstraction(pcpt_range, pcpt_res, initial_position, label_func)
    MDP = abs_model.MDP

    # safe_frag = LTL_Spec("G(r->!c)")
    safe_frag = LTL_Spec("G(!o)")
    scltl_frag = LTL_Spec("F(t)")
    prod_auto = Product(MDP, safe_frag.dfa, scltl_frag.dfa)

    for n in prod_auto.accepting_states:
        print(prod_auto.prod_state_set[n])
    for n in prod_auto.trap_states:
        print(prod_auto.prod_state_set[n])
