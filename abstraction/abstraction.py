#!/usr/bin/env python
import gurobipy as grb
import numpy as np
from abstraction.MDP import MDP
from scipy.stats import norm

class Abstraction:

    def __init__(self, route_size, route_res, speed_range, speed_res, initial_state, label_function):
        '''
        route_size: (l, d), where l is the length of the route and d is the width of the route
        '''
        self.route_size = route_size
        self.route_res = route_res
        self.map_shape = None
        self.speed_range = speed_range
        self.speed_res = speed_res
        self.state_set = self.gen_abs_state(route_size, route_res, speed_range, speed_res)
        self.action_set = self.gen_abs_action()
        trans_matrix = self.gen_transitions()
        # self.sanity_check_trans_func()
        label_map = self.gen_labels(label_function)
        self.init_abs_state = [int(initial_state[0]//self.route_res[0]), 
                               int((initial_state[1] + self.route_res[1]/2) // self.route_res[1]), 
                               int(initial_state[3]//self.speed_res)]
        initial_state_index = self.get_state_index(self.init_abs_state)
        state_index_set = np.arange(len(self.state_set))
        action_index_set = np.arange(len(self.action_set))
        self.MDP = MDP(state_index_set, action_index_set, trans_matrix, label_map, initial_state_index)

    def gen_abs_state(self, route_size, route_res, speed_range, speed_res):
        # Position dimensions
        r_bl = 0
        r_bu = int(route_size[0] / route_res[0])
        ey_bl = -(int(route_size[1] / route_res[1]) // 2)   # division by 2 due to two sides (left and right)
        ey_bu = int(route_size[1] / route_res[1]) // 2 + 1

        # Velocity dimension
        v_bl = 0
        v_bu = int(speed_range / speed_res)  # Number of discrete velocity levels

        # Save shape (R, EY, V)
        self.map_shape = (r_bu - r_bl, ey_bu - ey_bl, v_bu - v_bl)

        # Build state list without meshgrid, with r varying fastest, then ey, then v
        # This matches P_sn.flatten(order='F') used in trans_func
        states = []
        for v in range(v_bl, v_bu):
            for ey in range(ey_bl, ey_bu):
                for r in range(r_bl, r_bu):
                    states.append([r, ey, v])
        return np.array(states)


    def gen_abs_action(self):
        # vx_set = np.array([-2, -1, 0, 1, 2])
        # vy_set = np.array([-2, -1, 0, 1, 2])
        # vx_set = np.array([-1, 0, 1])
        # vy_set = np.array([-1, 0, 1])

        # x_set = np.array([0, 1])
        # y_set = np.array([-1, 0, 1])
        # v_set = np.array([-1, 0, 1])
        # A, B, C = np.meshgrid(x_set, y_set, v_set)
        move_set = np.array(['l', 'f', 'r'])   # 'l' for 'left', 'f' for 'forward', 'r' for 'right'
        speed_set = np.array(['d', 'c', 'a'])  # 'd' for 'decelerate', 'c' for 'cruise', 'a' for 'accelerate'
        A, B = np.meshgrid(move_set, speed_set)
        return np.array([A.flatten(), B.flatten()]).T

    def gen_transitions(self):
        P = None
        for state in self.state_set:
            P_s = None
            for action in self.action_set:
                P_s_a = self.trans_func(state, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P

    def gen_labels(self, label_function):
        # The setting of resolution should correspond to regions of each label
        label_map = np.array(["_"]*len(self.state_set), dtype=object)
        for state_region, label in label_function.items():
            r_bl, r_bu, ey_bl, ey_bu, v_bl, v_bu = state_region
            
            # r discretization: (0, res) -> r_discrete=0, (res, 2*res) -> r_discrete=1, etc.
            # So r_continuous ∈ [i*res, (i+1)*res) maps to r_discrete = i
            r_bl_idx = int(np.floor(r_bl / self.route_res[0]))
            r_bu_idx = int(np.ceil(r_bu / self.route_res[0]))
            
            # ey discretization: (-res/2, res/2) -> ey_discrete=0, (res/2, 3*res/2) -> ey_discrete=1, etc.
            # So ey_continuous ∈ [i*res - res/2, (i+1)*res - res/2) maps to ey_discrete = i
            # Equivalently: ey_continuous ∈ [(i-0.5)*res, (i+0.5)*res) maps to ey_discrete = i
            ey_bl_idx = int(np.floor((ey_bl + self.route_res[1]/2) / self.route_res[1]))
            ey_bu_idx = int(np.ceil((ey_bu + self.route_res[1]/2) / self.route_res[1]))
            
            # v discretization: similar to r, (0, res) -> v_discrete=0, etc.
            v_bl_idx = int(np.floor(v_bl / self.speed_res))
            v_bu_idx = int(np.ceil(v_bu / self.speed_res))
            
            for n in range(len(self.state_set)):
                if (r_bl_idx <= self.state_set[n, 0] < r_bu_idx and 
                    ey_bl_idx <= self.state_set[n, 1] < ey_bu_idx and 
                    v_bl_idx <= self.state_set[n, 2] < v_bu_idx):
                    label_map[n] = label if label_map[n] == '_' else label_map[n] + label
        
        # def sanity_gen_labels(label_map):
        #     for i, label in enumerate(label_map):
        #         if label != "_":
        #             print(f"state {self.state_set[i]} has label {label}")
        # sanity_gen_labels(label_map)
        
        return label_map

    def get_abs_state(self, system_state):
        abs_state = [int(system_state[0]//self.route_res[0]), 
                     int((system_state[1] + self.route_res[1]/2)//self.route_res[1]), 
                     int(system_state[3]//self.speed_res)]
        return abs_state
    
    def get_state_index(self, abs_state):
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    # def get_abs_ind_state(self, position):
    #     abs_state = [int(position[0]//self.route_res[0]), int(position[1]//self.route_res[1])]
    #     state_index = self.get_state_index(abs_state)
    #     return state_index, abs_state


    def trans_func(self, state, action):
        def get_move_transitions(move_action):
            """
            Returns combined probability distribution for (r, ey) changes based on move action.
            """
            # Create a 5x5 probability matrix for (δr, δey) changes
            # Index 2,2 represents no change (δr=0, δey=0)
            prob_spatial = np.zeros((5, 5))
            
            if move_action == 'l': 
                # Primary transition: r+1, ey+1 (forward and left)
                prob_spatial[3, 3] = 0.7  # r+1, ey+1
                # Some uncertainty around the main transition
                prob_spatial[3, 2] = 0.1  # r+1, ey+0 (forward only)
                prob_spatial[2, 3] = 0.1  # r+0, ey+1 (left only)
                prob_spatial[3, 4] = 0.05 # r+1, ey+2 (forward, more left)
                prob_spatial[4, 3] = 0.05 # r+2, ey+1 (more forward, left)
                
            elif move_action == 'f': 
                # Primary transition: r+1, ey+0 (forward only)
                prob_spatial[3, 2] = 0.8  # r+1, ey+0
                # Some uncertainty
                prob_spatial[2, 2] = 0.1  # r+0, ey+0 (no movement)
                prob_spatial[4, 2] = 0.05 # r+2, ey+0 (more forward)
                prob_spatial[3, 1] = 0.025 # r+1, ey-1 (slight right drift)
                prob_spatial[3, 3] = 0.025 # r+1, ey+1 (slight left drift)
                
            elif move_action == 'r':
                # Primary transition: r+1, ey-1 (forward and right)
                prob_spatial[3, 1] = 0.7  # r+1, ey-1
                # Some uncertainty around the main transition
                prob_spatial[3, 2] = 0.1  # r+1, ey+0 (forward only)
                prob_spatial[2, 1] = 0.1  # r+0, ey-1 (right only)
                prob_spatial[3, 0] = 0.05 # r+1, ey-2 (forward, more right)
                prob_spatial[4, 1] = 0.05 # r+2, ey-1 (more forward, right)

            return prob_spatial

        def get_speed_transitions(speed_action):
            """
            Returns probability distribution for velocity changes based on speed action.
            """
            prob_v = np.zeros(5)  # Index 2 represents no change (δv=0)
            
            if speed_action == 'a':
                prob_v[3] = 0.8  # v+1
                prob_v[2] = 0.1  # v+0 (failed acceleration)
                prob_v[4] = 0.1  # v+2 (over-acceleration)
            elif speed_action == 'c':  
                prob_v[2] = 1.0  # v+0 (no velocity change)
            elif speed_action == 'd':  
                prob_v[1] = 0.8  # v-1
                prob_v[2] = 0.1  # v+0 (failed deceleration)
                prob_v[0] = 0.1  # v-2 (over-deceleration)
            
            return prob_v

        # Get state space dimensions
        state_shape = (self.route_size[0] // self.route_res[0], 
                       self.route_size[1] // self.route_res[1], 
                       self.speed_range // self.speed_res)
        
        # Initialize probability matrix for next states
        P_sn = np.zeros(len(self.state_set)).reshape(state_shape)

        # Get transition probabilities for the action
        move_action, speed_action = action[0], action[1]
        prob_spatial = get_move_transitions(move_action)  # 5x5 matrix for (r, ey)
        prob_v = get_speed_transitions(speed_action)      # 1x5 array for v

        # Apply transitions to the current state
        for i in range(5):  # δr changes
            for j in range(5):  # δey changes
                for k in range(5):  # δv changes
                    if prob_spatial[i, j] > 0 and prob_v[k] > 0:
                        # Calculate next state indices (subtract 2 to center around current state)
                        next_r = state[0] + i - 2
                        next_ey = state[1] + j - 2
                        next_v = state[2] + k - 2
                        
                        # # Check bounds
                        # if (0 <= next_r <= state_shape[0] - 1) and \
                        #    (0 <= next_ey <= state_shape[1] - 1) and \
                        #    (0 <= next_v <= state_shape[2] - 1):
                        #     # Combined probability = spatial_prob * velocity_prob
                        #     P_sn[next_r, next_ey, next_v] = prob_spatial[i, j] * prob_v[k]

                        # Check bounds
                        r_bl = 0
                        r_bu = int(self.route_size[0] / self.route_res[0]) - 1
                        ey_bl = -(int(self.route_size[1] / self.route_res[1]) // 2)
                        ey_bu = int(self.route_size[1] / self.route_res[1]) // 2
                        v_bl = 0
                        v_bu = int(self.speed_range / self.speed_res) - 1
                        if (r_bl <= next_r <= r_bu) and \
                           (ey_bl <= next_ey <= ey_bu) and \
                           (v_bl <= next_v <= v_bu):
                            # Convert actual coordinates to array indices
                            r_idx = next_r - r_bl
                            ey_idx = next_ey - ey_bl  # This converts negative ey values to positive array indices
                            v_idx = next_v - v_bl
                            # Combined probability = spatial_prob * velocity_prob
                            P_sn[r_idx, ey_idx, v_idx] = prob_spatial[i, j] * prob_v[k]

        # Normalize the transition probabilities (since some probabilities might be filtered due to bounds)
        total = np.sum(P_sn)
        if total > 0:
            P_sn /= total

        return P_sn.flatten(order='F')


    def update_abs_init_state(self, system_state):
        abs_state = self.get_abs_state(system_state)
        state_index = self.get_state_index(abs_state)
        self.init_abs_state = abs_state
        self.MDP.initial_state = state_index

    # def sanity_check_trans_func(self, verbose=True):
    #     """
    #     Comprehensive sanity check for the trans_func method.
    #     Tests probability normalization, action semantics, bounds checking, and expected transitions.
        
    #     Args:
    #         verbose (bool): If True, prints detailed information during checks
            
    #     Returns:
    #         dict: Dictionary containing check results and any issues found
    #     """
    #     if verbose:
    #         print("=== Sanity Check for trans_func() ===")
        
    #     issues = []
    #     check_results = {
    #         'probability_normalization': [],
    #         'action_semantics': [],  
    #         'bounds_checking': [],
    #         'state_coverage': [],
    #         'overall_valid': True
    #     }
        
    #     # Get state space dimensions and bounds
    #     r_bl = 0
    #     r_bu = int(self.route_size[0] / self.route_res[0]) - 1
    #     ey_bl = -(int(self.route_size[1] / self.route_res[1]) // 2)
    #     ey_bu = int(self.route_size[1] / self.route_res[1]) // 2
    #     v_bl = 0
    #     v_bu = int(self.speed_range / self.speed_res) - 1
        
    #     state_shape = (self.route_size[0] // self.route_res[0], 
    #                    self.route_size[1] // self.route_res[1], 
    #                    self.speed_range // self.speed_res)
        
    #     if verbose:
    #         print(f"State space shape: {state_shape}")
    #         print(f"r bounds: [{r_bl}, {r_bu}]")
    #         print(f"ey bounds: [{ey_bl}, {ey_bu}]") 
    #         print(f"v bounds: [{v_bl}, {v_bu}]")
    #         print(f"Total states: {len(self.state_set)}")
    #         print(f"Total actions: {len(self.action_set)}")
        
    #     # Test a sample of states and actions (updated for new coordinate system)
    #     test_states = [
    #         [1, 0, 1],    # Interior state with ey=0 (center line)
    #         [r_bl, ey_bl, v_bl],  # Corner state (min values)
    #         [r_bu, ey_bu, v_bu],  # Corner state (max values)
    #         [r_bu//2, 0, v_bu//2],  # Center state
    #         [1, ey_bl, 1],  # Left boundary state
    #         [1, ey_bu, 1]   # Right boundary state
    #     ]
        
    #     test_actions = [
    #         ['l', 'a'], ['l', 'c'], ['l', 'd'],  # Left with different speeds
    #         ['f', 'a'], ['f', 'c'], ['f', 'd'],  # Forward with different speeds  
    #         ['r', 'a'], ['r', 'c'], ['r', 'd']   # Right with different speeds
    #     ]
        
    #     if verbose:
    #         print(f"\nTesting {len(test_states)} states with {len(test_actions)} actions...")
        
    #     for state_idx, state in enumerate(test_states):
    #         # Skip invalid states (check new bounds)
    #         if (state[0] < r_bl or state[0] > r_bu or 
    #             state[1] < ey_bl or state[1] > ey_bu or
    #             state[2] < v_bl or state[2] > v_bu):
    #             if verbose:
    #                 print(f"Skipping invalid test state {state}")
    #             continue
                
    #         if verbose:
    #             print(f"\n--- Testing State {state} ---")
            
    #         for action in test_actions:
    #             try:
    #                 # Get transition probabilities
    #                 P_transition = self.trans_func(state, action)
    #                 P_reshaped = P_transition.reshape(state_shape, order='F')  # Must match the flatten(order='F') in trans_func
                    
    #                 # Check 1: Probability normalization
    #                 prob_sum = np.sum(P_transition)
    #                 if abs(prob_sum - 1.0) > 1e-6:
    #                     issue = f"State {state}, Action {action}: Probabilities sum to {prob_sum:.6f}, not 1.0"
    #                     issues.append(issue)
    #                     check_results['probability_normalization'].append(issue)
    #                     if verbose:
    #                         print(f"  ❌ {issue}")
    #                 elif verbose:
    #                     print(f"  ✅ Action {action}: Probabilities sum to {prob_sum:.6f}")
                    
    #                 # Check 2: No negative probabilities
    #                 if np.any(P_transition < 0):
    #                     issue = f"State {state}, Action {action}: Contains negative probabilities"
    #                     issues.append(issue)
    #                     check_results['probability_normalization'].append(issue)
    #                     if verbose:
    #                         print(f"  ❌ {issue}")
                    
    #                 # Check 3: Action semantics (test expected transitions)
    #                 non_zero_transitions = np.where(P_reshaped > 0)
    #                 move_action, speed_action = action[0], action[1]
                    
    #                 # Verify main transition exists
    #                 expected_r_change = 1 if move_action in ['l', 'f', 'r'] else 0
    #                 expected_ey_change = 1 if move_action == 'l' else (-1 if move_action == 'r' else 0)
    #                 expected_v_change = 1 if speed_action == 'a' else (-1 if speed_action == 'd' else 0)
                    
    #                 expected_next_r = state[0] + expected_r_change
    #                 expected_next_ey = state[1] + expected_ey_change  
    #                 expected_next_v = state[2] + expected_v_change
                    
    #                 # Check if expected transition is within bounds and has high probability
    #                 if (r_bl <= expected_next_r <= r_bu and 
    #                     ey_bl <= expected_next_ey <= ey_bu and
    #                     v_bl <= expected_next_v <= v_bu):
                        
    #                     # Convert to array indices for P_reshaped access
    #                     r_idx = expected_next_r - r_bl
    #                     ey_idx = expected_next_ey - ey_bl  # Adjust for negative ey values
    #                     v_idx = expected_next_v - v_bl
                        
    #                     expected_prob = P_reshaped[r_idx, ey_idx, v_idx]
    #                     if expected_prob < 0.3:  # Should have high probability for main transition
    #                         issue = f"State {state}, Action {action}: Expected main transition to ({expected_next_r}, {expected_next_ey}, {expected_next_v}) has low probability {expected_prob:.3f}"
    #                         issues.append(issue)
    #                         check_results['action_semantics'].append(issue)
    #                         if verbose:
    #                             print(f"  ⚠️  {issue}")
    #                     elif verbose:
    #                         print(f"  ✅ Main transition to ({expected_next_r}, {expected_next_ey}, {expected_next_v}) has probability {expected_prob:.3f}")
                    
    #                 # Check 4: All transitions are within bounds (convert indices back to coordinates)
    #                 for i, (r_idx, ey_idx, v_idx) in enumerate(zip(non_zero_transitions[0], non_zero_transitions[1], non_zero_transitions[2])):
    #                     actual_r = r_idx + r_bl
    #                     actual_ey = ey_idx + ey_bl  # Convert back to actual ey coordinate
    #                     actual_v = v_idx + v_bl
                        
    #                     if (actual_r < r_bl or actual_r > r_bu or 
    #                         actual_ey < ey_bl or actual_ey > ey_bu or
    #                         actual_v < v_bl or actual_v > v_bu):
    #                         issue = f"State {state}, Action {action}: Transition to out-of-bounds state ({actual_r}, {actual_ey}, {actual_v})"
    #                         issues.append(issue)
    #                         check_results['bounds_checking'].append(issue)
    #                         if verbose:
    #                             print(f"  ❌ {issue}")
                    
    #                 if verbose and len(non_zero_transitions[0]) > 0:
    #                     print(f"  📊 Number of possible next states: {len(non_zero_transitions[0])}")
    #                     # Show top 3 most likely transitions (convert indices to coordinates)
    #                     flat_indices = np.argsort(P_transition)[::-1][:3]
    #                     print(f"  📍 Top 3 transitions:")
    #                     for idx in flat_indices:
    #                         if P_transition[idx] > 0:
    #                             next_state_indices = np.unravel_index(idx, state_shape, order='F')  # Must match flatten/reshape order
    #                             # Convert indices to actual coordinates
    #                             actual_coords = (next_state_indices[0] + r_bl, 
    #                                            next_state_indices[1] + ey_bl, 
    #                                            next_state_indices[2] + v_bl)
    #                             print(f"     -> {actual_coords} with prob {P_transition[idx]:.3f}")
                    
    #             except Exception as e:
    #                 issue = f"State {state}, Action {action}: Exception occurred - {str(e)}"
    #                 issues.append(issue)
    #                 check_results['overall_valid'] = False
    #                 if verbose:
    #                     print(f"  ❌ {issue}")
        
    #     # Summary
    #     if verbose:
    #         print(f"\n=== Summary ===")
    #         if len(issues) == 0:
    #             print("✅ All sanity checks passed!")
    #         else:
    #             print(f"❌ Found {len(issues)} issues:")
    #             for issue in issues[:10]:  # Show first 10 issues
    #                 print(f"  - {issue}")
    #             if len(issues) > 10:
    #                 print(f"  ... and {len(issues) - 10} more issues")
        
    #     check_results['total_issues'] = len(issues)
    #     check_results['overall_valid'] = len(issues) == 0
    #     check_results['issues'] = issues
        
    #     if verbose:
    #         print("=== End Sanity Check ===\n")
        
    #     return check_results
    
    # def check_P_matrix_bounds(self):
    #     """
    #     Check if the P matrix contains any transitions to out-of-bounds states.
    #     This function validates that all transitions in the P matrix respect state space boundaries.
        
    #     Returns:
    #         dict: Dictionary containing check results with keys:
    #             - 'valid': Boolean indicating if all transitions are valid
    #             - 'invalid_transitions': List of invalid transitions found
    #             - 'summary': String summary of the check results
    #     """
    #     print("=== Checking P Matrix for Out-of-Bounds Transitions ===")
        
    #     # Get state space bounds
    #     state_bounds = {
    #         'x_max': self.route_size[0] // self.route_res[0] - 1,
    #         'y_max': self.route_size[1] // self.route_res[1] - 1,
    #         'v_max': self.speed_range // self.speed_res - 1
    #     }
        
    #     print(f"State space bounds: x=[0,{state_bounds['x_max']}], y=[0,{state_bounds['y_max']}], v=[0,{state_bounds['v_max']}]")
    #     print(f"Total states: {len(self.state_set)}")
    #     print(f"Total actions: {len(self.action_set)}")
        
    #     invalid_transitions = []
    #     P_matrix = self.MDP.transitions
        
    #     # Check each state-action pair
    #     for state_idx in range(len(self.state_set)):
    #         current_state = self.state_set[state_idx]
            
    #         for action_idx in range(len(self.action_set)):
    #             current_action = self.action_set[action_idx]
                
    #             # Get transition probabilities for this state-action pair
    #             transition_probs = P_matrix[state_idx, action_idx, :]
                
    #             # Check each possible next state
    #             for next_state_idx in range(len(self.state_set)):
    #                 if transition_probs[next_state_idx] > 0:  # Non-zero transition probability
    #                     next_state = self.state_set[next_state_idx]
                        
    #                     # Check if next state is within bounds
    #                     if (next_state[0] < 0 or next_state[0] > state_bounds['x_max'] or
    #                         next_state[1] < 0 or next_state[1] > state_bounds['y_max'] or
    #                         next_state[2] < 0 or next_state[2] > state_bounds['v_max']):
                            
    #                         invalid_transitions.append({
    #                             'from_state': current_state.tolist(),
    #                             'from_state_idx': state_idx,
    #                             'action': current_action.tolist(),
    #                             'action_idx': action_idx,
    #                             'to_state': next_state.tolist(),
    #                             'to_state_idx': next_state_idx,
    #                             'probability': transition_probs[next_state_idx]
    #                         })
        
    #     # Report results
    #     if invalid_transitions:
    #         print(f"❌ FOUND {len(invalid_transitions)} INVALID TRANSITIONS!")
    #         print("\nFirst 10 invalid transitions:")
    #         for i, trans in enumerate(invalid_transitions[:10]):
    #             print(f"  {i+1}: State {trans['from_state']} --{trans['action']}--> State {trans['to_state']} (prob={trans['probability']:.6f})")
            
    #         if len(invalid_transitions) > 10:
    #             print(f"  ... and {len(invalid_transitions) - 10} more invalid transitions")
                
    #         summary = f"INVALID: Found {len(invalid_transitions)} transitions to out-of-bounds states"
    #     else:
    #         print("✅ All transitions are valid - no out-of-bounds states found")
    #         summary = "VALID: All transitions respect state space boundaries"
        
    #     print("=== End P Matrix Bounds Check ===\n")
        
    #     return {
    #         'valid': len(invalid_transitions) == 0,
    #         'invalid_transitions': invalid_transitions,
    #         'summary': summary
    #     }

    # def check_state_action_simple_addition(self):
    #     """
    #     Check which state-action pairs would lead to out-of-bounds if using simple addition.
    #     This helps identify potential issues with the LP solver's target state calculation.
        
    #     Returns:
    #         dict: Dictionary containing check results
    #     """
    #     print("=== Checking State+Action Simple Addition for Out-of-Bounds ===")
        
    #     # Get state space bounds
    #     state_bounds = {
    #         'x_max': self.route_size[0] // self.route_res[0] - 1,
    #         'y_max': self.route_size[1] // self.route_res[1] - 1,
    #         'v_max': self.speed_range // self.speed_res - 1
    #     }
        
    #     problematic_combinations = []
        
    #     # Check each state-action combination
    #     for state_idx in range(len(self.state_set)):
    #         current_state = self.state_set[state_idx]
            
    #         for action_idx in range(len(self.action_set)):
    #             action = self.action_set[action_idx]
                
    #             # Calculate target state using simple addition (as LP solver does)
    #             target_state = current_state + action
                
    #             # Check if target state is out of bounds
    #             if (target_state[0] < 0 or target_state[0] > state_bounds['x_max'] or
    #                 target_state[1] < 0 or target_state[1] > state_bounds['y_max'] or
    #                 target_state[2] < 0 or target_state[2] > state_bounds['v_max']):
                    
    #                 problematic_combinations.append({
    #                     'state': current_state.tolist(),
    #                     'state_idx': state_idx,
    #                     'action': action.tolist(),
    #                     'action_idx': action_idx,
    #                     'target': target_state.tolist(),
    #                     'out_of_bounds_dims': []
    #                 })
                    
    #                 # Identify which dimensions are out of bounds
    #                 combo = problematic_combinations[-1]
    #                 if target_state[0] < 0 or target_state[0] > state_bounds['x_max']:
    #                     combo['out_of_bounds_dims'].append(f"x={target_state[0]} (bounds: [0,{state_bounds['x_max']}])")
    #                 if target_state[1] < 0 or target_state[1] > state_bounds['y_max']:
    #                     combo['out_of_bounds_dims'].append(f"y={target_state[1]} (bounds: [0,{state_bounds['y_max']}])")
    #                 if target_state[2] < 0 or target_state[2] > state_bounds['v_max']:
    #                     combo['out_of_bounds_dims'].append(f"v={target_state[2]} (bounds: [0,{state_bounds['v_max']}])")
        
    #     # Report results
    #     if problematic_combinations:
    #         print(f"⚠️  FOUND {len(problematic_combinations)} STATE+ACTION COMBINATIONS THAT LEAD OUT-OF-BOUNDS!")
    #         print("\nFirst 10 problematic combinations:")
    #         for i, combo in enumerate(problematic_combinations[:10]):
    #             print(f"  {i+1}: State {combo['state']} + Action {combo['action']} = {combo['target']}")
    #             print(f"      Out of bounds: {', '.join(combo['out_of_bounds_dims'])}")
            
    #         if len(problematic_combinations) > 10:
    #             print(f"  ... and {len(problematic_combinations) - 10} more problematic combinations")
                
    #         summary = f"PROBLEMATIC: Found {len(problematic_combinations)} state+action combinations leading out-of-bounds"
    #     else:
    #         print("✅ All state+action combinations stay within bounds")
    #         summary = "VALID: All state+action combinations stay within bounds"
        
    #     print("=== End State+Action Simple Addition Check ===\n")
        
    #     return {
    #         'valid': len(problematic_combinations) == 0,
    #         'problematic_combinations': problematic_combinations,
    #         'summary': summary
    #     }


class Abstraction_2:

    def __init__(self, route_size, route_res):
        self.route_res = route_res
        self.map_shape = None
        self.state_set = self.abs_state(route_size, route_res)
        self.action_set = self.abs_action()


    def abs_state(self, route_size, route_res):
        r_bl = 0
        r_bu = route_size[0]
        ey_bl = 0
        ey_bu = route_size[1]
        grid_x = np.arange(r_bl, r_bu, route_res[0])
        grid_y = np.arange(ey_bl, ey_bu, route_res[1])
        X, Y = np.meshgrid(grid_x, grid_y)
        self.map_shape = (len(grid_x), len(grid_y))
        return np.array([X.flatten(), Y.flatten()]).T

    def abs_action(self):
        vx_set = np.array([-1, 0, 1])
        # vy_set = np.array([-1, 0, 1])
        # vx_set = np.array([-2, -1, 0, 1, 2])
        vy_set = np.array([-2, -1, 0, 1, 2])
        A, B = np.meshgrid(vx_set, vy_set)
        return np.array([A.flatten(), B.flatten()]).T

    def get_state_index(self, abs_state):
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    def linear(self):
        # based on single integrator
        P = None
        for i in range(len(self.state_set)):
            P_s = None
            for n in range(len(self.action_set)):
                position = (i % self.map_shape[0], int(i / self.map_shape[0]))
                action = self.action_set[n]
                P_s_a = self.transition(position, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P


    def transition(self, position, action):
        def action_prob(action, std_dev, size):
            n = int(size / 2)
            x = np.linspace(-n, n, size)
            # gaussian_array = norm.pdf(x, 0, (abs(action) + 1) * std_dev)
            gaussian_array = norm.pdf(x, action,  std_dev)
            gaussian_array /= gaussian_array.sum()
            return gaussian_array

        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)
        prob_x = action_prob(action[0], 1.0, 5)
        prob_y = action_prob(action[1],  1.0, 5)
        prob_map = np.outer(prob_x, prob_y)
        k = int(len(prob_x) / 2)

        for m in range(len(prob_x)):
            for n in range(len(prob_y)):
                if (0 <= position[0] + m - k <= self.map_shape[0] -1) and (0 <= position[1] + n - k <= self.map_shape[1]-1):
                    P_sn[position[0] + m - k, position[1] + n - k] = prob_map[m, n]

        return P_sn.flatten(order='F')





if __name__ == '__main__':
    pcpt_range = (50, 10)
    pcpt_res = (5, 2)
    dt = 1
    initial_state = (2, 0, 0, 0)
    speed_range = 80
    speed_res = 10
    label_func = {(15, 20, 1, 3, 0, 80): "t",
                  (5, 15, -3, 1, 0, 80): "o",
                  (15, 20, 2, 3, 0, 80): "r"}

    abs_model = Abstraction(pcpt_range, pcpt_res, speed_range, speed_res, initial_state, label_func)
    MDP = abs_model.MDP
    print("action_set:", abs_model.action_set)
    print("transitions:", abs_model.MDP.transitions)
    print("labelling:", abs_model.MDP.labelling)
    print("initial_state:", abs_model.MDP.initial_state)