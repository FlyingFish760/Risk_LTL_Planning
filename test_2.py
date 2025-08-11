# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: January 6nd, 2025
import matplotlib.pyplot as plt
import numpy as np

from abstraction.MDP import MDP
from sim.visualizer import Visualizer
import sim.simulator as sim
from risk_LP.ltl_risk_LP import Risk_LTL_LP
from specification.prod_auto import Product
from specification.specification import LTL_Spec
from abstraction.abstraction import Abstraction
from sim.controller import MPC
from abstraction.prod_MDP import Prod_MDP


SPEED_LOW_BOUND = 10


def dyn_labelling(sta_labels, speed_sign):
    labels = sta_labels.copy()

    # g_x = int(np.floor(v_pos[0] / region_res[0]))   # Convert postion of opposite vehicle to grid index
    # g_y = int(np.floor(v_pos[1] / region_res[1]))
    # v_1 = ((g_x + min(0, v_action[0])) * region_res[0],   # The next cell the opponent is moving into
    #         (g_x + max(1, 1 + v_action[0])) * region_res[0],
    #         (g_y + min(0, v_action[1])) * region_res[1], 
    #         (g_y + max(1, 1 + v_action[1])) * region_res[1])
    # v_0 = (g_x * region_res[0],    # Current opponent cell (this is the immediate danger zone)
    #         (g_x + 1) * region_res[0], 
    #         g_y * region_res[1], 
    #         (g_y + 1) * region_res[1])
    # labels[v_0] = "v0"
    # labels[v_1] = "v1"

    speed_limit, speed_pos = speed_sign

    return labels

def cal_target_ref(target_abs_state, region_res, speed_res):
    target_ref_0 = target_abs_state[0] * region_res[0] + region_res[0] / 2
    target_ref_1 = target_abs_state[1] * region_res[1]   # no puls region_res[0] / 2, becuase of the symmetric ey coordinate
    target_ref_2 = target_abs_state[2] * speed_res + speed_res / 2
    return np.array([target_ref_0, target_ref_1, target_ref_2])

def action_to_state_transition(action):
    '''
    action: (m, s). m is the movement action, s is the speeding action. 
    m in {'l', 'r', 'f'}, s in {'a', 'd', 'c'}.

    return: delta_state = (delta_r, delta_ey, delta_v)
    '''
    delta_state = np.array([0, 0, 0])

    # movement action
    delta_state[0] = 1
    if action[0] == 'l':
        delta_state[1] = 1
    elif action[0] == 'r':
        delta_state[1] = -1
    elif action[0] == 'f':
        delta_state[1] = 0

    # speeding action
    if action[1] == 'a':
        delta_state[2] = 1
    elif action[1] == 'd':
        delta_state[2] = -1
    elif action[1] == 'c':
        delta_state[2] = 0

    return delta_state
        






def main():
    # # ---------- MDP Environment  ---------------------
    # # Create a trivial MDP with one state and no actions
    # state_set = [0]
    # action_set = [0]
    # transitions = np.array([[[1.0]]])  # Stay in state 0
    # initial_state = 0
    # mdp_env = MDP(state_set, action_set, transitions, ["e"], initial_state)

    # ---------- MPD System (Abstraction) --------------------
    region_size = (50, 20)
    region_res = (10,4)   # Now region_size[1]/region_res[1] must be odd
    max_speed = 10
    speed_res = 2

    # static_label = {(15, 20, 15, 20): "t"}  
    # label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
    # speed_limit = 30

    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    #basic
    label_func = {(30, 40, 6, 10, 0, max_speed): "t",
                   (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    # # overspeed test 1
    # speed_limit = 6
    # label_func = {(30, 40, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, -2, 10, speed_limit, max_speed): "s"}   # "s" for "overspeed"
    
    ego_state = np.array([0, 0, np.pi / 2, 0])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # basic
    cost_func = {'o': 5}  

    # # overspeed test 1
    # cost_func = {'o': 5, 's': 20}  

    # ---------- LP problem --------------------
    LP_prob = Risk_LTL_LP()

    # ---------- MPC --------------------
    params = {"dt": 0.05, "WB": 1.5}
    mpc_con = MPC(params, horizon_steps=5)

    # ---------- Visualization ---------------------
    fig, ax_1 = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis('off')
    plt.axis('equal')
    vis = Visualizer(ax_1)
    vis.set_velocity_params(speed_res)

    # ---------- Initialization --------------------
    # abs_state_env = 0
    abs_state_sys = abs_model.get_abs_state(ego_state) 
    # oppo_abs_state = abs_model.get_abs_state(init_oppo_car_pos)  

    # Initial product MDP state 
    # (the first state does not matter;
    #  the second state is the initial state of the cs DFA;
    #  the third state is the initial state of the safety DFA)
    prod_state = (0, 1, 1)   

    target_abs_state_sys = None
    occ_measure = None
    

    
    # -------------- Simulation Loop --------------------
    iter = 1

    while True:
        iter += 1
        if iter == 150:
            # Modified: Remove traffic light state change
            # abs_state_env = 0 # change the traffic light
            abs_state_sys = [-1, -1]   # force re-plan

        ax_1.cla()
        ax_1.set_aspect(1)
        
        
        if ((abs_state_sys != abs_model.init_abs_state) or   # The ego vehicle moves to a state
               # (oppo_abs_state != abs_model.get_abs_state(oppo_car_pos)) or   # The opponent vehicle moves to a state
                (occ_measure is None)):   # There's no occupation measure
            # abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
            mdp_sys = abs_model.MDP
            # mdp_prod = Prod_MDP(mdp_sys, mdp_env)
            # prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
            prod_auto = Product(mdp_sys, scltl_frag.dfa, safe_frag.dfa)
            P_matrix = prod_auto.prod_transitions
            cost_map = prod_auto.gen_cost_map(cost_func)
            
            # # Check the cost map for potential issues
            # print("Checking cost map...")
            # cost_map_result = prod_auto.check_cost_map(cost_func)
            
            # # Debug: Show some example labels to understand the current state
            # print("\n=== Debug: Product MDP Label Examples (After Fix) ===")
            # print("Environment MDP label:", mdp_env.labelling[0])
            # print("First 10 product MDP labels:")
            # for i in range(min(10, len(mdp_prod.MDP.labelling))):
            #     prod_state = mdp_prod.prod_state_set[i]
            #     x_sys, x_env = prod_state
            #     sys_label = mdp_sys.labelling[x_sys]
            #     env_label = mdp_env.labelling[x_env]
            #     combined_label = mdp_prod.MDP.labelling[i]
            #     print(f"  State {i}: sys='{sys_label}' + env='{env_label}' = '{combined_label}'")
            # print("=== End Debug ===\n")
            
            # # Check if the critical bug is fixed
            # print("=== Checking if the critical bug is fixed ===")
            # test_labels = ['_e', 'te', 'oe']  # New concatenated labels with 'e'
            
            # for test_label in test_labels:
            #     # Check how the safety DFA (G(~o)) processes this label
            #     safety_alphabet = safe_frag.dfa.get_alphabet(test_label)
            #     cs_alphabet = scltl_frag.dfa.get_alphabet(test_label)
                
            #     print(f"Label '{test_label}':")
            #     print(f"  Safety DFA (G(~o)) alphabet: {safety_alphabet}")
            #     print(f"  CS DFA (F(t)) alphabet: {cs_alphabet}")
            #     print(f"  Contains 'o'? {'o' in test_label}")
            #     print(f"  Contains 't'? {'t' in test_label}")
            #     print()
            
            # # Count states that still have false obstacle detection
            # affected_states = []
            # for i, label in enumerate(mdp_prod.MDP.labelling):
            #     if 'o' in label:
            #         prod_state = mdp_prod.prod_state_set[i]
            #         x_sys, x_env = prod_state
            #         sys_label = mdp_sys.labelling[x_sys]
                    
            #         # Check if the system part actually has obstacle 'o'
            #         has_real_obstacle = 'o' in sys_label
                    
            #         if not has_real_obstacle:
            #             # This is a false positive - 'o' comes from somewhere else
            #             affected_states.append({
            #                 'prod_state_idx': i,
            #                 'label': label,
            #                 'sys_label': sys_label,
            #                 'env_label': mdp_env.labelling[x_env],
            #                 'has_real_obstacle': has_real_obstacle
            #             })
            
            # print(f"States with false obstacle detection: {len(affected_states)}")
            # if affected_states:
            #     print("First 5 affected states:")
            #     for i, state in enumerate(affected_states[:5]):
            #         print(f"  {i+1}: State {state['prod_state_idx']}: '{state['label']}' (sys='{state['sys_label']}', env='{state['env_label']}')")
            # else:
            #     print("✅ No false obstacle detection found!")
            
            # print("=== End Bug Check ===\n")
            
            # if not cost_map_result['valid']:
            #     print(f"WARNING: Found issues with cost map!")
            #     for issue in cost_map_result['issues']:
            #         print(f"  - {issue['type']}: {issue['message']}")
            
            # # Check the abstraction P matrix for invalid transitions
            # print("Checking abstraction P matrix bounds...")
            # p_matrix_result = abs_model.check_P_matrix_bounds()
            # state_action_result = abs_model.check_state_action_simple_addition()
            
            # if not p_matrix_result['valid']:
            #     print(f"WARNING: Found {len(p_matrix_result['invalid_transitions'])} invalid transitions in abstraction P matrix!")
            
            # if not state_action_result['valid']:
            #     print(f"WARNING: Found {len(state_action_result['problematic_combinations'])} problematic state+action combinations!")
            #     print("This is likely the cause of the '[0, 2, -1] is not in list' error")
            
            # # Check the product automaton P matrix for invalid transitions
            # print("Checking product automaton P matrix bounds...")
            # prod_consistency_result = prod_auto.check_product_state_consistency()
            # prod_bounds_result = prod_auto.check_product_P_matrix_bounds(abs_model, mdp_prod)
            # prod_mdp_sys_bounds_result = prod_auto.check_product_mdp_sys_bounds(abs_model, mdp_prod)
            
            # if not prod_consistency_result['consistent']:
            #     print(f"WARNING: Found {len(prod_consistency_result['inconsistent_states'])} inconsistent product states!")
            
            # if not prod_bounds_result['valid']:
            #     print(f"WARNING: Found {len(prod_bounds_result['invalid_transitions'])} invalid transitions in product automaton!")
            
            # if not prod_mdp_sys_bounds_result['valid']:
            #     print(f"WARNING: Found {len(prod_mdp_sys_bounds_result['invalid_sys_transitions'])} invalid mdp_sys transitions in product automaton!")
            #     print("This indicates out-of-bounds mdp_sys states in the product automaton!")
            
            # print(f"Debug: cost_map shape={cost_map.shape}")
            # print(f"Debug: cost_map min={np.min(cost_map)}, max={np.max(cost_map)}")
            # print(f"Debug: cost_map non-zero elements={np.count_nonzero(cost_map)}")
            # print(f"Debug: cost_map sample values={cost_map[:10]}")

            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            # oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
            # state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
            # prod_state_index, prod_state  = prod_auto.update_prod_state(state_sys_env_index, prod_state)
            prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)
            
            # print(f"Debug: ego_state={ego_state}")
            # print(f"Debug: abs_state_sys={abs_state_sys}")
            # print(f"Debug: abs_state_sys_index={abs_state_sys_index}")
            # print(f"Debug: state_sys_env_index={state_sys_env_index}")
            # print(f"Debug: prod_state_index={prod_state_index}")
            # print(f"Debug: prod_state={prod_state}")
            
            occ_measure = LP_prob.solve(P_matrix, cost_map, prod_state_index,
                                        prod_auto.accepting_states, None)
            optimal_policy, Z = LP_prob.extract(occ_measure)
            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            print("abs_state_sys:", abs_state_sys)
            print("decision:", sys_decision)
            delta_state = action_to_state_transition(sys_decision)
            target_abs_state_sys = abs_state_sys + delta_state
            
        target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
        print("target_ref:", target_ref)
        control_input = mpc_con.solve(ego_state, target_ref)
        
        # print("target_point", target_point)
        # print("control_input:", control_input)
        # print("ego_pos:", ego_pos)

        if iter == 120:
            print('d')


        ego_state = sim.car_dyn(ego_state, control_input, params)
        print("ego_state:", ego_state)
        abs_model.update_abs_init_state(ego_state)

        ego_pos = ego_state[:2]
        plt.gca().set_aspect(1)
        # vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_grid(region_size, region_res, label_func)
        vis.plot_car(ego_state, -control_input[1])
        plt.pause(0.001)

        print("------------new iter-------------")










# def check_P_matrix_bounds(P_matrix, abs_model, prod_auto):
#     """Check if P_matrix allows transitions to invalid states"""
#     print("=== Checking Product Automaton P_matrix for out-of-bounds transitions ===")
    
#     # Get state space bounds for the original MDP part
#     state_bounds = {
#         'x_max': abs_model.map_range[0] // abs_model.map_res[0] - 1,
#         'y_max': abs_model.map_range[1] // abs_model.map_res[1] - 1, 
#         'v_max': abs_model.speed_range // abs_model.speed_res - 1
#     }
#     print(f"MDP State bounds: x=[0,{state_bounds['x_max']}], y=[0,{state_bounds['y_max']}], v=[0,{state_bounds['v_max']}]")
    
#     num_prod_states = len(prod_auto.prod_state_set)
#     num_actions = len(prod_auto.prod_action_set)
    
#     print(f"Product automaton has {num_prod_states} states and {num_actions} actions")
    
#     invalid_transitions = []
    
#     for prod_state_idx in range(num_prod_states):
#         # Product state is (mdp_state_idx, dfa1_state, dfa2_state)
#         prod_state = prod_auto.prod_state_set[prod_state_idx]
#         mdp_state_idx, dfa1_state, dfa2_state = prod_state
        
#         # Get the actual MDP state (x, y, v coordinates)
#         current_mdp_state = abs_model.state_set[mdp_state_idx]
        
#         for action_idx in range(num_actions):
#             action = abs_model.action_set[action_idx]  # This should be the MDP action
            
#             # Get transition probabilities for this product state-action pair
#             trans_probs = P_matrix[prod_state_idx, action_idx, :]
            
#             # Check each possible next product state
#             for next_prod_state_idx in range(num_prod_states):
#                 if trans_probs[next_prod_state_idx] > 0:  # Non-zero transition probability
#                     next_prod_state = prod_auto.prod_state_set[next_prod_state_idx]
#                     next_mdp_state_idx, next_dfa1_state, next_dfa2_state = next_prod_state
                    
#                     # Get the actual next MDP state coordinates
#                     next_mdp_state = abs_model.state_set[next_mdp_state_idx]
                    
#                     # Check if next MDP state is within bounds
#                     if (next_mdp_state[0] < 0 or next_mdp_state[0] > state_bounds['x_max'] or
#                         next_mdp_state[1] < 0 or next_mdp_state[1] > state_bounds['y_max'] or
#                         next_mdp_state[2] < 0 or next_mdp_state[2] > state_bounds['v_max']):
                        
#                         invalid_transitions.append({
#                             'from_prod_state': prod_state,
#                             'from_mdp_state': current_mdp_state,
#                             'action': action,
#                             'to_prod_state': next_prod_state,
#                             'to_mdp_state': next_mdp_state,
#                             'probability': trans_probs[next_prod_state_idx],
#                             'prod_state_idx': prod_state_idx,
#                             'action_idx': action_idx,
#                             'next_prod_state_idx': next_prod_state_idx
#                         })
    
#     if invalid_transitions:
#         print(f"FOUND {len(invalid_transitions)} INVALID TRANSITIONS IN PRODUCT AUTOMATON!")
#         for i, trans in enumerate(invalid_transitions[:5]):  # Show first 5
#             print(f"  {i+1}: MDP State {trans['from_mdp_state']} --{trans['action']}--> MDP State {trans['to_mdp_state']} (prob={trans['probability']:.4f})")
#             print(f"      Product: {trans['from_prod_state']} --> {trans['to_prod_state']}")
#         if len(invalid_transitions) > 5:
#             print(f"  ... and {len(invalid_transitions) - 5} more")
#     else:
#         print("No invalid MDP transitions found in Product Automaton P_matrix")
    
#     # Also check if there are MDP states that can lead to out-of-bounds via simple addition
#     print("\n=== Checking MDP state+action combinations that would be out-of-bounds ===")
#     problematic_combinations = []
    
#     for mdp_state_idx in range(len(abs_model.state_set)):
#         current_mdp_state = abs_model.state_set[mdp_state_idx]
        
#         for action_idx in range(len(abs_model.action_set)):
#             action = abs_model.action_set[action_idx]
#             target_state = current_mdp_state + action
            
#             if (target_state[0] < 0 or target_state[0] > state_bounds['x_max'] or
#                 target_state[1] < 0 or target_state[1] > state_bounds['y_max'] or
#                 target_state[2] < 0 or target_state[2] > state_bounds['v_max']):
                
#                 problematic_combinations.append({
#                     'mdp_state': current_mdp_state,
#                     'action': action, 
#                     'target': target_state,
#                     'mdp_state_idx': mdp_state_idx,
#                     'action_idx': action_idx
#                 })
    
#     if problematic_combinations:
#         print(f"FOUND {len(problematic_combinations)} MDP STATE+ACTION COMBINATIONS THAT LEAD OUT-OF-BOUNDS!")
#         for i, combo in enumerate(problematic_combinations[:10]):  # Show first 10
#             print(f"  {i+1}: MDP State {combo['mdp_state']} + Action {combo['action']} = {combo['target']} (OUT OF BOUNDS)")
#         if len(problematic_combinations) > 10:
#             print(f"  ... and {len(problematic_combinations) - 10} more")
#     else:
#         print("No problematic MDP state+action combinations found")
    
#     print("=== End Product Automaton P_matrix bounds check ===\n")
#     return invalid_transitions, problematic_combinations













if __name__ == '__main__':
    main()
    # sys_decision = ['l', 'd']
    # abs_state_sys = [0, 0, 0]
    # delta_state = action_to_state_transition(sys_decision)
    # target_abs_state_sys = abs_state_sys + delta_state
    # print("target_abs_state_sys:", target_abs_state_sys)
    # target_ref = cal_target_ref(target_abs_state_sys, (10, 4), 10) 
    # print("target_ref:", target_ref)