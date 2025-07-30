#!/usr/bin/env python
# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: January 6nd, 2025
# Optimized version - caches static components
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
    speed_limit, speed_pos = speed_sign
    return labels

def cal_target_ref(target_abs_state, region_res, speed_res):
    target_ref_0 = target_abs_state[0] * region_res[0] + region_res[0] / 2
    target_ref_1 = target_abs_state[1] * region_res[1] + region_res[1] / 2
    target_ref_2 = target_abs_state[2] * speed_res + speed_res / 2
    return np.array([target_ref_0, target_ref_1, target_ref_2])


def main():
    # ---------- MDP Environment  ---------------------
    # Create a trivial MDP with one state and no actions
    state_set = [0]
    action_set = [0]
    transitions = np.array([[[1.0]]])  # Stay in state 0
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, ["none"], initial_state)

    # ---------- MPD System (Abstraction) --------------------
    region_size = (20, 20)
    region_res = (5, 5)
    max_speed = 80
    speed_res = 10

    speed_limit = 30
    label_func = {(15, 20, 5, 10, 0, max_speed): "t",
                   (5, 20, 5, 10, speed_limit, max_speed): "o"}   # "o" for "overspeed"
    
    ego_state = np.array([2.5, 12.5, np.pi / 2, 0])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    cost_func = {'o': 1}  

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

    # ---------- OPTIMIZATION: Pre-compute static components --------------------
    print("Pre-computing static components...")
    
    # These components never change during simulation
    mdp_sys = abs_model.MDP
    mdp_prod = Prod_MDP(mdp_sys, mdp_env)
    prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
    P_matrix = prod_auto.prod_transitions
    cost_map = prod_auto.gen_cost_map(cost_func)
    
    print(f"Static components computed:")
    print(f"  - Product MDP states: {len(mdp_prod.prod_state_set)}")
    print(f"  - Product automaton states: {len(prod_auto.prod_state_set)}")
    print(f"  - P_matrix shape: {P_matrix.shape}")
    print(f"  - Cost map shape: {cost_map.shape}")
    print()

    # ---------- Initialization --------------------
    abs_state_env = 0
    abs_state_sys = abs_model.get_abs_state(ego_state) 
    prod_state = (64, 1, 2)   # Initial product MDP state
    target_abs_state_sys = None
    occ_measure = None
    
    # -------------- Simulation Loop --------------------
    iter = 1

    while True:
        iter += 1
        if iter == 150:
            abs_state_sys = [-1, -1]   # force re-plan

        ax_1.cla()
        ax_1.set_aspect(1)
        
        # OPTIMIZED: Only recalculate dynamic components
        if ((abs_state_sys != abs_model.init_abs_state) or   # The ego vehicle moves to a state
                (occ_measure is None)):   # There's no occupation measure
            
            print(f"Re-planning at iteration {iter}...")
            
            # Only recalculate the dynamic parts
            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
            prod_state_index, prod_state = prod_auto.update_prod_state(state_sys_env_index, prod_state)
            
            print(f"  - Current abstract state: {abs_state_sys}")
            print(f"  - Product state index: {prod_state_index}")
            
            # Solve LP with pre-computed static components
            occ_measure = LP_prob.solve(P_matrix, cost_map, prod_state_index,
                                        prod_auto.accepting_states, None)
            optimal_policy, Z = LP_prob.extract(occ_measure)
            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            target_abs_state_sys = abs_state_sys + sys_decision
            
            print(f"  - Optimal action: {sys_decision}")
            print(f"  - Target state: {target_abs_state_sys}")

        target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
        control_input = mpc_con.solve(ego_state, target_ref)

        if iter == 120:
            print('d')

        # Update ego state
        ego_state = sim.car_dyn(ego_state, control_input, params)
        abs_model.update_abs_init_state(ego_state)

        # Visualization
        ego_pos = ego_state[:2]
        plt.gca().set_aspect(1)
        vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_car(ego_pos[0], ego_pos[1], ego_state[2], -control_input[1])
        plt.pause(0.001)

        print()

if __name__ == '__main__':
    main() 