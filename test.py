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

    # static_label = {(15, 20, 15, 20): "t"}  
    # label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
    speed_limit = 30
    label_func = {(15, 20, 5, 10, 0, max_speed): "t",
                   (5, 20, 5, 10, speed_limit, max_speed): "o",   # "o" for "overspeed"
                   (0, 20, 0, 20, 0, SPEED_LOW_BOUND): "s"}    # "s" for "slow"
    
    ego_state = np.array([2.5, 12.5, np.pi / 2, 0])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    cost_func = {'o': 1, 's': 0.5}  


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

    # ---------- Initialization --------------------
    abs_state_env = 0
    abs_state_sys = abs_model.get_abs_state(ego_state) 
    # oppo_abs_state = abs_model.get_abs_state(init_oppo_car_pos)  

    prod_state = (0, 1, 1)   # Initial product MDP state

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
            mdp_prod = Prod_MDP(mdp_sys, mdp_env)
            prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
            P_matrix = prod_auto.prod_transitions
            cost_map = prod_auto.gen_cost_map(cost_func)
            
            # print(f"Debug: cost_map shape={cost_map.shape}")
            # print(f"Debug: cost_map min={np.min(cost_map)}, max={np.max(cost_map)}")
            # print(f"Debug: cost_map non-zero elements={np.count_nonzero(cost_map)}")
            # print(f"Debug: cost_map sample values={cost_map[:10]}")

            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            # oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
            state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
            prod_state_index, prod_state  = prod_auto.update_prod_state(state_sys_env_index, prod_state)
            
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
            target_abs_state_sys = abs_state_sys + sys_decision


        target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
        control_input = mpc_con.solve(ego_state, target_ref)
        # print("abs_state_sys:", abs_state_sys)
        # print("decision:", sys_decision)
        # print("target_point", target_point)
        print("control_input:", control_input)
        # print("ego_pos:", ego_pos)

        if iter == 120:
            print('d')

        # Updated to handle 4D state and 2D control [acceleration, steering]
        ego_state = sim.car_dyn(ego_state, control_input, params)
        abs_model.update_abs_init_state(ego_state)

        ego_pos = ego_state[:2]
        plt.gca().set_aspect(1)
        vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_car(ego_pos[0], ego_pos[1], ego_state[2], -control_input[1])
        plt.pause(0.001)

        print("------------new iter-------------")

if __name__ == '__main__':
    main()