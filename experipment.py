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
from risk_LP.occ_measure_plot import plot_occ_measure

def cal_target_ref(target_abs_state, region_res, speed_res):
    target_ref_0 = target_abs_state[0] * region_res[0] + region_res[0] / 2
    target_ref_1 = target_abs_state[1] * region_res[1] + region_res[1] / 2 
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
    # ---------- MPD(Abstraction) --------------------
    region_size = (50, 20)
    region_res = (10,4)   
    max_speed = 4
    speed_res = 1

    # Speed limit test
    speed_limit = 3
    # (x1, x2, y1, y2, v1, v2) for the entries of label_func
    label_func = {(40, 50, 16, 20, 0, max_speed): "t",   # "t" for target
                   (10, 50, 0, 20, speed_limit, max_speed): "s", # "s" for "speeding"
                   (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed

    ego_state = np.array([0, 10, 0, 0])
    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~s) & G(~h)", AP_set=['s', 'h'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    cost_func = {'s': 3, 'h': 30}  

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
    abs_state_sys = abs_model.get_abs_state(ego_state) 

    # Initial product MDP state (the first state does not matter;
    #  the second state is the initial state of the cs DFA;
    #  the third state is the initial state of the safety DFA)
    prod_state = (0, 1, 1)   

    target_abs_state_sys = None
    occ_measure = None
    
    # ---------- Compute optimal policy --------------------
    mdp_sys = abs_model.MDP
    prod_auto = Product(mdp_sys, scltl_frag.dfa, safe_frag.dfa)

    # (Run online, please uncomment these lines)
    # LP_prob = Risk_LTL_LP(abs_model, prod_auto)
    # P_matrix = prod_auto.prod_transitions
    # cost_map = prod_auto.gen_cost_map(cost_func, abs_model)
    # abs_state_sys = abs_model.get_abs_state(ego_state)
    # abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
    # prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)
    # occ_measure = LP_prob.solve(P_matrix, cost_map, prod_state_index,
    #                             prod_auto.accepting_states, None)
    # optimal_policy, Z = LP_prob.extract(occ_measure)

    # (Run offline, please fill the policy_path below)
    # (Run online, please comment the three lines below)
    # Load a pre-comupte contorl policy
    policy_path = ""
    optimal_policy = np.load(policy_path)

    
    # -------------- Simulation Loop --------------------
    iter = 1
    first_decision_made = False

    while True:
        iter += 1

        ax_1.cla()
        ax_1.set_aspect(1)
        
        # Make decision
        if ((not first_decision_made) or                      # The first decision is not made
            (np.array_equal(abs_model.init_abs_state, target_abs_state_sys))):    # The ego vehicle finished the last action
            
            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)

            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            print("----------------------------------")
            print("abs_state_sys:", abs_state_sys)
            print("decision:", sys_decision)
            delta_state = action_to_state_transition(sys_decision)
            target_abs_state_sys = abs_state_sys + delta_state
            target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
            first_decision_made = True

        control_input = mpc_con.solve(ego_state, target_ref)

        ego_state = sim.car_dyn(ego_state, control_input, params)
        abs_model.update_abs_init_state(ego_state)

        plt.gca().set_aspect(1)
        vis.plot_grid(region_size, region_res, label_func)
        vis.plot_car(ego_state, -control_input[1])
        plt.pause(0.001)

        # print("------------new iter-------------")



if __name__ == '__main__':
    main()