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

# import csv
# import datetime
# import os


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
    max_speed = 4
    speed_res = 1

    # region_size = (30, 30)
    # region_res = (10,10)   # Now region_size[1]/region_res[1] must be odd
    # max_speed = 3
    # speed_res = 1

    # static_label = {(15, 20, 15, 20): "t"}  
    # label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
    # speed_limit = 30

    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    #basic
    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o"}   # "o" for "obstacle"

    # # speed test 1
    # speed_limit = 2
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, 8, 20, speed_limit, max_speed): "s"}   # "s" for "overspeed"

    # # speed test 2
    # speed_limit = 4
    # speed_limit_2 = 4
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, 8, 20, speed_limit, max_speed): "s", # "s" for "speed"
    #                (30, 40, 8, 20, 0, speed_limit): "s",
    #                (0, 50, 0, 20, -speed_res, 0): "n",   # "n" for negative speed 
    #                (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed

    # # speed test 3
    # speed_limit = 4
    # speed_limit_2 = 4
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, 8, 20, speed_limit, max_speed): "s", # "s" for "speed"
    #                (30, 40, 8, 20, 0, speed_limit): "s",
    #                (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed

    # speed test 3
    speed_limit = 3
    label_func = {(40, 50, 16, 20, 0, max_speed): "t",
                   (10, 50, 0, 20, speed_limit, max_speed): "s", # "s" for "speed"
                   (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed
    
    ego_state = np.array([0, 10, 0, 0])

    # # speed test 4
    # speed_limit = 2
    # label_func = {(20, 30, 20, 30, 0, max_speed): "t",
    #                (10, 30, 0, 30, speed_limit, max_speed): "s", # "s" for "speed"
    #                (0, 30, 0, 30, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed
    
    # ego_state = np.array([0, 10, 0, 0])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    # safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    # safe_frag = LTL_Spec("G(~o) & G(~s)", AP_set=['o', 's'])
    # safe_frag = LTL_Spec("G(~o) & G(~s) & G(~n) & G(~h)", AP_set=['o', 's', 'n', 'h'])
    safe_frag = LTL_Spec("G(~o) & G(~s) & G(~h)", AP_set=['o', 's', 'h'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # basic
    # cost_func = {'o': 5}
    # cost_func = {'o': 5, 's': 5}  
    # cost_func = {'o': 5, 's': 3, 'n': 30, 'h': 30}  
    cost_func = {'o': 5, 's': 3, 'h': 30}  

    # # overspeed test 1
    # cost_func = {'o': 5, 's': 20}  

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
    
    # ---------- Compute optimal policy --------------------
    mdp_sys = abs_model.MDP
    prod_auto = Product(mdp_sys, scltl_frag.dfa, safe_frag.dfa)

    # LP_prob = Risk_LTL_LP(abs_model, prod_auto)
    # P_matrix = prod_auto.prod_transitions
    # cost_map = prod_auto.gen_cost_map(cost_func, abs_model)
    # abs_state_sys = abs_model.get_abs_state(ego_state)
    # abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
    # prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)
    # occ_measure = LP_prob.solve(P_matrix, cost_map, prod_state_index,
    #                             prod_auto.accepting_states, None)
    # plot_occ_measure(occ_measure, prod_auto, abs_model)

    # optimal_policy, Z = LP_prob.extract(occ_measure)
    # np.save("optimal_policy_risk_aware2.npy", optimal_policy)

    policy_path = "/home/student/Risk_LTL_Planning/optimal_policy_risk_aware2.npy"
    optimal_policy = np.load(policy_path)

    # # --- log directory ---
    # log_dir = os.path.expanduser("~/ego_logs")   # goes into your home folder
    # os.makedirs(log_dir, exist_ok=True)

    # # --- timestamped file ---
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_filename = os.path.join(log_dir, f"ego_state_log_{timestamp}.csv")

    # # --- create CSV with header immediately ---
    # with open(log_filename, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["time", "pos_x", "pos_y", "yaw", "velocity", "acce", "delta_theta"])

    # # --- running time counter ---
    # running_time = 0.0
    
    # -------------- Simulation Loop --------------------
    iter = 1
    first_decision_made = False

    while True:
        iter += 1
        # if iter == 150:
        #     # Modified: Remove traffic light state change
        #     # abs_state_env = 0 # change the traffic light
        #     abs_state_sys = [-1, -1]   # force re-plan

        ax_1.cla()
        ax_1.set_aspect(1)
        
        
        # if ((abs_state_sys != abs_model.init_abs_state) or   # The ego vehicle state changes
        #        # (oppo_abs_state != abs_model.get_abs_state(oppo_car_pos)) or   # The opponent vehicle moves to a state
        #         (occ_measure is None)):   # There's no occupation measure
        
        # Make decision
        if ((not first_decision_made) or                      # The first decision is not made
            (np.array_equal(abs_model.init_abs_state, target_abs_state_sys))):    # The ego vehicle finished the last action
            
            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)

            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            print("abs_state_sys:", abs_state_sys)
            print("decision:", sys_decision)
            delta_state = action_to_state_transition(sys_decision)
            target_abs_state_sys = abs_state_sys + delta_state
            # print("target_abs_state_sys:", target_abs_state_sys)
            target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
            first_decision_made = True

        control_input = mpc_con.solve(ego_state, target_ref)
        
        # print("target_point", target_point)
        # print("control_input:", control_input)
        # print("ego_pos:", ego_pos)

        if iter == 120:
            print('d')


        ego_state = sim.car_dyn(ego_state, control_input, params)
        # print("ego_state:", ego_state)
        abs_model.update_abs_init_state(ego_state)

        # # --- log to CSV ---
        # with open(log_filename, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([running_time, ego_state[0], ego_state[1], ego_state[2], ego_state[3], control_input[0], control_input[1]])

        # # --- update running time ---
        # running_time += params["dt"]


        ego_pos = ego_state[:2]
        plt.gca().set_aspect(1)
        # vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_grid(region_size, region_res, label_func)
        vis.plot_car(ego_state, -control_input[1])
        plt.pause(0.001)

        # print("------------new iter-------------")



if __name__ == '__main__':
    main()