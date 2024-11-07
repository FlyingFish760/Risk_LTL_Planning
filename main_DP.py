# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: Nov 6nd, 2024

import matplotlib.pyplot as plt
import numpy as np

from abstraction.MDP import MDP
from sim.visualizer import Visualizer
import sim.simulator as sim
from risk_LP.LP import Risk_LTL_LP
from risk_LP.prod_auto import Product
from specification.specification import LTL_Spec
from abstraction.abstraction import Abstraction
from sim.low_level_controller import MPC
from abstraction.prod_MDP import Prod_MDP
from DP.policy_iteration import *



def main():

    # ---------- Environment Setting ---------------------
    params = {"dt": 0.05, "WB": 1.5}
    region_size = (20, 20)
    region_res = (5, 5)

    # ---------- Traffic Light --------------------
    traffic_light = ['g', 'r']
    state_set = range(len(traffic_light))
    action_set = [0]
    transitions = np.array([[[0.9, 0.1],[0.1, 0.9]]])
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, traffic_light, initial_state)

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~g -> ~c) & G(~o)", AP_set=['c', 'g', 'o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # ------------- Labelling -----------------------------

    label_func = {  (5, 10, 5, 10): "o",
                    (10, 20, 10, 15): "c",
                    (10, 15, 15, 20): "c",
                    (15, 20, 15, 20): "t"}

    # label_func = {(0, 5, 0, 25): "o",
    #                 (5, 25, 0, 5): "o",
    #                 (10, 15, 10, 15): "o",
    #                 (15, 25, 15, 20): "c",
    #                 (15, 20, 20, 25): "c",
    #                 (20, 25, 20, 25): "t"}
    cost_func = {"c": -5, "o": -10}

    # ---------- Visualization ---------------------
    fig = plt.figure()
    grid = plt.GridSpec(2, 2)
    ax_1 = fig.add_subplot(grid[0:, 0])
    plt.axis('off')
    plt.axis('equal')
    ax_2 = fig.add_subplot(grid[0, 1], projection='3d')
    ax_3 = fig.add_subplot(grid[1, 1])
    ax_3.axis('off') # risk measure profile
    vis = Visualizer(ax_1)

    # ---------- Initialization --------------------
    mpc_con = MPC(params, horizon_steps=5)
    ego_state = np.array([2.5, 2.5, np.pi / 2])
    ego_pos = ego_state[:2]
    prod_state = (0, 1, 1)
    abs_state_sys = [-1, -1]
    abs_state_env = 1

    oppo_car_state = np.array([12, 7.5, np.pi])
    oppo_car_pos = oppo_car_state[:2]

    abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
    # oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
    target_abs_state_sys = None

    mdp_sys = abs_model.MDP
    mdp_prod = Prod_MDP(mdp_sys, mdp_env)
    prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa, cost_func)
    value_function, policy = policy_iteration(prod_auto.prod_state_set,
                                              prod_auto.prod_action_set,
                                              prod_auto.product_transition,
                                              prod_auto.cost)

    # -------------- Simulation Loop --------------------
    iter = 1
    while True:
        iter += 1
        if iter == 150:
            abs_state_env = 0 # change the traffic light

        ax_1.cla()
        ax_2.cla()
        ax_1.set_aspect(1)
        ego_pos = ego_state[:2]
        # label_func = dyn_labelling(static_label, oppo_car_pos, [-1, 0])

        # ----------- Abstraction -----------------------------
        # abs_model.update_abs_state(ego_pos)
        # if (abs_state_sys != abs_model.init_abs_state): # replan only when the state changes
        #     abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
        #     mdp_sys = abs_model.MDP
        #     mdp_prod = Prod_MDP(mdp_sys, mdp_env)
        #     prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa, cost_func)
        #     cost_map = prod_auto.gen_cost_map(cost_func)
        #
        #     value_function, optimal_policy = policy_iteration(prod_auto.prod_state_set,
        #                                                       prod_auto.prod_action_set,
        #                                                       prod_auto.product_transition,
        #                                                       prod_auto.cost)

            # get the current product state
        abs_state_sys_index, abs_state_sys = abs_model.get_abs_ind_state(ego_pos)
        state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
        prod_state_index, prod_state  = prod_auto.update_prod_state(state_sys_env_index, prod_state)

        decision_index = max(policy[prod_state], key=policy[prod_state].get)
        sys_decision = abs_model.action_set[int(decision_index)]
        target_abs_state_sys = abs_state_sys + sys_decision

        target_point = target_abs_state_sys * 5 + np.array([2.5, 2.5])
        control_input = mpc_con.solve(ego_state, target_point)
        print("decision:", sys_decision)
        print("target_point", target_point)
        print("control_input:", control_input)


        ego_state = sim.car_dyn(ego_state, control_input, params)

        plt.gca().set_aspect(1)
        vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_car(ego_pos[0], ego_pos[1], ego_state[2], -control_input[1])
        plt.pause(0.001)


if __name__ == '__main__':
    main()