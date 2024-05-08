# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: August 6nd, 2023

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from sim.visualizer import Visualizer
import sim.perception as pept
import sim.simulator as sim
from risk_LP.risk_LP import RiskLP
import risk_field.risk_field as rf
from risk_LP.abstraction import Abstraction


def main():

    # ---------- Environment Setting ---------------------
    road_size = [12, 80]
    square_obs_list = [[-6, 20, 6, 10], [0, 50, 6, 10]]
    params = {"dt": 0.1, "WB": 3.5}
    car_state = np.array([0, 0, np.pi/2])

    # ---------- Realtime States ---------------------
    fig = plt.figure()
    grid = plt.GridSpec(2, 2)
    ax_1 = fig.add_subplot(grid[0:, 0])
    plt.axis('off')
    plt.axis('equal')
    ax_2 = fig.add_subplot(grid[0, 1], projection='3d')
    ax_3 = fig.add_subplot(grid[1, 1])
    ax_3.axis('off')

    vis = Visualizer(ax_1, road_size, square_obs_list)
    LP_prob = RiskLP()

    pcpt_range = (18, 25)
    pcpt_res = (2, 5)

    X, Y =  rf.torus_meshgrid(pcpt_range, pcpt_res)
    plt.pause(5)

    while True:

        ax_1.cla()
        ax_2.cla()
        ax_1.set_aspect(1)
        car_pos = car_state[:2]

        pcpt_dic = pept.gen_pcpt_dic(road_size, square_obs_list)
        cost_map = pept.gen_cost_map(pcpt_dic, car_pos, pcpt_range, pcpt_res)
        print(cost_map)
        cost_map =  cost_map.flatten(order='F')

        abs_model = Abstraction(pcpt_range, pcpt_res)
        P_matrix = abs_model.linear()
        init_state = int(pcpt_range[0] / (2 * pcpt_res[0]))
        occ_measure = LP_prob.solve(P_matrix, cost_map, init_state)
        optimal_policy, Z = LP_prob.extract(occ_measure)
        Z = Z.reshape((int(pcpt_range[1]/pcpt_res[1]), int(pcpt_range[0]/pcpt_res[0])))

        plt.gca().set_aspect(1)
        vis.plot_boundary_lines()
        vis.plot_obstacle()
        vis.plot_perception((car_pos[0], car_pos[1]), pcpt_range)
        vis.plot_car(car_pos[0], car_pos[1], car_state[2], 0)
        ax_2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # policy_iter = PolicyIteration(risk_map, discount_factor=0.8)
        # optimal_policy = policy_iter.run()
        # action = optimal_policy[int(pcpt_range[0]/(2 * pcpt_res))+1, 0]
        action_index = optimal_policy[init_state]
        action = abs_model.action_set[int(action_index)]

        print("action:", action)
        for i in range(5):
            car_state = sim.update_1(car_state, action, params)
        print("state:", car_state)
        plt.pause(0.001)


if __name__ == '__main__':
    main()